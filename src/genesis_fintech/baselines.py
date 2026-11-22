from __future__ import annotations

import json
import random
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


def _paginate(env: Any, table: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str | None] = set()
    while True:
        if cursor in seen_cursors:
            raise RuntimeError(f"Pagination cursor repeated while reading {table}")
        seen_cursors.add(cursor)
        obs, _, term, trunc, _ = env.step({"name": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}})
        if term or trunc:
            raise RuntimeError(f"Episode ended while reading {table}")
        if obs.get("error") is not None:
            raise RuntimeError(f"Failed to read {table}: {obs['error']}")
        res = obs.get("result") or {}
        out.extend(res.get("records", []))
        cursor = res.get("next_cursor")
        if cursor is None:
            break
    return out


def run_random_policy(env: Any, init_obs: dict[str, Any], seed: int = 42, max_steps: int = 10) -> dict[str, Any]:
    rng = random.Random(seed)
    eid = init_obs.get("manifest", {}).get("episode_id", "")
    invs = _paginate(env, "invoices")
    for _ in range(max(1, max_steps - 2)):
        if invs:
            _, _, term, trunc, _ = env.step({"name": "get_document", "args": {"doc_id": rng.choice(invs)["doc_id"]}})
        else:
            _, _, term, trunc, _ = env.step({"name": "calculator", "args": {"expression": "10 + 20"}})
        if term or trunc:
            return {"episode_id": eid, "findings": []}
    payload = {"episode_id": eid, "findings": []}
    env.step({"name": "submit", "args": {"findings": payload}})
    return payload


def run_simple_reconciliation_policy(env: Any, init_obs: dict[str, Any]) -> dict[str, Any]:
    eid = init_obs.get("manifest", {}).get("episode_id", "")
    invs = _paginate(env, "invoices")
    pays = _paginate(env, "payments")
    ledger = _paginate(env, "ledger")
    fx_rows = _paginate(env, "fx")
    findings: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, int, str, str], list[dict[str, Any]]] = {}
    for invoice in invs:
        key = (
            invoice["entity_id"],
            invoice["issued_at"],
            invoice["due_at"],
            int(invoice["total_minor"]),
            invoice["currency"],
            json.dumps(invoice["line_items"], sort_keys=True, separators=(",", ":")),
        )
        groups.setdefault(key, []).append(invoice)
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda invoice: invoice["issued_at"])
        original = group[0]
        for duplicate in group[1:]:
            findings.append({
                "type": "duplicate_invoice",
                "affected_ids": [original["invoice_id"], duplicate["invoice_id"]],
                "claimed_amount_minor": int(duplicate["total_minor"]),
                "currency": duplicate["currency"],
                "evidence_ids": [original["doc_id"], duplicate["doc_id"]],
                "explanation": "Invoices have the same entity, amount, and currency.",
            })

    base_currency = str(init_obs.get("manifest", {}).get("base_currency", ""))
    period_start = str(init_obs.get("manifest", {}).get("period_start", ""))[:10]
    period_end = str(init_obs.get("manifest", {}).get("period_end", ""))[:10]
    fx_rates = {
        (row["date"], row["source_currency"], row["target_currency"]): Decimal(str(row["rate"]))
        for row in fx_rows
    }
    ar_rows: dict[str, list[dict[str, Any]]] = {}
    for row in ledger:
        if row.get("account") == "ACCOUNTS_RECEIVABLE" and int(row["debit_minor"]) > 0 and row.get("invoice_id"):
            ar_rows.setdefault(row["invoice_id"], []).append(row)

    for invoice in invs:
        invoice_id = invoice["invoice_id"]
        rows = ar_rows.get(invoice_id, [])
        if not rows:
            continue
        ar_row = rows[0]
        issued_date = str(invoice["issued_at"])[:10]
        if invoice["currency"] != base_currency:
            rate = fx_rates.get((issued_date, invoice["currency"], base_currency))
            if rate is not None:
                expected_minor = int(
                    (Decimal(int(invoice["total_minor"])) * rate).quantize(
                        Decimal("1"), rounding=ROUND_HALF_UP
                    )
                )
                difference = abs(int(ar_row["debit_minor"]) - expected_minor)
                if difference:
                    findings.append({
                        "type": "bad_fx_date_or_rate",
                        "affected_ids": [invoice_id],
                        "claimed_amount_minor": difference,
                        "currency": base_currency,
                        "evidence_ids": [invoice["doc_id"], ar_row["row_id"]],
                        "explanation": "The posted base amount differs from the invoice amount converted at the listed issue-date rate.",
                    })

        ledger_date = str(ar_row["timestamp"])[:10]
        if issued_date < period_start and period_start <= ledger_date <= period_end:
            findings.append({
                "type": "period_misclassification",
                "affected_ids": [invoice_id],
                "claimed_amount_minor": int(ar_row["debit_minor"]),
                "currency": base_currency,
                "evidence_ids": [invoice["doc_id"], ar_row["row_id"]],
                "explanation": "The invoice date precedes the reporting period but its receivable was posted within the period.",
            })

    for payment in pays:
        if payment.get("payment_type") != "refund" and not payment.get("invoice_refs"):
            findings.append({
                "type": "unallocated_or_partial_payment",
                "affected_ids": [payment["payment_id"]],
                "claimed_amount_minor": int(payment["amount_minor"]),
                "currency": payment["currency"],
                "evidence_ids": [payment["doc_id"]],
                "explanation": "Payment has no invoice references.",
            })

    for invoice in invs:
        linked_payments = [
            payment
            for payment in pays
            if invoice["invoice_id"] in (payment.get("invoice_refs") or [])
            and payment.get("payment_type") != "refund"
            and payment["currency"] == invoice["currency"]
        ]
        if not linked_payments:
            continue
        linked_payments.sort(key=lambda payment: payment["timestamp"])
        paid_minor = sum(int(payment["amount_minor"]) for payment in linked_payments)
        total_minor = int(invoice["total_minor"])
        if paid_minor == total_minor:
            continue

        if paid_minor > total_minor:
            running_total = 0
            for payment in linked_payments:
                previous_total = running_total
                running_total += int(payment["amount_minor"])
                if running_total > total_minor:
                    findings.append({
                        "type": "unallocated_or_partial_payment",
                        "affected_ids": [payment["payment_id"]],
                        "claimed_amount_minor": running_total - max(previous_total, total_minor),
                        "currency": invoice["currency"],
                        "evidence_ids": [invoice["doc_id"], payment["doc_id"]],
                        "explanation": "Payments linked to the invoice exceed its total.",
                    })
        else:
            findings.append({
                "type": "unallocated_or_partial_payment",
                "affected_ids": [payment["payment_id"] for payment in linked_payments],
                "claimed_amount_minor": total_minor - paid_minor,
                "currency": invoice["currency"],
                "evidence_ids": [invoice["doc_id"], *[payment["doc_id"] for payment in linked_payments]],
                "explanation": "Payments linked to the invoice do not cover its total.",
            })
    payload = {"episode_id": eid, "findings": findings[:30]}
    env.step({"name": "submit", "args": {"findings": payload}})
    return payload


def run_flag_all_policy(env: Any, init_obs: dict[str, Any]) -> dict[str, Any]:
    eid = init_obs.get("manifest", {}).get("episode_id", "")
    invs = _paginate(env, "invoices")
    types = init_obs.get("policy", {}).get("supported_error_types") or ["duplicate_invoice"]
    findings: list[dict[str, Any]] = []
    for inv in invs[:6]:
        for t in types:
            if len(findings) >= 30:
                break
            findings.append({
                "type": t,
                "affected_ids": [inv["invoice_id"]],
                "claimed_amount_minor": int(inv["total_minor"]),
                "currency": inv["currency"],
                "evidence_ids": [inv["doc_id"]],
                "explanation": "Flag-all baseline.",
            })
        if len(findings) >= 30:
            break
    payload = {"episode_id": eid, "findings": findings}
    env.step({"name": "submit", "args": {"findings": payload}})
    return payload


def run_oracle_policy(env: Any, init_obs: dict[str, Any], truth_issues: list[dict[str, Any]]) -> dict[str, Any]:
    eid = init_obs.get("manifest", {}).get("episode_id", "")
    invs = _paginate(env, "invoices")
    if invs:
        env.step({"name": "get_document", "args": {"doc_id": invs[0]["doc_id"]}})
    findings = [
        {
            "type": it["type"],
            "affected_ids": list(it["affected_ids"]),
            "claimed_amount_minor": int(it["expected_amount_minor"]),
            "currency": it["currency"],
            "evidence_ids": list(it.get("evidence_ids") or [])[:10],
            "explanation": f"Verified {it['issue_id']}",
        }
        for it in truth_issues[:30]
    ]
    payload = {"episode_id": eid, "findings": findings}
    env.step({"name": "submit", "args": {"findings": payload}})
    return payload