from __future__ import annotations

import csv
import copy
import json
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.3.0"
ERROR_TYPES = (
    "duplicate_invoice",
    "unallocated_or_partial_payment",
    "bad_fx_date_or_rate",
    "period_misclassification",
)
INTERACTION_SCENARIOS = (
    "partial_fx_payment_near_period_end",
    "duplicate_invoice_with_partial_payment",
)
CURRENCY_EXPONENTS = {"KZT": 2, "USD": 2, "EUR": 2, "GBP": 2}


def _fmt_dt(dt: datetime) -> str:
    return dt.isoformat()


def _round_minor(amount_minor: int, rate: Decimal) -> int:
    return int((Decimal(amount_minor) * rate).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _relabel_ids(
    ledger: list[dict[str, Any]],
    invoices: list[dict[str, Any]],
    payments: list[dict[str, Any]],
    truth_issues: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, dict[str, str]]:
    row_ids = [str(row["row_id"]) for row in ledger]
    entity_ids = sorted({
        str(row["entity_id"]) for row in ledger
    } | {
        str(invoice["entity_id"]) for invoice in invoices
    })
    invoice_ids = [str(invoice["invoice_id"]) for invoice in invoices]
    payment_ids = [str(payment["payment_id"]) for payment in payments]
    payment_ids.extend(sorted({
        str(row["payment_id"])
        for row in ledger
        if row["payment_id"] and str(row["payment_id"]) not in payment_ids
    }))
    invoice_map = {
        old: f"INV-{new:04d}"
        for old, new in zip(invoice_ids, rng.sample(range(1, len(invoice_ids) + 1), len(invoice_ids)))
    }
    payment_map = {
        old: f"PAY-{new:04d}"
        for old, new in zip(payment_ids, rng.sample(range(1, len(payment_ids) + 1), len(payment_ids)))
    }
    row_map = {
        old: f"LED-{new:05d}"
        for old, new in zip(row_ids, rng.sample(range(1, len(row_ids) + 1), len(row_ids)))
    }
    entity_map = {
        old: f"ENT-{new:03d}"
        for old, new in zip(entity_ids, rng.sample(range(1, len(entity_ids) + 1), len(entity_ids)))
    }
    document_map = {
        str(invoice["doc_id"]): f"DOC-{invoice_map[str(invoice['invoice_id'])]}"
        for invoice in invoices
    }
    document_map.update({
        str(payment["doc_id"]): f"DOC-{payment_map[str(payment['payment_id'])]}"
        for payment in payments
    })
    all_map = {**invoice_map, **payment_map, **document_map, **row_map, **entity_map}

    for invoice in invoices:
        old_id = str(invoice["invoice_id"])
        invoice["invoice_id"] = invoice_map[old_id]
        invoice["doc_id"] = document_map[str(invoice["doc_id"])]
        invoice["entity_id"] = entity_map[str(invoice["entity_id"])]
    for payment in payments:
        old_id = str(payment["payment_id"])
        payment["payment_id"] = payment_map[old_id]
        payment["doc_id"] = document_map[str(payment["doc_id"])]
        if payment.get("invoice_refs"):
            payment["invoice_refs"] = [invoice_map[ref] for ref in payment["invoice_refs"]]
    for row in ledger:
        row["row_id"] = row_map[str(row["row_id"])]
        row["entity_id"] = entity_map[str(row["entity_id"])]
        if row["invoice_id"]:
            row["invoice_id"] = invoice_map[row["invoice_id"]]
        if row["payment_id"]:
            row["payment_id"] = payment_map[row["payment_id"]]
    for issue in truth_issues:
        issue["affected_ids"] = sorted(all_map[item] for item in issue["affected_ids"])
        issue["evidence_ids"] = [all_map[item] for item in issue["evidence_ids"]]
        if issue.get("valid_evidence_sets"):
            issue["valid_evidence_sets"] = [
                [all_map[item] for item in evidence_set]
                for evidence_set in issue["valid_evidence_sets"]
            ]

    invoices.sort(key=lambda invoice: invoice["invoice_id"])
    payments.sort(key=lambda payment: payment["payment_id"])
    return {
        "invoices": invoice_map,
        "payments": payment_map,
        "documents": document_map,
        "ledger_rows": row_map,
        "entities": entity_map,
    }


def generate_episode(seed: int, config: dict[str, Any] | None, output_dir: str | Path) -> None:
    cfg = dict(config or {})
    unknown_errors = set(cfg.get("errors", [])) - set(ERROR_TYPES)
    unknown_interactions = set(cfg.get("interactions", [])) - set(INTERACTION_SCENARIOS)
    if unknown_errors:
        raise ValueError(f"Unsupported error types: {sorted(unknown_errors)}")
    if unknown_interactions:
        raise ValueError(f"Unsupported interaction scenarios: {sorted(unknown_interactions)}")
    rng = random.Random(seed)
    placement_rng = random.Random(f"placement:{seed}")
    relabel_rng = random.Random(int(cfg.get("relabel_seed", seed)))
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    episode_id = str(cfg.get("episode_id") or f"ep_{seed:06d}")
    base_currency = str(cfg.get("base_currency") or "KZT")
    tz_offset_hours = int(cfg.get("tz_offset_hours", 5))
    tz = timezone(timedelta(hours=tz_offset_hours))
    timezone_str = f"UTC{tz_offset_hours:+03d}:00"
    period_start = datetime(2026, 10, 1, 0, 0, 0, tzinfo=tz)
    period_end = datetime(2026, 10, 31, 23, 59, 59, tzinfo=tz)
    num_invoices = int(cfg.get("num_invoices", rng.randint(12, 20)))
    if not 6 <= num_invoices <= 500:
        raise ValueError("num_invoices must be in range 6..500")
    foreign_currency = "USD" if base_currency != "USD" else "EUR"
    scenario_params = dict(cfg.get("scenario_params") or {})
    unknown_scenario_params = set(scenario_params) - set(INTERACTION_SCENARIOS)
    if unknown_scenario_params:
        raise ValueError(f"Unknown scenario parameter groups: {sorted(unknown_scenario_params)}")

    amount_tolerance_minor = int(cfg.get("amount_tolerance_minor", 0))
    type_tolerances_minor = {
        error_type: amount_tolerance_minor
        for error_type in ERROR_TYPES
    }
    configured_tolerances = cfg.get("type_tolerances_minor", {})
    if not isinstance(configured_tolerances, dict):
        raise ValueError("type_tolerances_minor must be an object")
    for error_type, tolerance in configured_tolerances.items():
        if error_type not in ERROR_TYPES or type(tolerance) is not int or tolerance < 0:
            raise ValueError(f"Invalid amount tolerance for {error_type!r}")
        type_tolerances_minor[error_type] = tolerance

    policy = {
        "fx_date_selection": "invoice_or_payment_date_nearest_prior",
        "fx_rounding": "half_up",
        "period_inclusive": True,
        "amount_tolerance_minor": amount_tolerance_minor,
        "type_tolerances_minor": type_tolerances_minor,
        "supported_error_types": list(ERROR_TYPES),
        "interaction_scenarios": list(INTERACTION_SCENARIOS),
        "payment_type_semantics": {
            "payment": "incoming customer funds",
            "refund": "outgoing funds; excluded from invoice settlement totals",
        },
    }

    fx_rows: list[dict[str, str]] = []
    fx_lookup: dict[tuple[str, str, str], Decimal] = {}
    cur_d = (period_start - timedelta(days=3)).date()
    end_d = (period_end + timedelta(days=3)).date()
    base_rate = Decimal("480.00") if base_currency == "KZT" else Decimal("1.10")
    while cur_d <= end_d:
        d_str = cur_d.isoformat()
        delta_divisor = Decimal("100") if base_currency == "KZT" else Decimal("10000")
        delta = Decimal(rng.randint(-300, 300)) / delta_divisor
        r_fb = (base_rate + delta).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        r_bf = (Decimal("1") / r_fb).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        for src, dst, rt in [
            (foreign_currency, base_currency, r_fb),
            (base_currency, foreign_currency, r_bf),
            (base_currency, base_currency, Decimal("1")),
        ]:
            fx_rows.append({"date": d_str, "source_currency": src, "target_currency": dst, "rate": format(rt, "f")})
            fx_lookup[(d_str, src, dst)] = rt
        cur_d += timedelta(days=1)

    entities = [f"ENT-{i:03d}" for i in range(1, 6)]
    invoices, payments, ledger_rows, provenance_events = [], [], [], []
    row_counter = 1
    split_count = max(0, min(num_invoices, int(cfg.get("split_payment_count", num_invoices // 7))))
    split_invoice_indices = set(placement_rng.sample(range(1, num_invoices + 1), split_count))
    default_refund_count = num_invoices // 9
    refund_count = max(0, min(num_invoices, int(cfg.get("refund_count", default_refund_count))))
    refund_invoice_indices = set(placement_rng.sample(range(1, num_invoices + 1), refund_count))

    def next_row() -> str:
        nonlocal row_counter
        rid = f"LED-{row_counter:05d}"
        row_counter += 1
        return rid

    def add_payment_ledger(
        payment_id: str,
        payment_dt: datetime,
        entity_id: str,
        invoice_id: str,
        currency: str,
        amount_minor: int,
        booked_minor: int,
    ) -> None:
        payment_rate = fx_lookup.get(
            (payment_dt.date().isoformat(), currency, base_currency),
            Decimal("1"),
        )
        cash_minor = _round_minor(amount_minor, payment_rate)
        timestamp = _fmt_dt(payment_dt)
        ledger_rows.append({
            "row_id": next_row(), "timestamp": timestamp, "account": "CASH_BANK",
            "debit_minor": cash_minor, "credit_minor": 0, "currency": base_currency,
            "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id,
        })
        ledger_rows.append({
            "row_id": next_row(), "timestamp": timestamp, "account": "ACCOUNTS_RECEIVABLE",
            "debit_minor": 0, "credit_minor": booked_minor, "currency": base_currency,
            "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id,
        })
        difference = cash_minor - booked_minor
        if difference > 0:
            ledger_rows.append({
                "row_id": next_row(), "timestamp": timestamp, "account": "FX_GAIN",
                "debit_minor": 0, "credit_minor": difference, "currency": base_currency,
                "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id,
            })
        elif difference < 0:
            ledger_rows.append({
                "row_id": next_row(), "timestamp": timestamp, "account": "FX_LOSS",
                "debit_minor": -difference, "credit_minor": 0, "currency": base_currency,
                "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id,
            })

    for idx in range(1, num_invoices + 1):
        inv_id = f"INV-{idx:04d}"
        doc_inv_id = f"DOC-INV-{idx:04d}"
        pay_id = f"PAY-{idx:04d}"
        doc_pay_id = f"DOC-PAY-{idx:04d}"
        ent_id = rng.choice(entities)
        inv_dt = period_start + timedelta(days=rng.randint(1, 25), hours=rng.randint(8, 17))
        due_dt = inv_dt + timedelta(days=14)
        pay_dt = inv_dt + timedelta(days=rng.randint(1, 3))
        inv_curr = foreign_currency if (idx % 4 == 0) else base_currency
        unit_minor = rng.randint(50, 300) * 100
        qty = rng.randint(1, 3)
        total_minor = unit_minor * qty
        if idx == 1 and cfg.get("period_boundary_invoice"):
            inv_dt = period_start
            due_dt = period_end
            pay_dt = period_end
        if idx == 2 and cfg.get("legal_repeat_amounts"):
            original = invoices[0]
            ent_id = original["entity_id"]
            inv_curr = original["currency"]
            total_minor = int(original["total_minor"])
            qty = int(original["line_items"][0]["quantity"])
            unit_minor = int(original["line_items"][0]["unit_amount_minor"])
        invoices.append({
            "doc_id": doc_inv_id,
            "invoice_id": inv_id,
            "entity_id": ent_id,
            "issued_at": _fmt_dt(inv_dt),
            "due_at": _fmt_dt(due_dt),
            "currency": inv_curr,
            "total_minor": total_minor,
            "line_items": [{
                "line_no": 1,
                "description": f"Line {idx}",
                "quantity": qty,
                "unit_amount_minor": unit_minor,
                "amount_minor": total_minor,
            }],
        })
        rate = fx_lookup.get((inv_dt.date().isoformat(), inv_curr, base_currency), Decimal("1"))
        base_amt = _round_minor(total_minor, rate)
        ledger_rows.append({"row_id": next_row(), "timestamp": _fmt_dt(inv_dt), "account": "ACCOUNTS_RECEIVABLE", "debit_minor": base_amt, "credit_minor": 0, "currency": base_currency, "entity_id": ent_id, "invoice_id": inv_id, "payment_id": None})
        ledger_rows.append({"row_id": next_row(), "timestamp": _fmt_dt(inv_dt), "account": "REVENUE", "debit_minor": 0, "credit_minor": base_amt, "currency": base_currency, "entity_id": ent_id, "invoice_id": inv_id, "payment_id": None})

        # Legitimate distractor: split payments & balanced refunds
        if idx in split_invoice_indices and total_minor >= 200:
            h1 = total_minor // 2
            h2 = total_minor - h1
            booked_remaining = base_amt
            for suffix, part_amt, dt_p in [("A", h1, pay_dt), ("B", h2, pay_dt + timedelta(hours=3))]:
                pid = f"{pay_id}-{suffix}"
                payments.append({
                    "doc_id": f"{doc_pay_id}-{suffix}",
                    "payment_id": pid,
                    "timestamp": _fmt_dt(dt_p),
                    "currency": inv_curr,
                    "amount_minor": part_amt,
                    "invoice_refs": [inv_id],
                    "payment_type": "payment",
                })
                if suffix == "A":
                    booked_part = int(
                        (Decimal(base_amt) * Decimal(part_amt) / Decimal(total_minor)).quantize(
                            Decimal("1"),
                            rounding=ROUND_HALF_UP,
                        )
                    )
                    booked_remaining -= booked_part
                else:
                    booked_part = booked_remaining
                add_payment_ledger(pid, dt_p, ent_id, inv_id, inv_curr, part_amt, booked_part)
        else:
            payments.append({"doc_id": doc_pay_id, "payment_id": pay_id, "timestamp": _fmt_dt(pay_dt), "currency": inv_curr, "amount_minor": total_minor, "invoice_refs": [inv_id], "payment_type": "payment"})
            add_payment_ledger(pay_id, pay_dt, ent_id, inv_id, inv_curr, total_minor, base_amt)

        if idx in refund_invoice_indices:
            refund_id = f"PAY-REFUND-{idx:04d}"
            refund_doc = f"DOC-PAY-REFUND-{idx:04d}"
            refund_minor = min(
                total_minor,
                max(1, int(cfg.get("refund_amount_minor", total_minor // 10))),
            )
            refund_dt = pay_dt + timedelta(days=1)
            payments.append({
                "doc_id": refund_doc,
                "payment_id": refund_id,
                "timestamp": _fmt_dt(refund_dt),
                "currency": inv_curr,
                "amount_minor": refund_minor,
                "invoice_refs": [inv_id],
                "payment_type": "refund",
            })
            refund_rate = fx_lookup.get((refund_dt.date().isoformat(), inv_curr, base_currency), Decimal("1"))
            base_refund_minor = _round_minor(refund_minor, refund_rate)
            ledger_rows.append({"row_id": next_row(), "timestamp": _fmt_dt(refund_dt), "account": "SALES_RETURNS", "debit_minor": base_refund_minor, "credit_minor": 0, "currency": base_currency, "entity_id": ent_id, "invoice_id": inv_id, "payment_id": refund_id})
            ledger_rows.append({"row_id": next_row(), "timestamp": _fmt_dt(refund_dt), "account": "CASH_BANK", "debit_minor": 0, "credit_minor": base_refund_minor, "currency": base_currency, "entity_id": ent_id, "invoice_id": inv_id, "payment_id": refund_id})
            provenance_events.append({
                "event": "balanced_refund",
                "invoice_id": inv_id,
                "payment_id": refund_id,
                "doc_id": refund_doc,
                "amount_minor": refund_minor,
                "balanced": True,
            })

        provenance_events.append({
            "event": "base_invoice_cycle",
            "invoice_id": inv_id,
            "doc_id": doc_inv_id,
            "payment_ids": [
                payment["payment_id"]
                for payment in payments
                if inv_id in payment.get("invoice_refs", [])
                and payment.get("payment_type") != "refund"
            ],
            "balanced": True,
        })

    min_rows = int(cfg.get("min_ledger_rows", len(ledger_rows)))
    pad_i = 1
    while len(ledger_rows) < min_rows:
        ent_id = entities[pad_i % len(entities)]
        amt = 1000 * (1 + (pad_i % 5))
        dt_p = _fmt_dt(period_start + timedelta(days=(pad_i % 25) + 1))
        ledger_rows.append({"row_id": next_row(), "timestamp": dt_p, "account": "SUSPENSE_CLEARING", "debit_minor": amt, "credit_minor": 0, "currency": base_currency, "entity_id": ent_id, "invoice_id": None, "payment_id": None})
        ledger_rows.append({"row_id": next_row(), "timestamp": dt_p, "account": "SUSPENSE_CLEARING", "debit_minor": 0, "credit_minor": amt, "currency": base_currency, "entity_id": ent_id, "invoice_id": None, "payment_id": None})
        pad_i += 1

    baseline_snapshot = copy.deepcopy({
        "ledger": ledger_rows,
        "invoices": invoices,
        "payments": payments,
        "fx": fx_rows,
    })
    truth_issues: list[dict[str, Any]] = []
    issue_idx = 1

    def next_issue() -> str:
        nonlocal issue_idx
        iid = f"ISS-{issue_idx:03d}"
        issue_idx += 1
        return iid

    pool = list(invoices)

    def pick_inv() -> dict[str, Any]:
        if not pool:
            raise ValueError("Not enough unused invoices to place all configured scenarios")
        invoice = placement_rng.choice(pool)
        pool.remove(invoice)
        return invoice

    def pick_invoice(currency: str | None = None) -> dict[str, Any]:
        candidates = [invoice for invoice in pool if currency is None or invoice["currency"] == currency]
        if not candidates:
            raise ValueError(f"No generated invoice uses currency {currency}")
        invoice = placement_rng.choice(candidates)
        pool.remove(invoice)
        return invoice

    def unique_id(candidate: str, used_ids: set[str]) -> str:
        unique = candidate
        suffix = 2
        while unique in used_ids:
            unique = f"{candidate}-{suffix}"
            suffix += 1
        used_ids.add(unique)
        return unique

    used_invoice_ids = {str(invoice["invoice_id"]) for invoice in invoices}
    used_document_ids = {str(invoice["doc_id"]) for invoice in invoices}

    def params_for(scenario: str) -> dict[str, Any]:
        params = scenario_params.get(scenario, {})
        if not isinstance(params, dict):
            raise ValueError(f"scenario_params[{scenario!r}] must be an object")
        return params

    def linked_payments(invoice: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            payment
            for payment in payments
            if invoice["invoice_id"] in (payment.get("invoice_refs") or [])
            and payment.get("payment_type") != "refund"
            and payment["currency"] == invoice["currency"]
        ]

    def update_partial_payment(
        invoice: dict[str, Any],
        shortfall_minor: int,
        payment_timestamp: datetime | None = None,
    ) -> list[dict[str, Any]]:
        invoice_payments = linked_payments(invoice)
        if not invoice_payments:
            raise ValueError(f"No posted payment available for invoice {invoice['invoice_id']}")
        if shortfall_minor <= 0 or shortfall_minor >= sum(int(p["amount_minor"]) for p in invoice_payments):
            raise ValueError("partial payment shortfall must be positive and less than the payment total")

        target = invoice_payments[-1]
        old_amount = int(target["amount_minor"])
        target["amount_minor"] = old_amount - shortfall_minor
        if payment_timestamp is not None:
            target["timestamp"] = _fmt_dt(payment_timestamp)
        target_date = str(target["timestamp"])[:10]
        payment_rate = fx_lookup.get((target_date, target["currency"], base_currency), Decimal("1"))
        cash_minor = _round_minor(int(target["amount_minor"]), payment_rate)
        invoice_date = str(invoice["issued_at"])[:10]
        invoice_rate = fx_lookup.get((invoice_date, invoice["currency"], base_currency), Decimal("1"))
        booked_minor = _round_minor(int(target["amount_minor"]), invoice_rate)
        target_id = target["payment_id"]
        ledger_rows[:] = [
            row
            for row in ledger_rows
            if not (
                row["payment_id"] == target_id
                and row["account"] in ("FX_GAIN", "FX_LOSS")
            )
        ]
        for row in ledger_rows:
            if row["payment_id"] != target_id:
                continue
            row["timestamp"] = target["timestamp"]
            if row["account"] == "CASH_BANK":
                row["debit_minor"] = cash_minor
                row["credit_minor"] = 0
            elif row["account"] == "ACCOUNTS_RECEIVABLE":
                row["debit_minor"] = 0
                row["credit_minor"] = booked_minor
        difference = cash_minor - booked_minor
        if difference > 0:
            ledger_rows.append({
                "row_id": next_row(), "timestamp": target["timestamp"], "account": "FX_GAIN",
                "debit_minor": 0, "credit_minor": difference, "currency": base_currency,
                "entity_id": invoice["entity_id"], "invoice_id": invoice["invoice_id"], "payment_id": target_id,
            })
        elif difference < 0:
            ledger_rows.append({
                "row_id": next_row(), "timestamp": target["timestamp"], "account": "FX_LOSS",
                "debit_minor": -difference, "credit_minor": 0, "currency": base_currency,
                "entity_id": invoice["entity_id"], "invoice_id": invoice["invoice_id"], "payment_id": target_id,
            })
        return invoice_payments

    for err_type in list(cfg.get("errors", [])):
        inv = pick_inv()
        if err_type == "duplicate_invoice":
            dup_id = unique_id(f"{inv['invoice_id']}-DUP", used_invoice_ids)
            dup_doc = unique_id(f"{inv['doc_id']}-DUP", used_document_ids)
            invoices.append({**inv, "invoice_id": dup_id, "doc_id": dup_doc, "line_items": copy.deepcopy(inv["line_items"])})
            rate = fx_lookup.get((inv["issued_at"][:10], inv["currency"], base_currency), Decimal("1"))
            b_amt = _round_minor(int(inv["total_minor"]), rate)
            ledger_rows.append({"row_id": next_row(), "timestamp": inv["issued_at"], "account": "ACCOUNTS_RECEIVABLE", "debit_minor": b_amt, "credit_minor": 0, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": dup_id, "payment_id": None})
            ledger_rows.append({"row_id": next_row(), "timestamp": inv["issued_at"], "account": "REVENUE", "debit_minor": 0, "credit_minor": b_amt, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": dup_id, "payment_id": None})
            iid = next_issue()
            truth_issues.append({"issue_id": iid, "type": "duplicate_invoice", "affected_ids": [inv["invoice_id"], dup_id], "expected_amount_minor": int(inv["total_minor"]), "currency": inv["currency"], "evidence_ids": [inv["doc_id"], dup_doc]})
            provenance_events.append({"event": "inject_duplicate_invoice", "issue_id": iid, "evidence_ids": [inv["doc_id"], dup_doc]})
        elif err_type == "unallocated_or_partial_payment":
            pid = f"PAY-UNALLOC-{issue_idx:03d}"
            pdoc = f"DOC-PAY-UNALLOC-{issue_idx:03d}"
            tot = int(inv["total_minor"])
            mode = str(cfg.get("payment_issue_mode", "unallocated"))
            if mode not in ("unallocated", "partial"):
                raise ValueError("payment_issue_mode must be 'unallocated' or 'partial'")
            discrepancy = int(cfg.get("payment_discrepancy_minor", max(100, tot // 3)))
            if discrepancy <= 0:
                raise ValueError("payment_discrepancy_minor must be positive")
            p_dt = _fmt_dt(period_start + timedelta(days=10))
            iid = next_issue()
            if mode == "unallocated":
                payments.append({
                    "doc_id": pdoc,
                    "payment_id": pid,
                    "timestamp": p_dt,
                    "currency": inv["currency"],
                    "amount_minor": discrepancy,
                    "invoice_refs": [],
                    "payment_type": "payment",
                })
                rate = fx_lookup.get((p_dt[:10], inv["currency"], base_currency), Decimal("1"))
                base_discrepancy = _round_minor(discrepancy, rate)
                ledger_rows.append({"row_id": next_row(), "timestamp": p_dt, "account": "CASH_BANK", "debit_minor": base_discrepancy, "credit_minor": 0, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": None, "payment_id": pid})
                ledger_rows.append({"row_id": next_row(), "timestamp": p_dt, "account": "UNAPPLIED_CASH", "debit_minor": 0, "credit_minor": base_discrepancy, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": None, "payment_id": pid})
                affected_ids = [pid]
                evidence_ids = [pdoc]
                provenance_events.append({
                    "event": "inject_unallocated_payment",
                    "issue_id": iid,
                    "payment_id": pid,
                    "amount_minor": discrepancy,
                    "evidence_ids": evidence_ids,
                })
            else:
                invoice_payments = linked_payments(inv)
                if discrepancy >= sum(int(payment["amount_minor"]) for payment in invoice_payments):
                    raise ValueError("payment_discrepancy_minor must be less than the invoice's posted payments")
                invoice_payments = update_partial_payment(inv, discrepancy)
                affected_ids = [payment["payment_id"] for payment in invoice_payments]
                evidence_ids = [inv["doc_id"], *[payment["doc_id"] for payment in invoice_payments]]
                provenance_events.append({
                    "event": "inject_partial_payment",
                    "issue_id": iid,
                    "invoice_id": inv["invoice_id"],
                    "payment_ids": affected_ids,
                    "shortfall_minor": discrepancy,
                    "evidence_ids": evidence_ids,
                })
            truth_issues.append({
                "issue_id": iid,
                "type": "unallocated_or_partial_payment",
                "affected_ids": affected_ids,
                "expected_amount_minor": discrepancy,
                "currency": inv["currency"],
                "evidence_ids": evidence_ids,
            })
        elif err_type == "bad_fx_date_or_rate":
            fx_id = f"INV-FXERR-{issue_idx:03d}"
            fx_doc = f"DOC-INV-FXERR-{issue_idx:03d}"
            amt_f = int(cfg.get("fx_invoice_amount_minor", 10000))
            d_str = (period_start + timedelta(days=12)).date().isoformat()
            ts_str = _fmt_dt(period_start + timedelta(days=12, hours=10))
            invoices.append({"doc_id": fx_doc, "invoice_id": fx_id, "entity_id": inv["entity_id"], "issued_at": ts_str, "due_at": _fmt_dt(period_start + timedelta(days=26)), "currency": foreign_currency, "total_minor": amt_f, "line_items": [{"line_no": 1, "description": f"Line {issue_idx}", "quantity": 1, "unit_amount_minor": amt_f, "amount_minor": amt_f}]})
            used_invoice_ids.add(fx_id)
            used_document_ids.add(fx_doc)
            correct_rate = fx_lookup[(d_str, foreign_currency, base_currency)]
            alternative_rates = [
                (date_str, fx_lookup[(date_str, foreign_currency, base_currency)])
                for date_str, src, dst in fx_lookup
                if src == foreign_currency and dst == base_currency and date_str != d_str
            ]
            wrong_date, wrong_rate = max(
                alternative_rates,
                key=lambda item: abs(
                    _round_minor(amt_f, item[1]) - _round_minor(amt_f, correct_rate)
                ),
            )
            exp_b = _round_minor(amt_f, correct_rate)
            wrong_b = _round_minor(amt_f, wrong_rate)
            ar_row_id = next_row()
            revenue_row_id = next_row()
            ledger_rows.append({"row_id": ar_row_id, "timestamp": ts_str, "account": "ACCOUNTS_RECEIVABLE", "debit_minor": wrong_b, "credit_minor": 0, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": fx_id, "payment_id": None})
            ledger_rows.append({"row_id": revenue_row_id, "timestamp": ts_str, "account": "REVENUE", "debit_minor": 0, "credit_minor": wrong_b, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": fx_id, "payment_id": None})
            fx_payment_id = f"PAY-FXERR-{issue_idx:03d}"
            fx_payment_doc = f"DOC-PAY-FXERR-{issue_idx:03d}"
            fx_payment_dt = period_start + timedelta(days=18, hours=12)
            payments.append({
                "doc_id": fx_payment_doc,
                "payment_id": fx_payment_id,
                "timestamp": _fmt_dt(fx_payment_dt),
                "currency": foreign_currency,
                "amount_minor": amt_f,
                "invoice_refs": [fx_id],
                "payment_type": "payment",
            })
            add_payment_ledger(
                fx_payment_id,
                fx_payment_dt,
                inv["entity_id"],
                fx_id,
                foreign_currency,
                amt_f,
                wrong_b,
            )
            iid = next_issue()
            fx_evidence = [fx_doc, ar_row_id]
            truth_issues.append({
                "issue_id": iid,
                "type": "bad_fx_date_or_rate",
                "affected_ids": [fx_id],
                "expected_amount_minor": abs(wrong_b - exp_b),
                "currency": base_currency,
                "evidence_ids": fx_evidence,
                "valid_evidence_sets": [fx_evidence],
            })
            provenance_events.append({
                "event": "inject_bad_fx",
                "issue_id": iid,
                "invoice_id": fx_id,
                "date": d_str,
                "currency": foreign_currency,
                "recorded_rate": format(wrong_rate, "f"),
                "recorded_rate_date": wrong_date,
                "expected_rate": format(correct_rate, "f"),
                "expected_base_minor": exp_b,
                "recorded_base_minor": wrong_b,
                "evidence_ids": fx_evidence,
            })
            provenance_events.append({
                "event": "settle_injected_fx_invoice",
                "invoice_id": fx_id,
                "payment_id": fx_payment_id,
                "balanced": True,
            })
        elif err_type == "period_misclassification":
            op_id = f"INV-OUTPER-{issue_idx:03d}"
            op_doc = f"DOC-INV-OUTPER-{issue_idx:03d}"
            out_dt = _fmt_dt(period_start - timedelta(days=int(cfg.get("period_misclassification_days", 5))))
            in_dt = _fmt_dt(period_start + timedelta(days=int(cfg.get("ledger_day_after_period_start", 2))))
            amt_m = int(cfg.get("period_misclassification_minor", 15000))
            invoices.append({"doc_id": op_doc, "invoice_id": op_id, "entity_id": inv["entity_id"], "issued_at": out_dt, "due_at": _fmt_dt(period_start + timedelta(days=9)), "currency": base_currency, "total_minor": amt_m, "line_items": [{"line_no": 1, "description": f"Line {issue_idx}", "quantity": 1, "unit_amount_minor": amt_m, "amount_minor": amt_m}]})
            used_invoice_ids.add(op_id)
            used_document_ids.add(op_doc)
            ar_row_id = next_row()
            revenue_row_id = next_row()
            ledger_rows.append({"row_id": ar_row_id, "timestamp": in_dt, "account": "ACCOUNTS_RECEIVABLE", "debit_minor": amt_m, "credit_minor": 0, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": op_id, "payment_id": None})
            ledger_rows.append({"row_id": revenue_row_id, "timestamp": in_dt, "account": "REVENUE", "debit_minor": 0, "credit_minor": amt_m, "currency": base_currency, "entity_id": inv["entity_id"], "invoice_id": op_id, "payment_id": None})
            period_payment_id = f"PAY-OUTPER-{issue_idx:03d}"
            period_payment_doc = f"DOC-PAY-OUTPER-{issue_idx:03d}"
            period_payment_dt = period_start + timedelta(days=10, hours=12)
            payments.append({
                "doc_id": period_payment_doc,
                "payment_id": period_payment_id,
                "timestamp": _fmt_dt(period_payment_dt),
                "currency": base_currency,
                "amount_minor": amt_m,
                "invoice_refs": [op_id],
                "payment_type": "payment",
            })
            add_payment_ledger(
                period_payment_id,
                period_payment_dt,
                inv["entity_id"],
                op_id,
                base_currency,
                amt_m,
                amt_m,
            )
            iid = next_issue()
            period_evidence = [op_doc, ar_row_id]
            truth_issues.append({
                "issue_id": iid,
                "type": "period_misclassification",
                "affected_ids": [op_id],
                "expected_amount_minor": amt_m,
                "currency": base_currency,
                "evidence_ids": period_evidence,
                "valid_evidence_sets": [period_evidence],
            })
            provenance_events.append({
                "event": "inject_period_misclassification",
                "issue_id": iid,
                "invoice_id": op_id,
                "invoice_date": out_dt,
                "ledger_date": in_dt,
                "amount_minor": amt_m,
                "evidence_ids": period_evidence,
            })
            provenance_events.append({
                "event": "settle_injected_period_invoice",
                "invoice_id": op_id,
                "payment_id": period_payment_id,
                "balanced": True,
            })

    for scen in list(cfg.get("interactions", [])):
        params = params_for(scen)
        if scen == "partial_fx_payment_near_period_end":
            inv = pick_invoice(foreign_currency)
            days_before_period_end = int(params.get("days_before_period_end", 1))
            if days_before_period_end < 0 or days_before_period_end > 3:
                raise ValueError("days_before_period_end must be in range 0..3")
            payment_timestamp = period_end - timedelta(days=days_before_period_end)
            invoice_payments = linked_payments(inv)
            posted_total = sum(int(payment["amount_minor"]) for payment in invoice_payments)
            shortfall_minor = int(params.get("shortfall_minor", max(1, posted_total // 4)))
            invoice_payments = update_partial_payment(inv, shortfall_minor, payment_timestamp)
            iid = next_issue()
            affected_ids = [payment["payment_id"] for payment in invoice_payments]
            evidence_ids = [inv["doc_id"], *[payment["doc_id"] for payment in invoice_payments]]
            truth_issues.append({
                "issue_id": iid,
                "type": "unallocated_or_partial_payment",
                "affected_ids": affected_ids,
                "expected_amount_minor": shortfall_minor,
                "currency": inv["currency"],
                "evidence_ids": evidence_ids,
            })
            provenance_events.append({
                "event": "interaction_partial_fx_payment_near_period_end",
                "issue_id": iid,
                "invoice_id": inv["invoice_id"],
                "payment_ids": affected_ids,
                "shortfall_minor": shortfall_minor,
                "payment_timestamp": _fmt_dt(payment_timestamp),
                "evidence_ids": evidence_ids,
            })
        elif scen == "duplicate_invoice_with_partial_payment":
            inv = pick_inv()
            dup_id = unique_id(f"{inv['invoice_id']}-INTDUP", used_invoice_ids)
            dup_doc = unique_id(f"{inv['doc_id']}-INTDUP", used_document_ids)
            invoices.append({**inv, "invoice_id": dup_id, "doc_id": dup_doc, "line_items": copy.deepcopy(inv["line_items"])})
            rate_inv = fx_lookup.get((inv["issued_at"][:10], inv["currency"], base_currency), Decimal("1"))
            b_inv = _round_minor(int(inv["total_minor"]), rate_inv)
            ledger_rows.append({
                "row_id": next_row(), "timestamp": inv["issued_at"], "account": "ACCOUNTS_RECEIVABLE",
                "debit_minor": b_inv, "credit_minor": 0, "currency": base_currency,
                "entity_id": inv["entity_id"], "invoice_id": dup_id, "payment_id": None,
            })
            ledger_rows.append({
                "row_id": next_row(), "timestamp": inv["issued_at"], "account": "REVENUE",
                "debit_minor": 0, "credit_minor": b_inv, "currency": base_currency,
                "entity_id": inv["entity_id"], "invoice_id": dup_id, "payment_id": None,
            })
            iid1 = next_issue()
            truth_issues.append({
                "issue_id": iid1,
                "type": "duplicate_invoice",
                "affected_ids": [inv["invoice_id"], dup_id],
                "expected_amount_minor": int(inv["total_minor"]),
                "currency": inv["currency"],
                "evidence_ids": [inv["doc_id"], dup_doc],
            })
            invoice_payments = linked_payments(inv)
            posted_total = sum(int(payment["amount_minor"]) for payment in invoice_payments)
            shortfall_minor = int(params.get("shortfall_minor", max(1, posted_total // 4)))
            invoice_payments = update_partial_payment(inv, shortfall_minor)
            iid2 = next_issue()
            payment_ids = [payment["payment_id"] for payment in invoice_payments]
            evidence_ids = [inv["doc_id"], *[payment["doc_id"] for payment in invoice_payments]]
            truth_issues.append({
                "issue_id": iid2,
                "type": "unallocated_or_partial_payment",
                "affected_ids": payment_ids,
                "expected_amount_minor": shortfall_minor,
                "currency": inv["currency"],
                "evidence_ids": evidence_ids,
            })
            provenance_events.append({
                "event": "interaction_duplicate_invoice_with_partial_payment",
                "issue_ids": [iid1, iid2],
                "invoice_id": inv["invoice_id"],
                "duplicate_invoice_id": dup_id,
                "payment_ids": payment_ids,
                "shortfall_minor": shortfall_minor,
                "evidence_ids": [inv["doc_id"], dup_doc, *[payment["doc_id"] for payment in invoice_payments]],
            })
            
    id_mapping = _relabel_ids(ledger_rows, invoices, payments, truth_issues, relabel_rng)
    if cfg.get("shuffle_records"):
        rng.shuffle(ledger_rows)
        rng.shuffle(invoices)
        rng.shuffle(payments)
        rng.shuffle(fx_rows)
    else:
        ledger_rows.sort(key=lambda row: (row["timestamp"], row["row_id"]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "base_currency": base_currency,
        "period_start": _fmt_dt(period_start),
        "period_end": _fmt_dt(period_end),
        "timezone": timezone_str,
        "currency_exponents": CURRENCY_EXPONENTS,
        "counts": {
            "ledger_rows": len(ledger_rows),
            "invoices": len(invoices),
            "payments": len(payments),
            "fx_rows": len(fx_rows),
            "documents": len(invoices) + len(payments),
        },
        "rounding_rule": policy["fx_rounding"],
    }
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_path / "policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_path / "ledger.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency", "entity_id", "invoice_id", "payment_id"])
        w.writeheader()
        for r in ledger_rows:
            w.writerow({
                **r,
                "invoice_id": "" if r["invoice_id"] is None else r["invoice_id"],
                "payment_id": "" if r["payment_id"] is None else r["payment_id"],
            })

    with (out_path / "invoices.jsonl").open("w", encoding="utf-8") as f:
        for inv in invoices:
            f.write(json.dumps(inv, ensure_ascii=False) + "\n")

    with (out_path / "payments.jsonl").open("w", encoding="utf-8") as f:
        for pay in payments:
            f.write(json.dumps(pay, ensure_ascii=False) + "\n")

    with (out_path / "fx.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "source_currency", "target_currency", "rate"])
        w.writeheader()
        w.writerows(fx_rows)

    (out_path / "truth.json").write_text(
        json.dumps({"episode_id": episode_id, "issues": truth_issues}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_path / "provenance.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "seed": seed,
                "schema_version": SCHEMA_VERSION,
                "generator_version": SCHEMA_VERSION,
                "config": cfg,
                "baseline": baseline_snapshot,
                "events": provenance_events,
                "id_mapping": id_mapping,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )