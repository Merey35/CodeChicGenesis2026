from __future__ import annotations

import copy
import csv
import json
import re
import time
from pathlib import Path

from genesis_fintech import (
    generate_episode,
    judge_adapter,
    make_env,
    max_bipartite_matching,
    score_findings,
)
from genesis_fintech.env import TIME_LIMIT_SECONDS
from genesis_fintech.matcher import compute_reward
from genesis_fintech.baselines import run_simple_reconciliation_policy
from genesis_fintech.runner import (
    LoggedEnvWrapper,
    _execute_policy_isolated,
    build_default_episodes,
    run_benchmark,
)
import genesis_fintech.runner as runner_module


def test_contract_and_isolation(tmp_path: Path) -> None:
    ep_c = tmp_path / "ep_c"
    generate_episode(seed=10, config={"episode_id": "ep_c", "errors": [], "min_ledger_rows": 105}, output_dir=ep_c)
    ep_e = tmp_path / "ep_e"
    generate_episode(seed=20, config={"episode_id": "ep_e", "errors": ["duplicate_invoice"]}, output_dir=ep_e)

    env = make_env()
    obs, info = env.reset(seed=10, config={"mode": "closed", "episode_dir": str(ep_c)})
    assert isinstance(obs, dict) and isinstance(info, dict)
    for k in ("task", "manifest", "policy", "remaining_steps"):
        assert k in obs
    assert obs["remaining_steps"] == 100
    assert "truth" not in json.dumps(obs) and "provenance" not in json.dumps(obs)

    p1, rew1, term1, trunc1, _ = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}})
    assert rew1 == 0.0 and not term1 and not trunc1 and len(p1["result"]["records"]) <= 50

    p1_rep, _, _, _, _ = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}})
    assert p1_rep["result"]["records"] == p1["result"]["records"]

    calc_ok, _, _, _, _ = env.step({"name": "calculator", "args": {"expression": "0.1 + 0.2"}})
    assert calc_ok["error"] is None and calc_ok["result"] == "0.3"

    calc_bad, _, _, _, _ = env.step({"name": "calculator", "args": {"expression": "__import__('os').system('id')"}})
    assert calc_bad["result"] is None and calc_bad["error"] is not None

    rem0 = env.step({"name": "get_record", "args": {"table": "invoices", "id": "MISSING_ID"}})[0]["remaining_steps"]
    rem1 = env.step({"name": "unknown_action", "args": {}})[0]["remaining_steps"]
    assert rem1 == rem0 - 1

    env.reset(seed=20, config={"mode": "closed", "episode_dir": str(ep_e)})
    for _ in range(99):
        _, rew, term, trunc, _ = env.step({"name": "calculator", "args": {"expression": "1 + 1"}})
        assert rew == 0.0 and not term and not trunc
    last_obs, last_rew, last_term, last_trunc, _ = env.step({
        "name": "submit",
        "args": {"findings": {"episode_id": "ep_e", "findings": []}},
    })
    assert last_rew == 0.0 and last_term is True and last_trunc is False and last_obs["result"] == {"accepted": True}


def test_truncated_at_100_steps(tmp_path: Path) -> None:
    ep = tmp_path / "ep_trunc"
    generate_episode(seed=30, config={"episode_id": "ep_trunc", "errors": []}, output_dir=ep)
    env = make_env()
    env.reset(seed=30, config={"mode": "closed", "episode_dir": str(ep)})
    for _ in range(99):
        env.step({"name": "calculator", "args": {"expression": "2 + 2"}})
    obs100, _, term100, trunc100, _ = env.step({"name": "calculator", "args": {"expression": "2 + 2"}})
    assert trunc100 is True and term100 is False and obs100["remaining_steps"] == 0
    assert env.get_submitted_findings()["findings"] == []


def test_matcher_cases() -> None:
    issue_a = {
        "issue_id": "A",
        "type": "duplicate_invoice",
        "affected_ids": ["R1"],
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1", "D2"],
        "tolerance_minor": 0,
    }
    avail = {"R1", "R2", "D1", "D2", "D3"}
    good_f = {
        "type": "duplicate_invoice",
        "affected_ids": ["R1"],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1", "D2"],
        "explanation": "ok",
    }
    assert abs(score_findings({"episode_id": "e", "findings": [good_f]}, [issue_a], avail, 10)["return"] - 0.98) < 1e-9
    assert abs(score_findings({"episode_id": "e", "findings": []}, [issue_a], avail, 10)["return"] - (-1.0)) < 1e-9
    assert abs(score_findings({"episode_id": "e", "findings": []}, [], avail, 10)["return"] - 0.98) < 1e-9
    assert abs(score_findings({"episode_id": "e", "findings": [good_f]}, [], avail, 10)["return"] - (-1.0)) < 1e-9
    assert abs(score_findings({"episode_id": "e", "findings": [good_f, copy.deepcopy(good_f)]}, [issue_a], avail, 10)["return"] - (47.0 / 150.0)) < 1e-9

    r18 = judge_adapter.score_episode(".", {"episode_id": "e", "findings": [good_f] * 31}, 10)
    assert r18["return"] == -1 and r18["tp"] is None

    issues_ab = [
        {"issue_id": "A", "type": "duplicate_invoice", "affected_ids": ["R1"], "expected_amount_minor": 10000, "currency": "KZT", "evidence_ids": ["D1"]},
        {"issue_id": "B", "type": "duplicate_invoice", "affected_ids": ["R2"], "expected_amount_minor": 10000, "currency": "KZT", "evidence_ids": ["D1"]},
    ]
    tp_m, pairs_m = max_bipartite_matching(2, issues_ab, [(0, 0), (0, 1), (1, 0)])
    assert tp_m == 2 and pairs_m == [("A", 1), ("B", 0)]
    assert compute_reward(1, 0, 0, 0)["return"] == 1.0


def test_matcher_rubric_normalization_and_failures() -> None:
    issue = {
        "issue_id": "A",
        "type": "duplicate_invoice",
        "affected_ids": ["R1", "R2"],
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1", "D2"],
    }
    finding = {
        "type": "duplicate_invoice",
        "affected_ids": ["R1", "R2"],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1", "D2"],
        "explanation": "verified",
    }
    evidence = {"D1", "D2", "D3"}

    normalized = copy.deepcopy(finding)
    normalized["affected_ids"] = ["R2", "R1", "R1"]
    normalized["evidence_ids"] = ["D2", "D1", "D3"]
    assert score_findings({"findings": [normalized]}, [issue], evidence, 10)["tp"] == 1

    invalid_cases = []
    wrong_type = copy.deepcopy(finding)
    wrong_type["type"] = "period_misclassification"
    invalid_cases.append(wrong_type)
    wrong_ids = copy.deepcopy(finding)
    wrong_ids["affected_ids"] = ["R1", "R2", "R3"]
    invalid_cases.append(wrong_ids)
    wrong_amount = copy.deepcopy(finding)
    wrong_amount["claimed_amount_minor"] = 10001
    invalid_cases.append(wrong_amount)
    wrong_currency = copy.deepcopy(finding)
    wrong_currency["currency"] = "kzt"
    invalid_cases.append(wrong_currency)
    incomplete_evidence = copy.deepcopy(finding)
    incomplete_evidence["evidence_ids"] = ["D1"]
    invalid_cases.append(incomplete_evidence)
    unknown_evidence = copy.deepcopy(finding)
    unknown_evidence["evidence_ids"] = ["D1", "U"]
    invalid_cases.append(unknown_evidence)
    for candidate in invalid_cases:
        result = score_findings({"findings": [candidate]}, [issue], evidence, 10)
        assert result["tp"] == 0 and result["fp"] == 1 and result["fn"] == 1
        assert result["return"] == -1.0

    assert score_findings({"findings": []}, [issue], evidence, 10)["return"] == -1.0
    assert score_findings({"findings": []}, [], evidence, 10)["return"] == 0.98
    assert score_findings({"findings": [finding]}, [], evidence, 10)["return"] == -1.0


def test_matcher_tolerances_limits_and_exact_rewards() -> None:
    issue = {
        "issue_id": "A",
        "type": "duplicate_invoice",
        "affected_ids": ["R1"],
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1"],
    }
    finding = {
        "type": "duplicate_invoice",
        "affected_ids": ["R1"],
        "claimed_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1"],
        "explanation": "ok",
    }
    avail = {"D1"}
    issue_tolerance = {**issue, "tolerance_minor": 1}
    amount = {**finding, "claimed_amount_minor": 10001}
    assert score_findings({"findings": [amount]}, [issue_tolerance], avail, 10)["tp"] == 1
    amount["claimed_amount_minor"] = 10002
    assert score_findings({"findings": [amount]}, [issue_tolerance], avail, 10)["tp"] == 0

    for amount_value in ("10000", True):
        malformed = {**finding, "claimed_amount_minor": amount_value}
        result = score_findings({"findings": [malformed]}, [issue], avail, 10)
        assert not result["valid"] and result["return"] == -1.0

    duplicate_result = score_findings({"findings": [finding, finding]}, [issue], avail, 10)
    assert duplicate_result["tp"] == 1 and duplicate_result["fp"] == 1 and duplicate_result["fn"] == 0
    assert duplicate_result["precision"] == 0.5
    assert duplicate_result["recall"] == 1.0
    assert abs(duplicate_result["f1"] - 2 / 3) < 1e-12
    assert abs(duplicate_result["return"] - 47 / 150) < 1e-12

    for findings in (
        [finding] * 31,
        [{**finding, "evidence_ids": ["D1"] + [f"D{i}" for i in range(2, 12)]}],
        [{**finding, "explanation": "x" * 1001}],
    ):
        result = score_findings({"findings": findings}, [issue], avail, 10)
        assert not result["valid"] and result["return"] == -1.0

    assert compute_reward(1, 0, 0, 0)["return"] == 1.0
    assert compute_reward(1, 0, 0, 50)["return"] == 0.9
    assert compute_reward(1, 0, 0, 100)["return"] == 0.8


def test_matcher_mass_flagging_and_multi_issue_matching() -> None:
    issue = {
        "issue_id": "A",
        "type": "duplicate_invoice",
        "affected_ids": ["R1"],
        "expected_amount_minor": 10000,
        "currency": "KZT",
        "evidence_ids": ["D1"],
    }
    types = ["duplicate_invoice", "unallocated_or_partial_payment", "bad_fx_date_or_rate", "period_misclassification"]
    mass_findings = [
        {
            "type": kind,
            "affected_ids": [record_id],
            "claimed_amount_minor": 10000,
            "currency": "KZT",
            "evidence_ids": ["D1"],
            "explanation": "flag",
        }
        for record_id in ["R1", "R2", "R3", "R4", "R5"]
        for kind in types
    ]
    mass = score_findings({"findings": mass_findings}, [issue], {"D1"}, 10)
    assert mass["tp"] == 1 and mass["fp"] == 19 and mass["fn"] == 0
    assert abs(mass["return"] - (-871 / 1050)) < 1e-12

    issue_b = {
        "issue_id": "B",
        "type": "unallocated_or_partial_payment",
        "affected_ids": ["P1"],
        "expected_amount_minor": 200,
        "currency": "KZT",
        "evidence_ids": ["D1"],
    }
    finding_b = {
        "type": issue_b["type"],
        "affected_ids": issue_b["affected_ids"],
        "claimed_amount_minor": issue_b["expected_amount_minor"],
        "currency": issue_b["currency"],
        "evidence_ids": issue_b["evidence_ids"],
        "explanation": "ok",
    }
    two_issues = [issue, issue_b]
    one_correct = score_findings({"findings": [mass_findings[0]]}, two_issues, {"D1"}, 10)
    assert one_correct["tp"] == 1 and one_correct["fp"] == 0 and one_correct["fn"] == 1
    assert abs(one_correct["return"] - 47 / 150) < 1e-12
    one_false = {**finding_b, "affected_ids": ["P2"]}
    mixed = score_findings(
        {"findings": [mass_findings[0], one_false]},
        two_issues,
        {"D1"},
        10,
    )
    assert mixed["tp"] == 1 and mixed["fp"] == 1 and mixed["fn"] == 1
    assert mixed["return"] == -0.02

    graph_issues = [
        {**issue, "issue_id": "A"},
        {**issue_b, "issue_id": "B"},
    ]
    assert max_bipartite_matching(2, graph_issues, [(0, 0), (0, 1), (1, 0)]) == (2, [("A", 1), ("B", 0)])
    tied = max_bipartite_matching(2, graph_issues, [(1, 1), (0, 1), (1, 0), (0, 0)])
    assert tied == (2, [("A", 0), ("B", 1)])


def test_paired_reproducibility(tmp_path: Path) -> None:
    orig = tmp_path / "orig"
    perm = tmp_path / "perm"
    generate_episode(seed=77, config={"episode_id": "ep_77", "errors": ["duplicate_invoice"], "shuffle_records": False}, output_dir=orig)
    generate_episode(seed=77, config={"episode_id": "ep_77", "errors": ["duplicate_invoice"], "shuffle_records": True}, output_dir=perm)
    t_orig = json.loads((orig / "truth.json").read_text(encoding="utf-8"))["issues"][0]
    payload = {
        "episode_id": "ep_77",
        "findings": [{
            "type": t_orig["type"],
            "affected_ids": t_orig["affected_ids"],
            "claimed_amount_minor": t_orig["expected_amount_minor"],
            "currency": t_orig["currency"],
            "evidence_ids": t_orig["evidence_ids"],
            "explanation": "ok",
        }],
    }
    assert judge_adapter.score_episode(str(orig), payload, 5) == judge_adapter.score_episode(str(perm), payload, 5)


def test_same_seed_reproduces_every_episode_file(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    config = {
        "episode_id": "repeatable",
        "errors": ["unallocated_or_partial_payment"],
        "payment_issue_mode": "partial",
        "payment_discrepancy_minor": 1250,
        "relabel_seed": 91,
        "shuffle_records": True,
    }
    generate_episode(91, config, first)
    generate_episode(91, config, second)
    for name in ("manifest.json", "policy.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "truth.json", "provenance.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _oracle_payload(ep_path: Path) -> dict:
    truth = json.loads((ep_path / "truth.json").read_text(encoding="utf-8"))
    return {
        "episode_id": truth["episode_id"],
        "findings": [
            {
                "type": issue["type"],
                "affected_ids": issue["affected_ids"],
                "claimed_amount_minor": issue["expected_amount_minor"],
                "currency": issue["currency"],
                "evidence_ids": issue["evidence_ids"],
                "explanation": "Verified issue.",
            }
            for issue in truth["issues"]
        ],
    }


def test_relabel_pair_preserves_oracle_reward_and_changes_ids(tmp_path: Path) -> None:
    paths = [tmp_path / "a", tmp_path / "b"]
    for path, relabel_seed in zip(paths, (1, 2)):
        generate_episode(
            77,
            {"episode_id": "e", "errors": ["duplicate_invoice"], "relabel_seed": relabel_seed},
            path,
        )

    returns = [
        judge_adapter.score_episode(str(path), _oracle_payload(path), 5)["return"]
        for path in paths
    ]
    assert returns[0] == returns[1]

    mappings = [
        json.loads((path / "provenance.json").read_text(encoding="utf-8"))["id_mapping"]
        for path in paths
    ]
    assert mappings[0] != mappings[1]
    provenance_events = [
        json.loads((path / "provenance.json").read_text(encoding="utf-8"))["events"]
        for path in paths
    ]
    assert provenance_events[0] == provenance_events[1]


def test_relabel_maps_all_id_namespaces_and_references(tmp_path: Path) -> None:
    paths = [tmp_path / "left", tmp_path / "right"]
    for path, relabel_seed in zip(paths, (13, 27)):
        generate_episode(
            127,
            {
                "episode_id": "pair",
                "errors": ["duplicate_invoice", "unallocated_or_partial_payment"],
                "relabel_seed": relabel_seed,
            },
            path,
        )

    mappings = [
        json.loads((path / "provenance.json").read_text(encoding="utf-8"))["id_mapping"]
        for path in paths
    ]
    assert all(mappings[0][namespace] != mappings[1][namespace] for namespace in mappings[0])

    for path, mapping in zip(paths, mappings):
        invoices = [json.loads(line) for line in (path / "invoices.jsonl").read_text(encoding="utf-8").splitlines()]
        payments = [json.loads(line) for line in (path / "payments.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = list(csv.DictReader((path / "ledger.csv").open(newline="", encoding="utf-8")))
        invoice_ids = {row["invoice_id"] for row in invoices}
        payment_ids = {row["payment_id"] for row in payments}
        doc_ids = {row["doc_id"] for row in invoices + payments}
        row_ids = {row["row_id"] for row in rows}
        entity_ids = {row["entity_id"] for row in invoices + rows}
        assert set(mapping["invoices"].values()) == invoice_ids
        assert set(mapping["payments"].values()).issubset(payment_ids | {row["payment_id"] for row in rows})
        assert set(mapping["documents"].values()) == doc_ids
        assert set(mapping["ledger_rows"].values()) == row_ids
        assert set(mapping["entities"].values()) == entity_ids
        assert all(set(payment.get("invoice_refs") or []).issubset(invoice_ids) for payment in payments)
        assert all(not row["invoice_id"] or row["invoice_id"] in invoice_ids for row in rows)
        assert all(not row["payment_id"] or row["payment_id"].startswith("PAY-") for row in rows)

        truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
        evidence = doc_ids | row_ids
        known_ids = invoice_ids | payment_ids
        assert all(set(issue["affected_ids"]).issubset(known_ids) for issue in truth["issues"])
        assert all(set(issue["evidence_ids"]).issubset(evidence) for issue in truth["issues"])


def test_default_episode_matrix_is_auditable_and_semantically_consistent(tmp_path: Path) -> None:
    build_default_episodes(tmp_path)
    train_paths = sorted((tmp_path / "episodes" / "train").iterdir())
    dev_paths = sorted((tmp_path / "episodes" / "dev").iterdir())
    episode_paths = train_paths + dev_paths
    assert len(train_paths) == len(dev_paths) == 10

    type_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}
    for path in episode_paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        policy = json.loads((path / "policy.json").read_text(encoding="utf-8"))
        truth = json.loads((path / "truth.json").read_text(encoding="utf-8"))
        provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
        invoices = [json.loads(line) for line in (path / "invoices.jsonl").read_text(encoding="utf-8").splitlines()]
        payments = [json.loads(line) for line in (path / "payments.jsonl").read_text(encoding="utf-8").splitlines()]
        fx_rows = list(csv.DictReader((path / "fx.csv").open(newline="", encoding="utf-8")))
        final_ledger = list(csv.DictReader((path / "ledger.csv").open(newline="", encoding="utf-8")))
        baseline_ledger = provenance["baseline"]["ledger"]
        for issue in truth["issues"]:
            type_counts[issue["type"]] = type_counts.get(issue["type"], 0) + 1
        for event in provenance["events"]:
            name = event["event"]
            event_counts[name] = event_counts.get(name, 0) + 1

        assert manifest["counts"]["ledger_rows"] == len(final_ledger)
        assert manifest["counts"]["invoices"] == len(invoices)
        assert manifest["counts"]["payments"] == len(payments)
        assert manifest["counts"]["fx_rows"] == len(fx_rows)
        assert policy["type_tolerances_minor"] == {kind: 0 for kind in policy["supported_error_types"]}
        assert "seed" not in manifest and "seed" not in policy
        for ledger in (baseline_ledger, final_ledger):
            by_currency = {}
            for row in ledger:
                by_currency.setdefault(row["currency"], [0, 0])
                by_currency[row["currency"]][0] += int(row["debit_minor"])
                by_currency[row["currency"]][1] += int(row["credit_minor"])
            assert all(debits == credits for debits, credits in by_currency.values())
        assert [row["timestamp"] for row in final_ledger] == sorted(
            row["timestamp"] for row in final_ledger
        )

        invoice_ids = {invoice["invoice_id"] for invoice in invoices}
        payment_ids = {payment["payment_id"] for payment in payments}
        doc_ids = {invoice["doc_id"] for invoice in invoices} | {payment["doc_id"] for payment in payments}
        for issue in truth["issues"]:
            assert set(issue["affected_ids"]).issubset(invoice_ids | payment_ids)
            assert set(issue["evidence_ids"]).issubset(doc_ids | {row["row_id"] for row in final_ledger})
            assert all(
                set(evidence_set).issubset(doc_ids | {row["row_id"] for row in final_ledger})
                for evidence_set in issue.get("valid_evidence_sets", [])
            )
            if issue["type"] == "duplicate_invoice":
                left, right = [next(inv for inv in invoices if inv["invoice_id"] == iid) for iid in issue["affected_ids"]]
                assert left["entity_id"] == right["entity_id"]
                assert left["issued_at"] == right["issued_at"]
                assert left["total_minor"] == right["total_minor"]
                assert left["currency"] == right["currency"]
                assert left["line_items"] == right["line_items"]
            elif issue["type"] == "unallocated_or_partial_payment":
                target_payments = [payment for payment in payments if payment["payment_id"] in issue["affected_ids"]]
                if all(not payment.get("invoice_refs") for payment in target_payments):
                    assert len(target_payments) == 1
                    assert target_payments[0]["amount_minor"] == issue["expected_amount_minor"]
                    assert target_payments[0]["doc_id"] in issue["evidence_ids"]
                else:
                    invoice_doc = issue["evidence_ids"][0]
                    target_invoice = next(inv for inv in invoices if inv["doc_id"] == invoice_doc)
                    paid = sum(
                        int(payment["amount_minor"])
                        for payment in payments
                        if target_invoice["invoice_id"] in (payment.get("invoice_refs") or [])
                        and payment["currency"] == target_invoice["currency"]
                        and payment.get("payment_type") != "refund"
                    )
                    assert target_invoice["total_minor"] - paid == issue["expected_amount_minor"]
            elif issue["type"] == "bad_fx_date_or_rate":
                event = next(event for event in provenance["events"] if event.get("issue_id") == issue["issue_id"])
                assert abs(event["recorded_base_minor"] - event["expected_base_minor"]) == issue["expected_amount_minor"]
                invoice = next(invoice for invoice in invoices if invoice["invoice_id"] == issue["affected_ids"][0])
                fx_rate = next(
                    row["rate"]
                    for row in fx_rows
                    if row["date"] == event["recorded_rate_date"]
                    and row["source_currency"] == invoice["currency"]
                    and row["target_currency"] == manifest["base_currency"]
                )
                assert event["recorded_rate"] == fx_rate
                assert event["recorded_rate_date"] != event["date"]
                evidence = issue["valid_evidence_sets"][0]
                assert invoice["doc_id"] in evidence
                assert all(line["description"].startswith("Line ") for line in invoice["line_items"])
                assert any(
                    row["row_id"] in evidence
                    and row["invoice_id"] == invoice["invoice_id"]
                    and row["account"] == "ACCOUNTS_RECEIVABLE"
                    for row in final_ledger
                )
                settlement = next(
                    payment
                    for payment in payments
                    if invoice["invoice_id"] in payment.get("invoice_refs", [])
                    and payment.get("payment_type") == "payment"
                )
                assert any(
                    row["payment_id"] == settlement["payment_id"]
                    and row["account"] == "CASH_BANK"
                    for row in final_ledger
                )
                assert any(
                    row["payment_id"] == settlement["payment_id"]
                    and row["account"] == "ACCOUNTS_RECEIVABLE"
                    and int(row["credit_minor"]) > 0
                    for row in final_ledger
                )
            elif issue["type"] == "period_misclassification":
                invoice = next(inv for inv in invoices if inv["invoice_id"] == issue["affected_ids"][0])
                assert invoice["issued_at"] < manifest["period_start"]
                assert all(line["description"].startswith("Line ") for line in invoice["line_items"])
                assert any(
                    row["row_id"] in issue["valid_evidence_sets"][0]
                    and row["invoice_id"] == invoice["invoice_id"]
                    and row["account"] == "ACCOUNTS_RECEIVABLE"
                    and manifest["period_start"] <= row["timestamp"] <= manifest["period_end"]
                    for row in final_ledger
                )
                settlement = next(
                    payment
                    for payment in payments
                    if invoice["invoice_id"] in payment.get("invoice_refs", [])
                    and payment.get("payment_type") == "payment"
                )
                assert any(
                    row["payment_id"] == settlement["payment_id"]
                    and row["account"] == "CASH_BANK"
                    for row in final_ledger
                )
                assert any(
                    row["payment_id"] == settlement["payment_id"]
                    and row["account"] == "ACCOUNTS_RECEIVABLE"
                    and int(row["credit_minor"]) > 0
                    for row in final_ledger
                )

    assert all(not json.loads((path / "truth.json").read_text(encoding="utf-8"))["issues"] for path in train_paths[:5])
    assert type_counts == {
        "duplicate_invoice": 6,
        "unallocated_or_partial_payment": 8,
        "bad_fx_date_or_rate": 5,
        "period_misclassification": 5,
    }
    for split in (train_paths, dev_paths):
        split_types = {
            issue["type"]
            for path in split
            for issue in json.loads((path / "truth.json").read_text(encoding="utf-8"))["issues"]
        }
        assert split_types == {
            "duplicate_invoice",
            "unallocated_or_partial_payment",
            "bad_fx_date_or_rate",
            "period_misclassification",
        }
    large_train_counts = {
        path.name: json.loads((path / "manifest.json").read_text(encoding="utf-8"))["counts"]["invoices"]
        for path in train_paths[6:]
    }
    assert large_train_counts == {"train_07": 60, "train_08": 80, "train_09": 100, "train_10": 150}
    assert all(
        json.loads((path / "provenance.json").read_text(encoding="utf-8"))["config"]["refund_count"] > 0
        for path in train_paths[6:]
    )
    assert event_counts["interaction_partial_fx_payment_near_period_end"] == 3
    assert event_counts["interaction_duplicate_invoice_with_partial_payment"] == 2
    refund_payments = [
        json.loads(line)
        for line in (train_paths[2] / "payments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(payment.get("payment_type") == "refund" for payment in refund_payments)
    repeat_invoices = [
        json.loads(line)
        for line in (train_paths[3] / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        left["entity_id"] == right["entity_id"]
        and left["currency"] == right["currency"]
        and left["total_minor"] == right["total_minor"]
        and left["issued_at"] != right["issued_at"]
        for index, left in enumerate(repeat_invoices)
        for right in repeat_invoices[index + 1:]
    )
    boundary_manifest = json.loads((train_paths[4] / "manifest.json").read_text(encoding="utf-8"))
    boundary_invoices = [
        json.loads(line)
        for line in (train_paths[4] / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    boundary_invoice = next(
        invoice for invoice in boundary_invoices
        if invoice["issued_at"] == boundary_manifest["period_start"]
    )
    assert boundary_invoice["due_at"] == boundary_manifest["period_end"]
    boundary_payments = [
        json.loads(line)
        for line in (train_paths[4] / "payments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        boundary_invoice["invoice_id"] in payment.get("invoice_refs", [])
        and payment["timestamp"] == boundary_manifest["period_end"]
        for payment in boundary_payments
    )
    for path in train_paths[:5]:
        env = make_env()
        obs, _ = env.reset(config={"mode": "closed", "episode_dir": str(path)})
        assert run_simple_reconciliation_policy(env, obs)["findings"] == []
    for path in dev_paths[1:5]:
        env = make_env()
        obs, _ = env.reset(config={"mode": "closed", "episode_dir": str(path)})
        findings = run_simple_reconciliation_policy(env, obs)["findings"]
        score = judge_adapter.score_episode(str(path), {"episode_id": path.name, "findings": findings}, env.get_inspection_steps())
        assert score["tp"] == 1 and score["fp"] == 0 and score["fn"] == 0


def test_reconciliation_reads_all_pages_for_large_episodes(tmp_path: Path) -> None:
    ep_path = tmp_path / "many_invoices"
    generate_episode(124, {"episode_id": "many_invoices", "num_invoices": 150, "errors": []}, ep_path)
    env = make_env()
    obs, _ = env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})
    findings = run_simple_reconciliation_policy(env, obs)
    assert findings["findings"] == []
    assert env.get_inspection_steps() < 100


def test_mixed_error_scenarios_are_unique_across_200_seeds(tmp_path: Path) -> None:
    config = {
        "episode_id": "stress",
        "num_invoices": 24,
        "errors": [
            "duplicate_invoice",
            "unallocated_or_partial_payment",
            "bad_fx_date_or_rate",
            "period_misclassification",
        ],
        "payment_issue_mode": "partial",
        "payment_discrepancy_minor": 1000,
        "split_payment_count": 4,
        "refund_count": 3,
        "interactions": [
            "partial_fx_payment_near_period_end",
            "duplicate_invoice_with_partial_payment",
        ],
        "scenario_params": {
            "partial_fx_payment_near_period_end": {
                "days_before_period_end": 1,
                "shortfall_minor": 500,
            },
            "duplicate_invoice_with_partial_payment": {"shortfall_minor": 750},
        },
    }
    for seed in range(200):
        episode_dir = tmp_path / f"seed_{seed}"
        generate_episode(seed, config, episode_dir)
        invoices = [
            json.loads(line)
            for line in (episode_dir / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        payments = [
            json.loads(line)
            for line in (episode_dir / "payments.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        truth = json.loads((episode_dir / "truth.json").read_text(encoding="utf-8"))

        invoice_ids = [invoice["invoice_id"] for invoice in invoices]
        payment_ids = [payment["payment_id"] for payment in payments]
        document_ids = [invoice["doc_id"] for invoice in invoices] + [
            payment["doc_id"] for payment in payments
        ]
        assert len(invoice_ids) == len(set(invoice_ids)), seed
        assert len(payment_ids) == len(set(payment_ids)), seed
        assert len(document_ids) == len(set(document_ids)), seed
        affected = [
            (issue["type"], tuple(sorted(issue["affected_ids"])))
            for issue in truth["issues"]
        ]
        assert len(affected) == len(set(affected)), seed


def test_reset_contract_pagination_and_copy_isolation(tmp_path: Path) -> None:
    ep_path = tmp_path / "contract"
    generate_episode(129, {"episode_id": "contract", "errors": [], "min_ledger_rows": 105}, ep_path)
    env = make_env()
    obs, info = env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})
    json.dumps([obs, info], allow_nan=False)
    assert set(("task", "manifest", "policy", "remaining_steps")).issubset(obs)
    assert obs["remaining_steps"] == 100
    assert obs["manifest"]["counts"]["ledger_rows"] >= 105
    assert "truth" not in json.dumps(obs) and "provenance" not in json.dumps(obs) and "seed" not in json.dumps(obs)
    assert env._truth["issues"] == []

    pages = []
    cursor = None
    while True:
        page, _, terminated, truncated, _ = env.step({
            "name": "list_records",
            "args": {"table": "ledger", "cursor": cursor, "limit": 50},
        })
        assert not terminated and not truncated
        pages.extend(page["result"]["records"])
        cursor = page["result"]["next_cursor"]
        if cursor is None:
            break
    expected_rows = obs["manifest"]["counts"]["ledger_rows"]
    assert len(pages) == expected_rows
    assert len({row["row_id"] for row in pages}) == expected_rows

    first, *_ = env.step({"name": "list_records", "args": {"table": "ledger", "limit": 1}})
    first["result"]["records"][0]["account"] = "MUTATED"
    repeated, *_ = env.step({"name": "list_records", "args": {"table": "ledger", "limit": 1}})
    assert repeated["result"]["records"][0]["account"] != "MUTATED"

    invoice = json.loads((ep_path / "invoices.jsonl").read_text(encoding="utf-8").splitlines()[0])
    record, *_ = env.step({"name": "get_record", "args": {"table": "invoices", "id": invoice["invoice_id"]}})
    document, *_ = env.step({"name": "get_document", "args": {"doc_id": invoice["doc_id"]}})
    assert record["result"] == document["result"]
    json.dumps([record, document], allow_nan=False)


def test_closed_reset_generates_temporary_episode_without_truth() -> None:
    env = make_env()
    obs, info = env.reset(
        seed=456,
        config={
            "mode": "closed",
            "team_config": {"episode_id": "generated_closed", "errors": ["duplicate_invoice"]},
        },
    )
    assert info["mode"] == "closed"
    assert obs["manifest"]["episode_id"] == "generated_closed"
    assert "truth" not in json.dumps(obs) and "provenance" not in json.dumps(obs)
    assert env._truth["issues"] == []

    flat_obs, _ = env.reset(
        seed=457,
        config={"mode": "closed", "episode_id": "flat_closed", "errors": []},
    )
    assert flat_obs["manifest"]["episode_id"] == "flat_closed"
    assert env._truth["issues"] == []


def test_invalid_unicode_and_oversized_cursors_do_not_crash(tmp_path: Path) -> None:
    ep_path = tmp_path / "cursor"
    generate_episode(128, {"episode_id": "cursor", "errors": []}, ep_path)
    env = make_env()
    env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})

    for cursor in ("²", "9" * 4300):
        obs, reward, terminated, truncated, _ = env.step({
            "name": "list_records",
            "args": {"table": "ledger", "cursor": cursor, "limit": 1},
        })
        assert reward == 0.0 and not terminated and not truncated
        assert obs["error"]["code"] == "INVALID_CURSOR"


def test_calculator_handles_infinite_exponents_and_large_integers(tmp_path: Path) -> None:
    ep_path = tmp_path / "calculator"
    generate_episode(127, {"episode_id": "calculator", "errors": []}, ep_path)
    env = make_env()
    env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})

    invalid, *_ = env.step({"name": "calculator", "args": {"expression": "2**1e999"}})
    assert invalid["error"]["code"] == "CALCULATOR_ERROR"
    large, *_ = env.step({"name": "calculator", "args": {"expression": "10**20*10**20"}})
    assert large["result"] == "10000000000000000000000000000000000000000"


def test_findings_episode_id_is_required_and_must_match(tmp_path: Path) -> None:
    ep_path = tmp_path / "episode_id"
    generate_episode(126, {"episode_id": "expected", "errors": []}, ep_path)
    env = make_env()
    env.reset(config={"mode": "open", "episode_dir": str(ep_path)})

    obs, reward, terminated, truncated, _ = env.step({
        "name": "submit",
        "args": {"findings": {"episode_id": "WRONG", "findings": []}},
    })
    assert reward == -1.0 and terminated and not truncated
    assert obs["error"]["code"] == "INVALID_FINDINGS_STRUCTURE"

    missing_result = score_findings(
        {"findings": []},
        [],
        set(),
        0,
        expected_episode_id="expected",
    )
    assert not missing_result["valid"] and missing_result["return"] == -1.0
    assert judge_adapter.score_episode(
        str(ep_path),
        {"episode_id": "WRONG", "findings": []},
        0,
    )["return"] == -1


def test_invoice_count_is_not_silently_truncated(tmp_path: Path) -> None:
    large_path = tmp_path / "large"
    generate_episode(125, {"episode_id": "large", "num_invoices": 150, "errors": []}, large_path)
    invoices = (large_path / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(invoices) == 150

    try:
        generate_episode(125, {"episode_id": "too_large", "num_invoices": 501}, tmp_path / "too_large")
    except ValueError as exc:
        assert "num_invoices must be in range 6..500" in str(exc)
    else:
        raise AssertionError("num_invoices above the supported maximum must fail")


def test_submission_is_final_and_immutable(tmp_path: Path) -> None:
    ep_path = tmp_path / "final"
    generate_episode(130, {"episode_id": "final", "errors": []}, ep_path)
    env = make_env()
    env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})
    payload = {"episode_id": "final", "findings": []}
    _, reward, terminated, truncated, _ = env.step({"name": "submit", "args": {"findings": payload}})
    assert reward == 0.0 and terminated and not truncated
    payload["findings"].append({"unexpected": "mutation"})
    assert env.get_submitted_findings()["findings"] == []

    obs, second_reward, second_terminated, second_truncated, _ = env.step({
        "name": "submit",
        "args": {"findings": {"episode_id": "final", "findings": []}},
    })
    assert second_reward == 0.0 and second_terminated and not second_truncated
    assert obs["error"]["code"] == "EPISODE_FINISHED"
    assert env.get_submitted_findings()["findings"] == []


def test_reconciliation_finds_duplicate_and_payment_discrepancy(tmp_path: Path) -> None:
    ep_path = tmp_path / "reconcile"
    generate_episode(
        88,
        {"episode_id": "reconcile", "errors": ["duplicate_invoice", "unallocated_or_partial_payment"]},
        ep_path,
    )
    truth_issues = json.loads((ep_path / "truth.json").read_text(encoding="utf-8"))["issues"]

    env = make_env()
    obs, _ = env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})
    findings = run_simple_reconciliation_policy(env, obs)["findings"]
    duplicate_truth = next(issue for issue in truth_issues if issue["type"] == "duplicate_invoice")
    payment_truth = next(issue for issue in truth_issues if issue["type"] == "unallocated_or_partial_payment")
    assert any(
        finding["type"] == "duplicate_invoice"
        and set(finding["affected_ids"]) == set(duplicate_truth["affected_ids"])
        for finding in findings
    )
    invoice_doc_id = payment_truth["evidence_ids"][-1]
    assert any(
        finding["type"] == "unallocated_or_partial_payment"
        and invoice_doc_id in finding["evidence_ids"]
        for finding in findings
    )
    score = judge_adapter.score_episode(str(ep_path), {"episode_id": "reconcile", "findings": findings}, 0)
    assert score["tp"] == 2 and score["fp"] == 0 and score["fn"] == 0

    partial_path = tmp_path / "partial"
    generate_episode(
        89,
        {
            "episode_id": "partial",
            "errors": ["unallocated_or_partial_payment"],
            "payment_issue_mode": "partial",
            "payment_discrepancy_minor": 1250,
            "refund_count": 0,
        },
        partial_path,
    )
    partial_env = make_env()
    partial_obs, _ = partial_env.reset(config={"mode": "closed", "episode_dir": str(partial_path)})
    partial_payload = run_simple_reconciliation_policy(partial_env, partial_obs)
    partial_score = judge_adapter.score_episode(
        str(partial_path),
        partial_payload,
        partial_env.get_inspection_steps(),
    )
    assert partial_score["tp"] == 1 and partial_score["fp"] == 0 and partial_score["fn"] == 0


def test_open_mode_rewards_valid_and_invalid_submissions(tmp_path: Path) -> None:
    ep_path = tmp_path / "open"
    generate_episode(91, {"episode_id": "open", "errors": ["duplicate_invoice"]}, ep_path)
    payload = _oracle_payload(ep_path)
    expected_reward = judge_adapter.score_episode(str(ep_path), payload, 0)["return"]

    env = make_env()
    env.reset(config={"mode": "open", "episode_dir": str(ep_path)})
    _, reward, terminated, truncated, _ = env.step({"name": "submit", "args": {"findings": payload}})
    assert reward == expected_reward and terminated and not truncated

    invalid_env = make_env()
    invalid_env.reset(config={"mode": "open", "episode_dir": str(ep_path)})
    obs, reward, terminated, truncated, _ = invalid_env.step({
        "name": "submit",
        "args": {"findings": "{not valid JSON"},
    })
    assert reward == -1.0 and terminated and not truncated
    assert obs["error"]["code"] == "INVALID_FINDINGS_STRUCTURE"

    closed_env = make_env()
    closed_env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})
    _, closed_reward, _, _, _ = closed_env.step({
        "name": "submit",
        "args": {"findings": "{not valid JSON"},
    })
    assert closed_reward == 0.0


def test_flag_everything_and_duplicate_findings_are_penalized(tmp_path: Path) -> None:
    ep_path = tmp_path / "flag_all"
    generate_episode(92, {"episode_id": "flag_all", "errors": ["duplicate_invoice"]}, ep_path)
    payload = _oracle_payload(ep_path)
    truth = json.loads((ep_path / "truth.json").read_text(encoding="utf-8"))["issues"]
    available = set()
    for name in ("invoices.jsonl", "payments.jsonl"):
        for line in (ep_path / name).read_text(encoding="utf-8").splitlines():
            available.add(json.loads(line)["doc_id"])

    correct = payload["findings"][0]
    duplicate_result = score_findings(
        {"episode_id": "flag_all", "findings": [correct, copy.deepcopy(correct)]},
        truth,
        available,
        0,
    )
    correct_result = score_findings(payload, truth, available, 0)
    assert duplicate_result["return"] < correct_result["return"]

    invoices = [json.loads(line) for line in (ep_path / "invoices.jsonl").read_text(encoding="utf-8").splitlines()]
    all_findings = [
        {
            "type": "duplicate_invoice",
            "affected_ids": [invoice["invoice_id"]],
            "claimed_amount_minor": invoice["total_minor"],
            "currency": invoice["currency"],
            "evidence_ids": [invoice["doc_id"]],
            "explanation": "Flag every invoice.",
        }
        for invoice in invoices
    ]
    all_result = score_findings(
        {"episode_id": "flag_all", "findings": all_findings},
        truth,
        available,
        0,
    )
    assert all_result["tp"] == 0


def test_fabricated_ids_evidence_and_currency_do_not_match() -> None:
    issue = {
        "issue_id": "A",
        "type": "duplicate_invoice",
        "affected_ids": ["INV-0001", "INV-0002"],
        "expected_amount_minor": 5000,
        "currency": "KZT",
        "evidence_ids": ["DOC-INV-0001", "DOC-INV-0002"],
    }
    valid = {
        "type": issue["type"],
        "affected_ids": issue["affected_ids"],
        "claimed_amount_minor": issue["expected_amount_minor"],
        "currency": issue["currency"],
        "evidence_ids": issue["evidence_ids"],
        "explanation": "ok",
    }
    available = set(issue["evidence_ids"])
    mutations = []
    fake_id = copy.deepcopy(valid)
    fake_id["affected_ids"][0] = "INV-FAKE"
    mutations.append(fake_id)
    fake_evidence = copy.deepcopy(valid)
    fake_evidence["evidence_ids"].append("DOC-OUTSIDE")
    mutations.append(fake_evidence)
    wrong_currency = copy.deepcopy(valid)
    wrong_currency["currency"] = "USD"
    mutations.append(wrong_currency)

    for finding in mutations:
        result = score_findings({"episode_id": "e", "findings": [finding]}, [issue], available, 0)
        assert result["tp"] == 0


def test_repeated_submit_is_rejected(tmp_path: Path) -> None:
    ep_path = tmp_path / "repeat"
    generate_episode(93, {"episode_id": "repeat", "errors": []}, ep_path)
    env = make_env()
    env.reset(config={"mode": "open", "episode_dir": str(ep_path)})
    payload = {"episode_id": "repeat", "findings": []}

    _, first_reward, first_terminated, _, _ = env.step({"name": "submit", "args": {"findings": payload}})
    obs, second_reward, second_terminated, _, _ = env.step({"name": "submit", "args": {"findings": payload}})
    assert first_reward == 1.0 and first_terminated
    assert second_reward == 0.0 and second_terminated
    assert obs["error"]["code"] == "EPISODE_FINISHED"


def test_episode_timeout_truncates(tmp_path: Path) -> None:
    ep_path = tmp_path / "timeout"
    generate_episode(94, {"episode_id": "timeout", "errors": []}, ep_path)
    env = make_env()
    env.reset(config={"mode": "closed", "episode_dir": str(ep_path)})
    env._start_time = time.monotonic() - TIME_LIMIT_SECONDS - 1

    obs, reward, terminated, truncated, _ = env.step({"name": "calculator", "args": {"expression": "1 + 1"}})
    assert reward == 0.0 and not terminated and truncated
    assert obs["error"]["code"] == "TIME_LIMIT_EXCEEDED"


def test_runner_default_includes_train_and_dev_episodes(tmp_path: Path) -> None:
    rows = run_benchmark(tmp_path, policy_name="random")
    assert len(rows) == 20
    assert len({row["episode_id"] for row in rows}) == 20
    assert any(row["episode_id"].startswith("dev_") for row in rows)
    metadata = json.loads((tmp_path / "artifacts" / "benchmark_metadata.json").read_text(encoding="utf-8"))
    assert metadata["limit_episodes"] == 20
    assert len(metadata["episodes"]) == 20
    assert metadata["package_version"] and metadata["schema_version"] and metadata["python_version"]
    assert all(row["status"] == "ok" for row in rows)


def test_isolated_policy_timeout_and_exception_are_reported(tmp_path: Path) -> None:
    ep_path = tmp_path / "policy_failure"
    generate_episode(123, {"episode_id": "policy_failure", "errors": []}, ep_path)
    env = make_env()
    trajectories: list[dict] = []
    logged = LoggedEnvWrapper(env, trajectories, "policy_failure")
    init_obs, _ = logged.reset(config={"mode": "closed", "episode_dir": str(ep_path)})

    _, timeout_status, timeout_error = _execute_policy_isolated(
        "random",
        init_obs,
        logged,
        timeout_seconds=0,
    )
    assert timeout_status == "timeout" and timeout_error

    _, error_status, error_message = _execute_policy_isolated(
        "not_a_policy",
        init_obs,
        logged,
        timeout_seconds=10,
    )
    assert error_status == "error" and "Unknown policy" in (error_message or "")


def test_benchmark_records_policy_failure_and_continues(tmp_path: Path, monkeypatch) -> None:
    def fail_policy(*args, **kwargs):
        return None, "error", "synthetic policy failure"

    monkeypatch.setattr(runner_module, "_execute_policy_isolated", fail_policy)
    rows = run_benchmark(tmp_path, policy_name="random", limit_episodes=1)
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error"] == "synthetic policy failure"
    assert rows[0]["return"] == -1.0