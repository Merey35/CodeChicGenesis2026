from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .env import AuditEnv, make_env
from .generator import generate_episode as _generate_episode
from .matcher import score_findings


def generate_episode(seed: int, config: dict[str, Any], output_dir: str) -> None:
    _generate_episode(seed=seed, config=config, output_dir=output_dir)


def _collect_available_evidence_ids(ep_path: Path) -> set[str]:
    ev: set[str] = set()
    ledger_p = ep_path / "ledger.csv"
    if ledger_p.exists():
        with ledger_p.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("row_id"):
                    ev.add(row["row_id"])
    for fname in ("invoices.jsonl", "payments.jsonl"):
        p = ep_path / fname
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    obj = json.loads(ln)
                    if obj.get("doc_id"):
                        ev.add(str(obj["doc_id"]))
    return ev


def score_episode(episode_dir: str, findings: dict[str, Any], inspection_steps: int) -> dict[str, Any]:
    ep_path = Path(episode_dir)
    truth_p = ep_path / "truth.json"
    pol_p = ep_path / "policy.json"
    truth = json.loads(truth_p.read_text(encoding="utf-8")) if truth_p.exists() else {}
    truth_issues = truth.get("issues", [])
    if pol_p.exists():
        policy = json.loads(pol_p.read_text(encoding="utf-8"))
        tol = policy.get(
            "type_tolerances_minor",
            int(policy.get("amount_tolerance_minor", 0)),
        )
    else:
        tol = 0
    avail = _collect_available_evidence_ids(ep_path) if ep_path.exists() else set()
    expected_episode_id = str(truth["episode_id"]) if "episode_id" in truth else None
    res = score_findings(
        findings,
        truth_issues,
        avail,
        inspection_steps,
        default_tolerance=tol,
        expected_episode_id=expected_episode_id,
    )
    if not res["valid"]:
        return {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1}
    return {
        "tp": int(res["tp"]),
        "fp": int(res["fp"]),
        "fn": int(res["fn"]),
        "precision": float(res["precision"]),
        "recall": float(res["recall"]),
        "f1": float(res["f1"]),
        "return": float(res["return"]),
    }