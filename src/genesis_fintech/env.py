from __future__ import annotations

import csv
import copy
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from .calculator import evaluate_expression
from .generator import generate_episode
from .matcher import MAX_INSPECTION_STEPS, score_findings, validate_findings_structure

VALID_TABLES = ("ledger", "invoices", "payments", "fx")
VALID_ACTIONS = ("list_records", "get_record", "get_document", "calculator", "submit")
TIME_LIMIT_SECONDS = 60.0


def _err(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


class AuditEnv:
    def __init__(self) -> None:
        self._mode = "closed"
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._manifest: dict[str, Any] = {}
        self._policy: dict[str, Any] = {}
        self._truth: dict[str, Any] = {}
        self._tables: dict[str, list[dict[str, Any]]] = {t: [] for t in VALID_TABLES}
        self._record_index: dict[tuple[str, str], dict[str, Any]] = {}
        self._doc_index: dict[str, dict[str, Any]] = {}
        self._available_evidence_ids: set[str] = set()
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._submitted_findings: dict[str, Any] | None = None
        self._start_time = 0.0

    def _cleanup_temp(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _load_episode_dir(self, ep_dir: Path) -> None:
        self._manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
        self._policy = json.loads((ep_dir / "policy.json").read_text(encoding="utf-8"))
        tp = ep_dir / "truth.json"
        self._truth = (
            json.loads(tp.read_text(encoding="utf-8"))
            if self._mode == "open" and tp.exists()
            else {"episode_id": self._manifest.get("episode_id", ""), "issues": []}
        )
        ledger = []
        with (ep_dir / "ledger.csv").open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ledger.append({
                    "row_id": row["row_id"],
                    "timestamp": row["timestamp"],
                    "account": row["account"],
                    "debit_minor": int(row["debit_minor"]),
                    "credit_minor": int(row["credit_minor"]),
                    "currency": row["currency"],
                    "entity_id": row["entity_id"],
                    "invoice_id": None if row["invoice_id"] == "" else row["invoice_id"],
                    "payment_id": None if row["payment_id"] == "" else row["payment_id"],
                })
        invoices = [
            json.loads(ln)
            for ln in (ep_dir / "invoices.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        payments = [
            json.loads(ln)
            for ln in (ep_dir / "payments.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        fx = []
        with (ep_dir / "fx.csv").open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fx.append({
                    "date": row["date"],
                    "source_currency": row["source_currency"],
                    "target_currency": row["target_currency"],
                    "rate": row["rate"],
                })
        self._tables = {"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx}
        self._record_index.clear()
        self._doc_index.clear()
        self._available_evidence_ids.clear()
        for r in ledger:
            self._record_index[("ledger", str(r["row_id"]))] = r
            self._available_evidence_ids.add(str(r["row_id"]))
        for inv in invoices:
            self._record_index[("invoices", str(inv["invoice_id"]))] = inv
            self._doc_index[str(inv["doc_id"])] = inv
            self._available_evidence_ids.add(str(inv["doc_id"]))
        for pay in payments:
            self._record_index[("payments", str(pay["payment_id"]))] = pay
            self._doc_index[str(pay["doc_id"])] = pay
            self._available_evidence_ids.add(str(pay["doc_id"]))
        for idx, fx_row in enumerate(fx):
            self._record_index[
                ("fx", f"{fx_row['date']}:{fx_row['source_currency']}:{fx_row['target_currency']}")
            ] = fx_row
            self._record_index[("fx", str(idx))] = fx_row

    def reset(
        self,
        seed: int = 0,
        config: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg = dict(config or {})
        self._mode = str(cfg.get("mode", "closed"))
        if self._mode not in ("open", "closed"):
            raise ValueError("mode must be 'open' or 'closed'")
        ep_dir_val = cfg.get("episode_dir")
        configured_team = cfg.get("team_config")
        team_cfg = (
            dict(configured_team)
            if configured_team is not None
            else {key: value for key, value in cfg.items() if key not in {"mode", "episode_dir"}}
        )

        if ep_dir_val:
            self._cleanup_temp()
            self._load_episode_dir(Path(ep_dir_val))
        else:
            self._cleanup_temp()
            self._temp_dir = tempfile.TemporaryDirectory(prefix="genesis_episode_")
            gen_path = Path(self._temp_dir.name) / f"ep_{seed}"
            generate_episode(seed=seed, config=team_cfg, output_dir=gen_path)
            self._load_episode_dir(gen_path)

        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._submitted_findings = None
        self._start_time = time.monotonic()

        obs = {
            "task": "Detect financial audit discrepancies across ledger, invoices, payments, and FX data and submit structured findings with evidence IDs.",
            "manifest": copy.deepcopy(self._manifest),
            "policy": {
                "tables": list(VALID_TABLES),
                "actions": list(VALID_ACTIONS),
                "max_steps": MAX_INSPECTION_STEPS,
                "max_page_size": 50,
                "fx_date_selection": self._policy.get("fx_date_selection"),
                "fx_rounding": self._policy.get("fx_rounding"),
                "period_inclusive": self._policy.get("period_inclusive"),
                "amount_tolerance_minor": self._policy.get("amount_tolerance_minor", 0),
                "type_tolerances_minor": copy.deepcopy(
                    self._policy.get("type_tolerances_minor", {})
                ),
                "currency_exponents": self._manifest.get("currency_exponents", {}),
                "supported_error_types": self._policy.get("supported_error_types", []),
            },
            "remaining_steps": MAX_INSPECTION_STEPS,
        }
        return obs, {"mode": self._mode, "episode_id": self._manifest.get("episode_id"), "step": 0}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._terminated and not self._truncated and (time.monotonic() - self._start_time) > TIME_LIMIT_SECONDS:
            self._truncated = True
            return (
                {"result": None, "error": _err("TIME_LIMIT_EXCEEDED", "Episode time limit exceeded"), "remaining_steps": max(0, MAX_INSPECTION_STEPS - self._step_count)},
                0.0,
                False,
                True,
                {"step": self._step_count},
            )
        if self._terminated or self._truncated:
            return (
                {"result": None, "error": _err("EPISODE_FINISHED", "Episode already finished"), "remaining_steps": max(0, MAX_INSPECTION_STEPS - self._step_count)},
                0.0,
                self._terminated,
                self._truncated,
                {"step": self._step_count},
            )

        self._step_count += 1
        cur_step = self._step_count
        rem = max(0, MAX_INSPECTION_STEPS - cur_step)

        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            obs = {"result": None, "error": _err("INVALID_ACTION", "Action must be an object with name"), "remaining_steps": rem}
            if cur_step >= MAX_INSPECTION_STEPS:
                self._truncated = True
                return obs, 0.0, False, True, {"step": cur_step}
            return obs, 0.0, False, False, {"step": cur_step}

        name = action["name"]
        args = action.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict) or name not in VALID_ACTIONS:
            obs = {"result": None, "error": _err("UNKNOWN_ACTION", "Invalid action or arguments"), "remaining_steps": rem}
            if cur_step >= MAX_INSPECTION_STEPS:
                self._truncated = True
                return obs, 0.0, False, True, {"step": cur_step}
            return obs, 0.0, False, False, {"step": cur_step}

        if name == "submit":
            findings_obj = args.get("findings")
            ok, reason = validate_findings_structure(
                findings_obj,
                expected_episode_id=str(self._manifest.get("episode_id", "")),
            )
            reward = 0.0
            if self._mode == "open":
                result = score_findings(
                    findings_obj,
                    self._truth.get("issues", []),
                    self._available_evidence_ids,
                    max(0, cur_step - 1),
                    self._policy.get(
                        "type_tolerances_minor",
                        int(self._policy.get("amount_tolerance_minor", 0)),
                    ),
                    expected_episode_id=str(self._manifest.get("episode_id", "")),
                )
                reward = float(result["return"])
            self._terminated = True
            if ok:
                self._submitted_findings = copy.deepcopy(findings_obj)
                return {"result": {"accepted": True}, "error": None, "remaining_steps": rem}, reward, True, False, {"step": cur_step}
            self._submitted_findings = {"episode_id": self._manifest.get("episode_id", ""), "findings": [], "_invalid": True}
            return {"result": None, "error": _err("INVALID_FINDINGS_STRUCTURE", reason or "Invalid structure"), "remaining_steps": rem}, reward, True, False, {"step": cur_step}

        if cur_step >= MAX_INSPECTION_STEPS:
            self._truncated = True
            self._submitted_findings = {"episode_id": self._manifest.get("episode_id", ""), "findings": []}
            return {"result": None, "error": _err("MAX_STEPS_REACHED", "100 steps reached without submit"), "remaining_steps": 0}, 0.0, False, True, {"step": cur_step}

        if name == "list_records":
            table = args.get("table")
            cursor = args.get("cursor")
            limit = args.get("limit", 50)
            if table not in VALID_TABLES:
                return {"result": None, "error": _err("INVALID_TABLE", "Unknown table"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
            if type(limit) is not int or limit < 1 or limit > 50:
                return {"result": None, "error": _err("INVALID_LIMIT", "limit must be 1..50"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
            start = 0
            if cursor is not None:
                if (
                    not isinstance(cursor, str)
                    or not cursor.isascii()
                    or not cursor.isdecimal()
                    or len(cursor) > 9
                ):
                    return {"result": None, "error": _err("INVALID_CURSOR", "Bad cursor"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
                start = int(cursor)
            rows = self._tables[table]
            if start < 0 or start > len(rows):
                return {"result": None, "error": _err("INVALID_CURSOR", "Cursor out of range"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
            page = rows[start : start + limit]
            nxt = str(start + len(page)) if (start + len(page)) < len(rows) else None
            return {"result": {"records": copy.deepcopy(page), "next_cursor": nxt}, "error": None, "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}

        if name == "get_record":
            table, rec_id = args.get("table"), args.get("id")
            if table not in VALID_TABLES:
                return {"result": None, "error": _err("INVALID_TABLE", "Unknown table"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
            if not isinstance(rec_id, str) or (table, rec_id) not in self._record_index:
                return {"result": None, "error": _err("NOT_FOUND", "Record not found"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
            return {"result": copy.deepcopy(self._record_index[(table, rec_id)]), "error": None, "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}

        if name == "get_document":
            doc_id = args.get("doc_id")
            if not isinstance(doc_id, str) or doc_id not in self._doc_index:
                return {"result": None, "error": _err("NOT_FOUND", "Document not found"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
            return {"result": copy.deepcopy(self._doc_index[doc_id]), "error": None, "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}

        if name == "calculator":
            try:
                val = evaluate_expression(args.get("expression"))
                return {"result": val, "error": None, "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}
            except ValueError as exc:
                return {"result": None, "error": _err("CALCULATOR_ERROR", str(exc)), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}

        return {"result": None, "error": _err("UNHANDLED_ACTION", "Unhandled"), "remaining_steps": rem}, 0.0, False, False, {"step": cur_step}

    def get_submitted_findings(self) -> dict[str, Any]:
        return (
            copy.deepcopy(self._submitted_findings)
            if self._submitted_findings is not None
            else {"episode_id": self._manifest.get("episode_id", ""), "findings": []}
        )

    def get_inspection_steps(self) -> int:
        if self._terminated and self._submitted_findings is not None:
            return max(0, self._step_count - 1)
        return self._step_count


def make_env() -> AuditEnv:
    return AuditEnv()