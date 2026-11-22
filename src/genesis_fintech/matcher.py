from __future__ import annotations

from typing import Any

MAX_FINDINGS = 30
MAX_EVIDENCE_PER_FINDING = 10
MAX_EXPLANATION_CHARS = 1000
MAX_INSPECTION_STEPS = 100


def _is_strict_int(val: Any) -> bool:
    return type(val) is int


def validate_findings_structure(
    payload: Any,
    expected_episode_id: str | None = None,
) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "INVALID_STRUCTURE: findings payload must be a JSON object"
    if expected_episode_id is not None and payload.get("episode_id") != expected_episode_id:
        return False, "INVALID_EPISODE_ID: episode_id missing or does not match"
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return False, "INVALID_STRUCTURE: findings must be a list"
    if len(findings) > MAX_FINDINGS:
        return False, f"LIMIT_EXCEEDED: at most {MAX_FINDINGS} findings allowed"
    for idx, item in enumerate(findings):
        if not isinstance(item, dict):
            return False, f"INVALID_STRUCTURE: finding[{idx}] must be an object"
        for k in ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"):
            if k not in item:
                return False, f"INVALID_STRUCTURE: finding[{idx}] missing {k}"
        if not isinstance(item["type"], str) or not item["type"]:
            return False, f"INVALID_STRUCTURE: finding[{idx}].type invalid"
        aff = item["affected_ids"]
        if not isinstance(aff, list) or not all(isinstance(x, str) and len(x) > 0 for x in aff):
            return False, f"INVALID_STRUCTURE: finding[{idx}].affected_ids invalid"
        if not _is_strict_int(item["claimed_amount_minor"]):
            return False, f"INVALID_STRUCTURE: finding[{idx}].claimed_amount_minor must be int"
        if not isinstance(item["currency"], str) or not item["currency"]:
            return False, f"INVALID_STRUCTURE: finding[{idx}].currency invalid"
        ev = item["evidence_ids"]
        if not isinstance(ev, list) or not all(isinstance(x, str) and len(x) > 0 for x in ev):
            return False, f"INVALID_STRUCTURE: finding[{idx}].evidence_ids invalid"
        if len(ev) > MAX_EVIDENCE_PER_FINDING:
            return False, f"LIMIT_EXCEEDED: finding[{idx}].evidence_ids > {MAX_EVIDENCE_PER_FINDING}"
        expl = item["explanation"]
        if not isinstance(expl, str) or len(expl) > MAX_EXPLANATION_CHARS:
            return False, f"LIMIT_EXCEEDED: finding[{idx}].explanation > {MAX_EXPLANATION_CHARS}"
    return True, None


def _is_match(
    finding: dict[str, Any],
    issue: dict[str, Any],
    available_evidence_ids: set[str],
    default_tolerance: int | dict[str, int] = 0,
) -> bool:
    if finding["type"] != issue["type"] or finding["currency"] != issue["currency"]:
        return False
    if set(finding["affected_ids"]) != set(issue["affected_ids"]):
        return False
    f_ev = set(finding["evidence_ids"])
    if not f_ev.issubset(available_evidence_ids):
        return False
    valid_sets = issue.get("valid_evidence_sets")
    if isinstance(valid_sets, list) and len(valid_sets) > 0:
        if not any(set(vs).issubset(f_ev) for vs in valid_sets):
            return False
    else:
        if not set(issue.get("evidence_ids", [])).issubset(f_ev):
            return False
    if isinstance(default_tolerance, dict):
        type_tolerance = default_tolerance.get(finding["type"], 0)
    else:
        type_tolerance = default_tolerance
    tol = max(0, int(issue.get("tolerance_minor", type_tolerance)))
    if abs(int(finding["claimed_amount_minor"]) - int(issue["expected_amount_minor"])) > tol:
        return False
    return True


def max_bipartite_matching(
    num_findings: int,
    issues: list[dict[str, Any]],
    edges: list[tuple[int, int]],
) -> tuple[int, list[tuple[str, int]]]:
    adj: dict[int, list[int]] = {i: [] for i in range(num_findings)}
    for f_idx, i_idx in edges:
        if 0 <= f_idx < num_findings and 0 <= i_idx < len(issues):
            adj[f_idx].append(i_idx)

    match_i_to_f: dict[int, int] = {}

    def dfs(f_idx: int, seen: set[int]) -> bool:
        for i_idx in adj.get(f_idx, []):
            if i_idx in seen:
                continue
            seen.add(i_idx)
            prev = match_i_to_f.get(i_idx)
            if prev is None or dfs(prev, seen):
                match_i_to_f[i_idx] = f_idx
                return True
        return False

    for f_idx in range(num_findings):
        dfs(f_idx, set())

    tp = len(match_i_to_f)
    issue_order = sorted(range(len(issues)), key=lambda k: str(issues[k].get("issue_id", "")))
    cands: dict[int, list[int]] = {i_idx: [] for i_idx in issue_order}
    for f_idx, i_idx in edges:
        if i_idx in cands:
            cands[i_idx].append(f_idx)
    for i_idx in cands:
        cands[i_idx].sort()

    chosen: list[tuple[str, int]] = []
    used_f: set[int] = set()

    def bt(pos: int, cur: list[tuple[str, int]]) -> bool:
        if len(cur) == tp:
            chosen.extend(cur)
            return True
        if pos >= len(issue_order) or len(issue_order) - pos < tp - len(cur):
            return False
        i_idx = issue_order[pos]
        iid = str(issues[i_idx].get("issue_id", ""))
        for f_idx in cands[i_idx]:
            if f_idx not in used_f:
                used_f.add(f_idx)
                cur.append((iid, f_idx))
                if bt(pos + 1, cur):
                    return True
                cur.pop()
                used_f.remove(f_idx)
        return bt(pos + 1, cur)

    bt(0, [])
    return tp, chosen


def compute_reward(tp: int, fp: int, fn: int, inspection_steps: int) -> dict[str, float | int]:
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    if tp == 0 and fp == 0 and fn == 0:
        f1 = 1.0
    elif (2 * tp + fp + fn) == 0:
        f1 = 0.0
    else:
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn)
    raw = 2.0 * f1 - 1.0 - 0.002 * float(inspection_steps)
    clipped = max(-1.0, min(1.0, raw))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "return": float(clipped),
    }


def score_findings(
    findings_payload: Any,
    truth_issues: list[dict[str, Any]],
    available_evidence_ids: set[str],
    inspection_steps: int,
    default_tolerance: int | dict[str, int] = 0,
    expected_episode_id: str | None = None,
) -> dict[str, Any]:
    ok, err = validate_findings_structure(findings_payload, expected_episode_id)
    if not ok:
        return {
            "valid": False,
            "error": err,
            "tp": None,
            "fp": None,
            "fn": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "return": -1.0,
            "matched_pairs": [],
        }
    flist = findings_payload["findings"]
    edges: list[tuple[int, int]] = []
    for f_idx, f_item in enumerate(flist):
        for i_idx, t_item in enumerate(truth_issues):
            if _is_match(f_item, t_item, available_evidence_ids, default_tolerance=default_tolerance):
                edges.append((f_idx, i_idx))
    tp, pairs = max_bipartite_matching(len(flist), truth_issues, edges)
    fp = len(flist) - tp
    fn = len(truth_issues) - tp
    m = compute_reward(tp, fp, fn, inspection_steps)
    return {"valid": True, "error": None, **m, "matched_pairs": pairs}