from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import time
import traceback
from pathlib import Path
from typing import Any

from .baselines import (
    run_flag_all_policy,
    run_oracle_policy,
    run_random_policy,
    run_simple_reconciliation_policy,
)
from .env import make_env
from .generator import SCHEMA_VERSION, generate_episode
from .judge_adapter import score_episode
from . import __version__

POLICY_TIMEOUT_SECONDS = 60.0


def build_default_episodes(base_dir: Path) -> None:
    train_dir = base_dir / "episodes" / "train"
    dev_dir = base_dir / "episodes" / "dev"
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    train_specs = [
        ("train_01", 101, {"errors": [], "num_invoices": 12, "refund_count": 0}),
        ("train_02", 102, {"errors": [], "num_invoices": 14, "split_payment_count": 2, "refund_count": 0}),
        ("train_03", 103, {"errors": [], "num_invoices": 12, "refund_count": 2}),
        ("train_04", 104, {"errors": [], "num_invoices": 12, "legal_repeat_amounts": True, "refund_count": 0}),
        ("train_05", 105, {"errors": [], "num_invoices": 12, "base_currency": "USD", "period_boundary_invoice": True, "refund_count": 0}),
        ("train_06", 106, {"errors": ["duplicate_invoice"], "num_invoices": 13, "refund_count": 0}),
        ("train_07", 107, {"errors": ["duplicate_invoice", "bad_fx_date_or_rate"], "num_invoices": 58, "split_payment_count": 8, "refund_count": 5}),
        ("train_08", 108, {"errors": ["duplicate_invoice", "period_misclassification"], "num_invoices": 78, "split_payment_count": 10, "refund_count": 6}),
        ("train_09", 109, {"errors": ["unallocated_or_partial_payment", "bad_fx_date_or_rate", "period_misclassification"], "payment_issue_mode": "unallocated", "payment_discrepancy_minor": 1800, "num_invoices": 98, "split_payment_count": 12, "refund_count": 8}),
        ("train_10", 110, {"errors": ["unallocated_or_partial_payment", "duplicate_invoice", "bad_fx_date_or_rate", "period_misclassification"], "payment_issue_mode": "partial", "payment_discrepancy_minor": 2300, "num_invoices": 147, "split_payment_count": 18, "refund_count": 12}),
    ]
    for eid, s, cfg in train_specs:
        generate_episode(seed=s, config={**cfg, "episode_id": eid}, output_dir=train_dir / eid)

    dev_specs = [
        ("dev_01", 201, {"errors": ["unallocated_or_partial_payment"], "payment_issue_mode": "partial", "payment_discrepancy_minor": 3100, "refund_count": 0}),
        ("dev_02", 202, {"errors": ["bad_fx_date_or_rate"], "refund_count": 0}),
        ("dev_03", 203, {"errors": ["bad_fx_date_or_rate"], "base_currency": "USD", "refund_count": 0}),
        ("dev_04", 204, {"errors": ["period_misclassification"], "period_misclassification_days": 3, "period_misclassification_minor": 12000, "refund_count": 0}),
        ("dev_05", 205, {"errors": ["period_misclassification"], "period_misclassification_days": 8, "period_misclassification_minor": 22500, "base_currency": "USD", "refund_count": 0}),
        ("dev_06", 206, {"interactions": ["partial_fx_payment_near_period_end"], "scenario_params": {"partial_fx_payment_near_period_end": {"days_before_period_end": 1, "shortfall_minor": 1100}}, "refund_count": 0}),
        ("dev_07", 207, {"interactions": ["partial_fx_payment_near_period_end"], "scenario_params": {"partial_fx_payment_near_period_end": {"days_before_period_end": 2, "shortfall_minor": 2200}}, "num_invoices": 16, "refund_count": 0}),
        ("dev_08", 208, {"interactions": ["partial_fx_payment_near_period_end"], "scenario_params": {"partial_fx_payment_near_period_end": {"days_before_period_end": 3, "shortfall_minor": 3300}}, "base_currency": "USD", "refund_count": 0}),
        ("dev_09", 209, {"interactions": ["duplicate_invoice_with_partial_payment"], "scenario_params": {"duplicate_invoice_with_partial_payment": {"shortfall_minor": 1400}}, "refund_count": 0}),
        ("dev_10", 210, {"interactions": ["duplicate_invoice_with_partial_payment"], "scenario_params": {"duplicate_invoice_with_partial_payment": {"shortfall_minor": 2800}}, "base_currency": "USD", "refund_count": 0}),
    ]
    for eid, s, cfg in dev_specs:
        generate_episode(seed=s, config={**cfg, "episode_id": eid}, output_dir=dev_dir / eid)


class LoggedEnvWrapper:
    def __init__(self, inner: Any, traj_records: list[dict[str, Any]], episode_id: str) -> None:
        self._inner = inner
        self._traj = traj_records
        self._eid = episode_id
        self._step = 0

    def reset(self, seed: int = 0, config: dict[str, Any] | None = None):
        self._step = 0
        obs, info = self._inner.reset(seed=seed, config=config)
        self._eid = obs.get("manifest", {}).get("episode_id", self._eid)
        return obs, info

    def step(self, action: dict[str, Any]):
        self._step += 1
        obs, reward, terminated, truncated, info = self._inner.step(action)
        self._traj.append({
            "episode_id": self._eid,
            "step": self._step,
            "action": action,
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })
        return obs, reward, terminated, truncated, info


class _RemotePolicyEnv:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def step(self, action: dict[str, Any]):
        self._connection.send({"kind": "action", "action": action})
        response = self._connection.recv()
        if response.get("kind") != "step_result":
            raise RuntimeError(response.get("error", "Environment request failed"))
        return response["value"]


def _policy_process_main(
    connection: Any,
    policy_name: str,
    init_obs: dict[str, Any],
    oracle_issues: list[dict[str, Any]] | None,
) -> None:
    try:
        remote_env = _RemotePolicyEnv(connection)
        if policy_name == "random":
            submission = run_random_policy(remote_env, init_obs)
        elif policy_name == "reconcile":
            submission = run_simple_reconciliation_policy(remote_env, init_obs)
        elif policy_name == "flag_all":
            submission = run_flag_all_policy(remote_env, init_obs)
        elif policy_name == "oracle" and oracle_issues is not None:
            submission = run_oracle_policy(remote_env, init_obs, oracle_issues)
        else:
            raise ValueError(f"Unknown policy or missing oracle issues: {policy_name}")
        connection.send({"kind": "complete", "submission": submission})
    except BaseException as exc:
        try:
            connection.send({
                "kind": "failure",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _execute_policy_isolated(
    policy_name: str,
    init_obs: dict[str, Any],
    logged_env: LoggedEnvWrapper,
    oracle_issues: list[dict[str, Any]] | None = None,
    timeout_seconds: float = POLICY_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, str, str | None]:
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe()
    process = context.Process(
        target=_policy_process_main,
        args=(child_connection, policy_name, init_obs, oracle_issues),
        name=f"audit-policy-{policy_name}",
    )
    process.start()
    child_connection.close()
    deadline = time.monotonic() + timeout_seconds
    status = "error"
    submission = None
    error_message = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "timeout"
                error_message = f"Policy exceeded {timeout_seconds:g}s timeout"
                break
            if not parent_connection.poll(min(0.05, remaining)):
                if not process.is_alive():
                    error_message = f"Policy process exited with code {process.exitcode} without a result"
                    break
                continue
            message = parent_connection.recv()
            if message.get("kind") == "action":
                try:
                    value = logged_env.step(message.get("action"))
                    parent_connection.send({"kind": "step_result", "value": value})
                except Exception as exc:
                    error_message = f"Environment action failed: {type(exc).__name__}: {exc}"
                    break
            elif message.get("kind") == "complete":
                submission = message.get("submission")
                status = "ok"
                break
            elif message.get("kind") == "failure":
                error_message = message.get("traceback") or message.get("error")
                break
            else:
                error_message = "Policy process sent an invalid response"
                break
    except (EOFError, OSError) as exc:
        error_message = f"Policy communication failed: {type(exc).__name__}: {exc}"
    finally:
        parent_connection.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()
        process.close()
    return submission, status, error_message


def run_benchmark(repo_root: Path, policy_name: str = "all", limit_episodes: int = 20) -> list[dict[str, Any]]:
    train_dir = repo_root / "episodes" / "train"
    dev_dir = repo_root / "episodes" / "dev"
    episode_paths = [
        train_dir / f"train_{index:02d}" for index in range(1, 11)
    ] + [
        dev_dir / f"dev_{index:02d}" for index in range(1, 11)
    ]
    regenerate = False
    for episode_path in episode_paths:
        manifest_path = episode_path / "manifest.json"
        if not episode_path.is_dir() or not manifest_path.is_file():
            regenerate = True
            break
        if json.loads(manifest_path.read_text(encoding="utf-8")).get("schema_version") != SCHEMA_VERSION:
            regenerate = True
            break
    if regenerate:
        build_default_episodes(repo_root)
    ep_dirs = episode_paths[:max(0, min(20, limit_episodes))]
    policies = ["random", "reconcile", "flag_all", "oracle"] if policy_name == "all" else [policy_name]
    invalid_policies = set(policies) - {"random", "reconcile", "flag_all", "oracle"}
    if invalid_policies:
        raise ValueError(f"Unknown policies: {sorted(invalid_policies)}")
    traj_path = repo_root / "artifacts" / "trajectories.jsonl"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj_records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    base_env = make_env()

    for pol in policies:
        for ep_path in ep_dirs:
            t0 = time.perf_counter()
            logged = LoggedEnvWrapper(base_env, traj_records, ep_path.name)
            init_obs, _ = logged.reset(seed=1, config={"mode": "closed", "episode_dir": str(ep_path)})
            oracle_issues = None
            if pol == "oracle":
                t_obj = json.loads((ep_path / "truth.json").read_text(encoding="utf-8"))
                oracle_issues = t_obj.get("issues", [])
            sub, status, policy_error = _execute_policy_isolated(
                pol,
                init_obs,
                logged,
                oracle_issues=oracle_issues,
            )
            insp = base_env.get_inspection_steps()
            if status == "ok" and sub is not None:
                sc = score_episode(str(ep_path), sub, insp)
            else:
                sc = {
                    "tp": None,
                    "fp": None,
                    "fn": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "return": -1.0,
                }
            summary.append({
                "policy": pol,
                "episode_id": ep_path.name,
                "inspection_steps": insp,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "status": status,
                **({"error": policy_error} if policy_error else {}),
                **sc,
            })

    with traj_path.open("w", encoding="utf-8") as f:
        for rec in traj_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (repo_root / "artifacts" / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    episode_metadata = []
    for episode_path in ep_dirs:
        provenance = json.loads(
            (episode_path / "provenance.json").read_text(encoding="utf-8")
        )
        episode_metadata.append({
            "episode_id": episode_path.name,
            "seed": provenance["seed"],
            "config": provenance["config"],
        })
    (repo_root / "artifacts" / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "package_version": __version__,
                "schema_version": SCHEMA_VERSION,
                "python_version": platform.python_version(),
                "policies": policies,
                "limit_episodes": limit_episodes,
                "policy_timeout_seconds": POLICY_TIMEOUT_SECONDS,
                "policy_execution": "separate process with environment RPC",
                "episodes": episode_metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-episodes", action="store_true")
    parser.add_argument("--policy", default="all", choices=["all", "random", "reconcile", "flag_all", "oracle"])
    parser.add_argument("--limit-episodes", type=int, default=20)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if args.generate_episodes:
        build_default_episodes(repo_root)
    rows = run_benchmark(repo_root, policy_name=args.policy, limit_episodes=args.limit_episodes)
    print(json.dumps({"episodes_evaluated": len(rows), "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()