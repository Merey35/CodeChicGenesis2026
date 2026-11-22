from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .runner import build_default_episodes, run_benchmark


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    build_default_episodes(project_root)
    rows = run_benchmark(project_root, policy_name="all", limit_episodes=20)
    policy_failures = [
        f"{row['policy']}/{row['episode_id']}: {row.get('error', row['status'])}"
        for row in rows
        if row.get("status") != "ok"
    ]
    print(f"Benchmark completed: {len(rows)} policy/episode evaluations")
    if policy_failures:
        print("Policy failures:")
        print("\n".join(policy_failures))

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=project_root,
        check=False,
    )
    if tests.returncode:
        return tests.returncode
    return 1 if policy_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
