# Design notes

## Components and data flow

`generator.py` creates a deterministic synthetic accounting episode from a seed and config. It first builds ordinary invoices, payments, FX rates, and a balanced double-entry ledger. It records that clean state in private provenance, then injects configured anomalies and records matching truth issues. A distinct placement RNG chooses different source invoices for each injection; the optional `relabel_seed` maps the generated IDs to neutral, shuffled public IDs after all references have been created. The final ledger is chronological unless `shuffle_records` is explicitly enabled.

`env.py` exposes an action/observation interface. `reset` can load a supplied episode directory or generate one from `seed` and `team_config` in a temporary directory. In closed mode the environment deliberately does not load `truth.json`; in open mode it does, to provide immediate development rewards. The observation exposes only public tables, policy metadata, and the episode manifest. Table reads are paginated; submissions are final and checked against the exact episode ID.

`matcher.py` validates the submission schema and matches findings to truth using maximum bipartite matching. Each finding can match at most one issue. Matching requires the correct type, affected IDs, currency, amount within configured tolerance, and evidence IDs that are available and satisfy one complete `valid_evidence_sets` alternative (or all required evidence IDs when alternatives are absent). Scoring uses F1 and subtracts a small inspection-step cost; malformed submissions score -1.

`judge_adapter.py` is the external scoring boundary for closed episodes. `runner.py` evaluates baseline policies, saves benchmark metadata and trajectories, and records each policy failure rather than aborting the rest of the matrix. Non-oracle policies run in a separate worker process and communicate with the parent environment through an RPC pipe. They receive public observations/actions, not the environment object, episode path, truth, or provenance. The oracle baseline is an explicitly privileged reference and is not representative of a contestant's access.

## Data integrity and verification

The generator keeps separate IDs and references consistent across invoices, payments, documents, ledger rows, entity IDs, truth, and evidence sets. The clean baseline is captured before injections so tests can compare balanced source data with the final episode. Synthetic FX errors use an actual alternative rate/date present in the episode's FX table, and period errors are visible by comparing the invoice date with the receivable posting. These injected invoices also have ordinary line descriptions and settlement payments so their shape is less distinctive.

`python -m genesis_fintech.check` regenerates the 10 train and 10 dev episodes, evaluates four policies over all 20 episodes, stores the report under `artifacts/`, and runs pytest. Tests cover environment contracts, malformed input, evidence requirements, paired ID relabeling, mixed-error generation across 200 seeds, ledger balance, large episodes, policy timeouts/failures, and benchmark metadata.

The reconciliation policy is a transparent baseline, not an optimal auditor. It catches exact duplicate patterns, unallocated or incomplete settlements, FX conversion differences using the listed issue-date rate, and invoices posted in a period inconsistent with their source date. It can miss semantically equivalent duplicates with changed details, ambiguous payment allocations, unsupported currencies/rates, issues requiring business context not represented in tables, and anomalies that do not alter the selected numeric/date checks. `flag_all` demonstrates the precision cost of speculative alerts; `random` is a weak sanity baseline; `oracle` provides a privileged upper-bound check.

### Measured default benchmark (schema 1.3.0)

The following is the observed result over the 20 fixed train/dev episodes (80 evaluations total). It is a reproducible local measurement, not a hidden-grader guarantee.

| Policy | Mean return | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| random | -0.5045 | 0 | 0 | 24 |
| reconcile | 0.9818 | 24 | 0 | 0 |
| flag_all | -1.0000 | 0 | 480 | 24 |
| oracle (privileged) | 0.9955 | 24 | 0 | 0 |

These results show that exact cross-table reconciliation performs well on this intentionally synthetic scenario set, while mass flagging is penalized. They do not measure performance on real accounting data.

## Security and known limitations

Closed-mode `AuditEnv` does not load truth, and the policy RPC excludes paths and truth objects from non-oracle worker arguments. This is process/API separation, not an OS sandbox. A worker running as the same user may still read any file that account can access; a deployment with untrusted agents must use a container or separate restricted account, hide truth/provenance from its filesystem, and restrict network access. The benchmark's generated files include private truth/provenance for the judge, so the clean source archive omits generated episode folders and regenerates them at check time.

The synthetic data currently uses a small currency set (KZT, USD, EUR, GBP), with generated FX pairs limited to the configured base currency and one foreign currency. The benchmark models one fixed monthly reporting period, not multi-period accounting, real banking rails, tax rules, or all accounting conventions. Results show correctness on these explicit local tests and scenarios only; no hidden evaluation score is promised.
