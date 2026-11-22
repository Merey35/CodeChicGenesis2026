# Genesis Fintech Audit Environment

Genesis Fintech is a deterministic audit benchmark. An agent inspects a generated episode containing invoices, payments, a double-entry ledger, and FX rates, then submits structured findings with evidence. A judge scores the submission against a separate truth file.

The project includes the environment API, reproducible episode generator, matcher/judge adapter, baseline policies, train/dev scenarios, and benchmark runner. It is designed to make audit behavior measurable and reproducible; local results do not guarantee a particular hidden-grader score.

## Run the checks and benchmark

From the project root, install the package with its pinned test dependency, then run one cross-platform command to regenerate the 10 train and 10 dev episodes, benchmark every policy, and run tests:

```text
python -m pip install -e ".[test]"
python -m genesis_fintech.check
```

The same check runs in the supplied Docker image with `docker build -t genesis-fintech .` followed by `docker run --rm genesis-fintech`.

The runner writes:

- `artifacts/benchmark_summary.json` — one result per policy/episode, including failure status.
- `artifacts/benchmark_metadata.json` — package/schema/Python versions, seeds, and episode configs.
- `artifacts/trajectories.jsonl` — policy actions, public observations, rewards, and termination details.

The supported policy names are `random`, `reconcile`, `flag_all`, and `oracle`. `oracle` is a privileged reference ceiling and is not an agent baseline. A policy error or timeout is recorded with `status` and an explicit error; it does not stop evaluation of the remaining episodes.

## Agent and submission contract

Create a closed-mode environment with a directory, or let reset generate an episode in a temporary directory:

```python
from genesis_fintech import make_env

env = make_env()
observation, info = env.reset(seed=42, config={
    "mode": "closed",
    "team_config": {"episode_id": "practice", "errors": ["duplicate_invoice"]},
})
```

To evaluate a generated benchmark episode instead, pass `{"mode": "closed", "episode_dir": "episodes/dev/dev_01"}`. The observation describes the task, episode manifest, supported tables/actions, pagination size, amount tolerances, and remaining action count. It does not contain truth or provenance. Use `list_records`, `get_record`, `get_document`, and `calculator` to inspect data; then submit one final result:

```python
submission = {"episode_id": "dev_01", "findings": []}
observation, reward, terminated, truncated, info = env.step(
    {"name": "submit", "args": {"findings": submission}}
)
```

`episode_id` is required and must exactly match the manifest. Each finding has this shape:

```json
{
  "type": "duplicate_invoice",
  "affected_ids": ["INV-0001", "INV-0002"],
  "claimed_amount_minor": 12500,
  "currency": "KZT",
  "evidence_ids": ["DOC-INV-0001", "DOC-INV-0002"],
  "explanation": "The documents have matching business details."
}
```

The supported issue types are `duplicate_invoice`, `unallocated_or_partial_payment`, `bad_fx_date_or_rate`, and `period_misclassification`. Evidence IDs must refer to documents or ledger rows in the episode. For FX and period findings, the expected evidence set includes both the source document and the relevant receivable ledger row. FX rates are available in the `fx` table and should be checked against the invoice's applicable date and currency.

Limits are 100 environment actions, 50 records per page, 30 findings, 10 evidence IDs per finding, and 1,000 explanation characters. Invalid submissions receive `-1` in open mode and are rejected; closed mode returns no judge reward. After a successful submit the episode is final. A closed-mode timeout or 100-step exhaustion is not a successful submission.

## Scoring and evidence

The matcher checks issue type, exact currency, affected-ID set, expected amount within the configured minor-unit tolerance, and required evidence. Evidence must belong to the episode. `valid_evidence_sets`, when present in truth, specify complete acceptable alternatives; otherwise all truth evidence IDs are required.

For valid submissions, the reward uses maximum bipartite matching so one finding cannot claim multiple truth issues:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 1 when TP=FP=FN=0, otherwise 2*TP / (2*TP + FP + FN)
return    = clamp(2*F1 - 1 - 0.002*inspection_steps, -1, 1)
```

The judge adapter scores closed-mode submissions outside the environment. The environment itself never loads truth in closed mode. Open mode loads truth and returns the matcher reward on submit for development and tests.

## Practical checklist for high-quality submissions

- Read each table through its final `next_cursor`; do not rely on the first page or fixed invoice positions.
- Compare invoice and ledger records using entity, currency, amount, dates, and line items. Repeated amounts alone are not proof of a duplicate.
- Reconcile invoice totals against incoming payments in the same currency. Treat refunds separately and investigate payments without invoice references.
- For FX findings, compare the posted base-currency receivable to the invoice amount converted with the published applicable-date rate. Cite both the invoice document and the relevant ledger row.
- For period findings, compare the source document date to the ledger posting date and cite both records.
- Use exact public IDs, currency codes, integer minor-unit amounts, and complete required evidence. Do not submit speculative or mass-generated findings: false positives reduce precision and reward.
- Include the exact manifest `episode_id`; finish with one submission and avoid spending actions after the required evidence is collected.

## Dataset and reproducibility

The default dataset has 10 training and 10 development episodes covering clean ledgers, legal repeated amounts, split payments, refunds, duplicates, partial/unallocated payments, FX misstatement, period misclassification, and two interaction scenarios. Four mixed-error train episodes contain 60, 80, 100, and 150 invoices, with split payments and refunds. Both train and dev cover every supported error category. Baseline policies inspect full table pagination; the reconciliation policy checks duplicate documents, settlement totals, FX conversion, and reporting-period dates.

`generate_episode(seed, config, output_dir)` is deterministic. Use `relabel_seed` to vary neutral IDs independently while preserving the underlying business scenario. The private `provenance.json` stores generation config, pre-injection baseline data, injection events, and ID mappings. It is for the judge/developer, not for an agent observation. The `num_invoices` option accepts 6–500 and raises a clear error outside that range instead of silently truncating.

The schema version is recorded in each manifest; the runner regenerates default episodes when their schema does not match the installed generator. Keep `benchmark_metadata.json` with results to make the data and software version auditable. The clean source archive excludes generated episode folders because they contain private truth/provenance; the unified check command regenerates them locally.

## Policy execution and security boundary

Non-oracle policies execute in a separate worker process and interact with the environment through an observation/action pipe. The episode path and truth are not sent to those workers. Each worker has a hard 60-second runner timeout; timed-out workers are terminated and the benchmark continues. The oracle reference receives truth by design and must not be interpreted as a contestant score.

This process boundary is not an operating-system sandbox: a process running as the same user may still be able to read files permitted to that user. For untrusted third-party agents, run the benchmark in a container or separate account with filesystem/network restrictions, and keep private truth/provenance outside the agent's readable paths.
