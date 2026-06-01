# agno/camel/flower Memory Subset Eval

This workflow isolates the successful generated-patch issues for `agno`, `camel`,
and `flower`, uses the non-eval issues as L1/L2/L3 memory, then evaluates the
held-out issues across memory ablations.

The runner defaults to `MiniMax-M2.5`. It reuses cached log analysis from:

```text
baselines/results/MiniMax-M2.5/log_details.json
```

It also reuses existing memory records from:

```text
baselines/results/trs/
```

Only records belonging to the selected memory seed issues are copied into this
subset memory bank. Missing seed issues are analyzed and added.

## Run

```bash
baselines/.venv/bin/python eval_scripts/memory_subset/run_memory_subset_ablation.py \
  --eval-per-repo 5
```

By default this uses `agno`, `camel`, and `flower`. For each repo, it holds out
up to `--eval-per-repo` success issues for eval and uses the remaining success
issues as memory seed.

Add repos on top of the defaults:

```bash
baselines/.venv/bin/python eval_scripts/memory_subset/run_memory_subset_ablation.py \
  --extra-repos axolotl litellm \
  --eval-per-repo 5
```

Replace the repo list entirely:

```bash
baselines/.venv/bin/python eval_scripts/memory_subset/run_memory_subset_ablation.py \
  --repos agno camel flower axolotl \
  --eval-per-repo 5
```

For a smaller memory bank, override the memory seed count:

```bash
baselines/.venv/bin/python eval_scripts/memory_subset/run_memory_subset_ablation.py \
  --memory-per-repo 2 \
  --eval-per-repo 5
```

Override cache locations if needed:

```bash
baselines/.venv/bin/python eval_scripts/memory_subset/run_memory_subset_ablation.py \
  --cached-log-details baselines/results/MiniMax-M2.5/log_details.json \
  --existing-memory-bank baselines/results/trs \
  --eval-per-repo 5
```

## Outputs

Default root:

```text
baselines/results/memory_subset_agno_camel_flower/
```

Important files:

```text
split/all_success_issues.json
split/memory_seed_issues.json
split/eval_issues.json
split/summary.json

memory_bank/failure_memory.json
memory_bank/repo_memory.json
memory_bank/cross_memory.json
memory_bank/memory_subset_build_manifest.json

trajectories/L1/trajectories.json
trajectories/L1/summary.json
trajectories/L1_L2/trajectories.json
trajectories/L1_L2/summary.json
trajectories/L1_L2_L3/trajectories.json
trajectories/L1_L2_L3/summary.json
```

Each trajectory record includes the input issue, log analysis, memory retrievals,
fault localization output, generated patch, and step trace where available.
