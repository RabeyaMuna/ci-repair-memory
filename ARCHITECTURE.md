# CI-REPAIR-BENCH — Full System Architecture

## Overview

The system is a **hierarchical-memory-augmented CI repair pipeline**. Given a failing commit, it:
1. Analyzes the CI failure log
2. Fetches relevant past-failure context from a 3-level memory bank
3. Localizes the fault to specific files
4. Generates a patch

The **same `MemoryPlugin`** is designed to be plugged into any agent or LLM, allowing you to compare different agents on identical data with the same memory system.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CI-REPAIR-BENCH Pipeline                             │
│                                                                             │
│  Dataset (HuggingFace)                                                      │
│    │  task_id, sha_fail, repo, logs, workflow, workflow_path                │
│    ▼                                                                        │
│ ┌──────────────────────────────────────────────────────────────────────┐    │
│ │  STEP 1 — CI Log Analysis                                            │    │
│ │                                                                      │    │
│ │  CILogAnalyzerLLM  (or CILogAnalyzerBM25)                           │    │
│ │  ┌────────────────────┐   ┌───────────────────────────────────────┐ │    │
│ │  │ ci_log_analysis()  │   │  generate_log_summary()               │ │    │
│ │  │ • chunk raw logs   │ → │  • per-step structured summary        │ │    │
│ │  │ • per-chunk LLM    │   │  • relevant_files WITH:               │ │    │
│ │  │   extract:         │   │    - issue_type (per file)            │ │    │
│ │  │   - relevant_files │   │    - failed_cmd  (per file)           │ │    │
│ │  │     issue_type ✓   │   │    - failed_tool (per file)           │ │    │
│ │  │     failed_cmd  ✓  │   │    - reason, line_number              │ │    │
│ │  │     failed_tool ✓  │   │  • error_types (category/subcategory) │ │    │
│ │  │   - relevant_fails │   │  • error_context (root cause text)    │ │    │
│ │  └────────────────────┘   └───────────────────────────────────────┘ │    │
│ │                                     │                               │    │
│ │                                     ▼                               │    │
│ │                           full_content_summary()                    │    │
│ │                           • aggregates all steps                    │    │
│ │                           • deduplicates files                      │    │
│ │                           • carries forward issue_type/cmd/tool     │    │
│ │                           • produces: log_analysis_result           │    │
│ │                             ┌─ error_context       (list[str])      │    │
│ │                             ├─ relevant_files      (list[dict])     │    │
│ │                             │   file, line_number,                  │    │
│ │                             │   issue_type, failed_cmd,             │    │
│ │                             │   failed_tool, reason                 │    │
│ │                             ├─ error_types         (list[dict])     │    │
│ │                             │   category, subcategory, evidence     │    │
│ │                             └─ failed_job          (list[dict])     │    │
│ │                                 job, step, command                  │    │
│ └──────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│ ┌──────────────────────────────────────────────────────────────────────┐    │
│ │  STEP 2 — Memory Plugin: Build Query + Retrieve                      │    │
│ │                                                                      │    │
│ │  memory_plugin.build_query(log_analysis_result, changed_files_info) │    │
│ │  ┌─────────────────────────────────────────────────────────────┐    │    │
│ │  │ query = {                                                    │    │    │
│ │  │   error_type, failure_pattern, failure_reason               │    │    │
│ │  │   failed_cmd, failed_tool           ← issue-level           │    │    │
│ │  │   relevant_files                    ← paths list            │    │    │
│ │  │   relevant_files_details            ← full per-file dicts   │    │    │
│ │  │     { file, issue_type,                                      │    │    │
│ │  │       failed_cmd, failed_tool,                               │    │    │
│ │  │       reason, line_number }                                  │    │    │
│ │  │   changed_files, workflow_path                               │    │    │
│ │  │ }                                                            │    │    │
│ │  └─────────────────────────────────────────────────────────────┘    │    │
│ │                                                                      │    │
│ │  memory_plugin.retrieve(query)                                       │    │
│ │  ┌───────────────────────────────────────────────────────────────┐  │    │
│ │  │  L1 _retrieve_l1()  — per-file failure records                │  │    │
│ │  │   • file_score  (exact path match)              weight: 0.45  │  │    │
│ │  │   • semantic_score (error+pattern+issue+reason) weight: 0.50  │  │    │
│ │  │   • tool_cmd_score  max(tool_jaccard, cmd_jaccard) w: 0.05    │  │    │
│ │  │   → per-file query doc enriched from relevant_files_details   │  │    │
│ │  │                                                               │  │    │
│ │  │  L2 _retrieve_l2()  — repo-level recurring patterns           │  │    │
│ │  │   • semantic_score (holistic doc embedding)     weight: 0.90  │  │    │
│ │  │   • tool_cmd_score                              weight: 0.10  │  │    │
│ │  │                                                               │  │    │
│ │  │  L3 _retrieve_l3()  — cross-repo generalizations             │  │    │
│ │  │   • semantic_score                              weight: 0.90  │  │    │
│ │  │   • tool_cmd_score                              weight: 0.10  │  │    │
│ │  │                                                               │  │    │
│ │  │  All levels: fix_strategies/fix_direction NOT in query doc    │  │    │
│ │  │  (unknown at query time) but ARE in returned records          │  │    │
│ │  │                                                               │  │    │
│ │  │  Returns: {l1_matches, l2_matches, l3_matches,               │  │    │
│ │  │            candidate_files, high_level_hints,                │  │    │
│ │  │            weighted_similarity, level_scores}                │  │    │
│ │  └───────────────────────────────────────────────────────────────┘  │    │
│ └──────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│ ┌──────────────────────────────────────────────────────────────────────┐    │
│ │  STEP 3 — Fault Localization                                         │    │
│ │                                                                      │    │
│ │  FaultLocalization(log_analysis_result, memory_plugin, memory_context)│   │
│ │                                                                      │    │
│ │  [3a] select_suspicious_files()                                      │    │
│ │       • LLM scans changed files + log analysis                      │    │
│ │       • returns candidate file list                                  │    │
│ │                                                                      │    │
│ │  [3b] _apply_memory_to_suspicious_files()                            │    │
│ │       • memory_plugin.augment_suspicious_files()                    │    │
│ │         adds L1/L2 candidate_files not already in list              │    │
│ │       • memory_plugin.rank_files()                                  │    │
│ │         re-ranks by L1/L2 similarity scores                         │    │
│ │                                                                      │    │
│ │  [3c] _final_fault_localization()  ← per-file loop                  │    │
│ │       For each suspicious file:                                      │    │
│ │                                                                      │    │
│ │       memory_plugin.retrieve_for_file(file_path, file_context)      │    │
│ │       ┌──────────────────────────────────────────────────────────┐  │    │
│ │       │ file_query enriched with per-file details:               │  │    │
│ │       │  • failed_tool/cmd overridden from relevant_files_detail │  │    │
│ │       │  • issue_type prepended to failure_reason                │  │    │
│ │       │  • file snippet appended to failure_reason               │  │    │
│ │       │ → L1/L2/L3 re-run with this file-specific query         │  │    │
│ │       │ → ranked candidates returned                             │  │    │
│ │       └──────────────────────────────────────────────────────────┘  │    │
│ │                                                                      │    │
│ │       memory_plugin.analyze_relevance_for_file() [LLM call]         │    │
│ │       ┌──────────────────────────────────────────────────────────┐  │    │
│ │       │ LLM sees: ranked L1/L2/L3 candidates with scores        │  │    │
│ │       │ LLM decides: which candidates are relevant for THIS file │  │    │
│ │       │ LLM returns: selected_items with:                        │  │    │
│ │       │   - justification, localization_hint, fix_direction      │  │    │
│ │       │   - dependent_files, additional_localization_files       │  │    │
│ │       └──────────────────────────────────────────────────────────┘  │    │
│ │                                                                      │    │
│ │       memory_plugin.format_for_file_prompt() → memory context text  │    │
│ │       → injected into the final FL prompt for this file             │    │
│ │                                                                      │    │
│ │  Produces: fault_localization_data                                   │    │
│ │    [ { file_path, faults: [{issue_type, reason, ...}] } ]           │    │
│ └──────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│ ┌──────────────────────────────────────────────────────────────────────┐    │
│ │  STEP 4 — Patch Generation                                           │    │
│ │  PatchGeneration(fault_localizer, log_analysis_result, ...)          │    │
│ │  → generates unified diff patch                                      │    │
│ └──────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│ ┌──────────────────────────────────────────────────────────────────────┐    │
│ │  STEP 5 — Memory Plugin: Save to L1 / L2 / L3                       │    │
│ │                                                                      │    │
│ │  memory_plugin.save_memory_entry(                                    │    │
│ │    log_analysis_result, fault_localizer, patch_generator)            │    │
│ │                                                                      │    │
│ │  L1 — failure_memory.json (per-file failure records)                 │    │
│ │  ┌────────────────────────────────────────────────────────────────┐  │    │
│ │  │ One row per file × issue:                                      │  │    │
│ │  │  sha_fail, repo, workflow_path, file                           │  │    │
│ │  │  error_type       ← from log analysis error_types             │  │    │
│ │  │  issue_type       ← descriptive phrase ("Dependency on env")  │  │    │
│ │  │  failure_pattern  ← keyword ("dependency_or_env")             │  │    │
│ │  │  failure_reason   ← from FL faults + log analyzer per-file   │  │    │
│ │  │  fix_direction    ← lines added in this file's diff chunk     │  │    │
│ │  │  failed_tool      ← per-file from log analyzer (or issue-lvl) │  │    │
│ │  │  failed_cmd       ← per-file from log analyzer (or issue-lvl) │  │    │
│ │  │  dependent_files  ← other files co-patched in same fix        │  │    │
│ │  └────────────────────────────────────────────────────────────────┘  │    │
│ │                                                                      │    │
│ │  L2 — repo_memory.json (repo-level patterns, one row per issue)      │    │
│ │  ┌────────────────────────────────────────────────────────────────┐  │    │
│ │  │  sha_fail, repo, error_type, issue_type, failure_pattern       │  │    │
│ │  │  overall_failure_reason, fix_approach                          │  │    │
│ │  │  files: [{file, issue_type, failure_reason, fix_direction,     │  │    │
│ │  │           dependent_files}]                                    │  │    │
│ │  │  failed_tool, failed_cmd (issue-level)                         │  │    │
│ │  └────────────────────────────────────────────────────────────────┘  │    │
│ │                                                                      │    │
│ │  L3 — cross_memory.json (cross-repo principles, merged by type)      │    │
│ │  ┌────────────────────────────────────────────────────────────────┐  │    │
│ │  │  error_type, issue_type, failure_pattern                       │  │    │
│ │  │  principle (abstract "error_type — issue_type: reason")        │  │    │
│ │  │  fix_strategies, failure_patterns, failure_reasons             │  │    │
│ │  │  example_files, repos, evidence_issue_ids                      │  │    │
│ │  │  [accumulates across issues of the same type]                  │  │    │
│ │  └────────────────────────────────────────────────────────────────┘  │    │
│ └──────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File → Role Reference

| File | Role |
|---|---|
| `main.py` | Orchestrator — runs the full pipeline for every dataset item |
| `ci_repair/ci_log_analyzer_llm.py` | Log analysis: chunk → step summary → final aggregation |
| `ci_repair/ci_log_analyzer_bm25.py` | Alternative log analysis using BM25 keyword extraction |
| `ci_repair/fault_localization.py` | File selection → per-file fault analysis → memory injection |
| `ci_repair/patch_generation.py` | Generates the actual diff patch given fault locations |
| `utilities/memory_plugin.py` | **Shared memory system** — build query, retrieve L1/L2/L3, save L1/L2/L3 |
| `utilities/llm_provider.py` | LLM factory + token-tracked wrappers |
| `utilities/token_tracker.py` | Cost/token tracking per agent step |
| `utilities/fl_evaluator.py` | Evaluates FL accuracy against ground-truth diffs |

---

## Memory Bank Files (shared across agents)

```
baselines/results/
├── failure_memory.json     ← L1: per-file failure records
├── repo_memory.json        ← L2: repo-level patterns (one per issue)
└── cross_memory.json       ← L3: cross-repo generalized principles
```

These are in `project_result_dir` (from `config.yaml`), which is **agent-independent**.
Each agent's *output* goes to a sub-folder; the memory banks are shared.

---

## Multi-Agent Design: Same Memory, Different Agents

The goal: plug the **same** `MemoryPlugin` into different LLM agents and compare their performance using identical memory context.

### How It Works

```
config.yaml
  project_result_dir: .../baselines/results    ← SHARED memory location

Agent A (GPT-4)       Agent B (Claude)      Agent C (DeepSeek)
  result_dir:           result_dir:           result_dir:
  results/gpt4_llm      results/claude_llm    results/deepseek_llm
  │                     │                     │
  └──────── all read from ──────────────────► failure_memory.json
  └──────── and write to  ──────────────────► repo_memory.json
                                              cross_memory.json
```

Each agent run:
- Has its own `log_details.json`, `fault_localization.json`, `generated_patches.json`, `fl_evaluation.json`
- Reads from and writes to the **same** shared memory banks
- Can be compared fairly: same dataset, same memory, different LLM reasoning

### Three Comparison Modes

| Mode | How to configure | Use case |
|---|---|---|
| **Shared memory** (default) | One `project_result_dir` for all agents | Agents learn from each other's fixes |
| **Isolated memory** | Different `project_result_dir` per agent in config | Each agent learns only from its own history |
| **Read-only memory** | `memory_writeback_enabled: false` per agent | Pre-built memory bank; agents only consume it |

### Running Multiple Agents

```python
# In main.py — run the same dataset through different agents
for model_key, log_type in [
    ("gpt-4o",        "llm"),
    ("claude-3-5-sonnet", "llm"),
    ("deepseek-chat", "llm"),
    ("gpt-4o",        "bm25"),   # same LLM, different log analyzer
]:
    results = process_entire_dataset(
        dataset, config, get_llm(model_key),
        model_key=model_key,
        log_analyzer_type=log_type,
        tracker=TokenTracker(model_name=model_key, log_analyzer_type=log_type),
    )
```

`process_entire_dataset` creates `result_dir = f"{model_dir_key}_{log_type}{run_suffix}"` automatically, so each agent gets its own output folder while sharing memory.

---

## Data Flow: What Each Stage Produces and Consumes

### log_analysis_result (produced by Step 1, consumed by Steps 2/3/5)

```json
{
  "sha_fail": "abc123",
  "error_context": ["In step 'Run tests', pytest failed because..."],
  "relevant_files": [
    {
      "file": "tests/test_cli.py",
      "line_number": 42,
      "issue_type": "Test Failure",
      "failed_cmd": "pytest tests/",
      "failed_tool": "pytest",
      "reason": "TypeError raised during test collection at line 42"
    }
  ],
  "error_types": [
    {
      "category": "Test Failure",
      "subcategory": "TypeError during pytest collection",
      "evidence": "TypeError: argument of type 'bool' is not iterable"
    }
  ],
  "failed_job": [
    { "job": "test", "step": "Run tests", "command": "pytest tests/" }
  ]
}
```

### memory_query (produced by Step 2a, consumed by retrieve/retrieve_for_file)

```json
{
  "repo": "flower/flwr",
  "error_type": "TypeError",
  "failure_pattern": "test_collection",
  "failure_reason": "TypeError during pytest collection...",
  "failed_tool": ["pytest"],
  "failed_cmd": ["pytest tests/"],
  "relevant_files": ["tests/test_cli.py", "src/app.py"],
  "relevant_files_details": [
    {
      "file": "tests/test_cli.py",
      "issue_type": "Test Failure",
      "failed_cmd": ["pytest tests/"],
      "failed_tool": ["pytest"],
      "reason": "TypeError raised here"
    }
  ]
}
```

### memory_context (produced by retrieve(), consumed by FaultLocalization)

```json
{
  "l1_matches": [
    {
      "memory_level": "L1",
      "file": "tests/test_cli.py",
      "similarity_score": 0.78,
      "matched_on": { "file_score": 1.0, "semantic_score": 0.71, "tool_score": 1.0, "cmd_score": 0.0 },
      "error_type": "TypeError",
      "issue_type": "Test Failure",
      "failure_pattern": "test_collection",
      "failure_reason": "Typer/Click bool flag issue",
      "fix_direction": "change flag default from False to ...",
      "failed_tool": ["pytest"],
      "failed_cmd": ["pytest tests/"],
      "dependent_files": [{"file": "src/app.py", "reason": "Co-modified"}]
    }
  ],
  "l2_matches": [...],
  "l3_matches": [...],
  "candidate_files": ["tests/test_cli.py", "src/app.py"],
  "high_level_hints": ["CLI app processes bool flag incorrectly via Typer/Click"],
  "weighted_similarity": 0.72
}
```

---

## Memory Scoring Summary

### What gets cosine similarity (semantic)

| Field | Storage | Use in query |
|---|---|---|
| `error_type` | All levels | ✅ in holistic doc |
| `failure_pattern` | All levels | ✅ in holistic doc |
| `issue_type` | All levels | ✅ derived at query time |
| `failure_reason` | All levels | ✅ in holistic doc |
| `fix_direction` / `fix_strategies` | All levels (saved) | ❌ NOT in query doc (unknown) |

### What gets text/structural matching

| Field | Method | Weight |
|---|---|---|
| `file` path | Exact normalized path (binary 0/1) | L1: 0.45 |
| `repo` | Hard filter (exact match) | Pre-filter |
| `failed_tool` | Jaccard set overlap | max(tool, cmd) × 0.05–0.10 |
| `failed_cmd` | Jaccard set overlap | max(tool, cmd) × 0.05–0.10 |

`tool_cmd_score = max(jaccard_tools, jaccard_cmds)` — scores if EITHER matches, not both required.

---

## Config Keys for Memory

```yaml
# config.yaml

memory_enabled: true                    # master switch
memory_top_k: 3                         # top K from each level
memory_similarity_threshold: 0.45      # fallback threshold (non-ablation)
memory_ablation_levels: "L1+L2+L3"     # "L1" | "L1+L2" | "L1+L2+L3"
memory_writeback_enabled: true          # whether to save after each fix
project_result_dir: .../results         # shared memory bank location

# Ablation thresholds (auto-selected):
#   L1       → 0.55
#   L1+L2    → 0.37
#   L1+L2+L3 → 0.33
```

---

## What's Still Needed / Open Questions

| Item | Status | Notes |
|---|---|---|
| Per-file `issue_type`/`failed_cmd`/`failed_tool` in log analysis output | ✅ Done | All 3 prompts updated |
| Holistic single-embedding retrieval | ✅ Done | Replaces separate error/pattern scores |
| `max(tool, cmd)` scoring | ✅ Done | Score if EITHER matches |
| `fix_direction` excluded from query doc | ✅ Done | Only saved, not searched by |
| `retrieve_for_file` enriched with per-file details | ✅ Done | File-specific embedding |
| Multi-agent orchestration (run N agents, compare) | 🔲 Not yet | Need loop in `main.py` |
| Isolated memory mode (agent-specific `project_result_dir`) | 🔲 Not yet | Add `memory_project_dir` override per run |
| Read-only shared memory bank (pre-built seed bank) | 🔲 Partial | `memory_writeback_enabled: false` exists |
| FL evaluation comparison across agents | 🔲 Not yet | `scripts/compare_memory_runs.py` partial |
