# Trajectory Logging - INTEGRATED ✅

## What Was Done

Integrated simple trajectory logging into `baselines/main.py`. Now **every issue** will save a `trajectory.json` file with:

1. ✅ CI log analysis results
2. ✅ **Memory retrieval details** (similarity scores, matched levels, matches)
3. ✅ Fault localization predictions
4. ✅ Patch generation results
5. ✅ Token usage and cost

## Output Location

```
results/trajectories/{model}/{condition}/{level}/{task_id}/trajectory.json
```

**Examples:**
- Baseline: `results/trajectories/gpt-4o-mini/baseline/123/trajectory.json`
- Memory L1: `results/trajectories/gpt-4o-mini/memory/L1/123/trajectory.json`
- Memory L1+L2+L3: `results/trajectories/gpt-4o-mini/memory/L1L2L3/123/trajectory.json`

## Trajectory Format

```json
{
  "task_id": "123",
  "sha_fail": "abc123",
  "repo_name": "example-repo",
  "model": "gpt-4o-mini",
  "condition": "memory",
  "level": "L1L2L3",
  "patch_generated": true,
  "fl_predictions": ["src/foo.py", "src/bar.py"],
  "total_tokens": {"input": 50000, "output": 5000},
  "cost_usd": 0.15,
  "duration_sec": 45.2,
  
  "steps": [
    {
      "step": 1,
      "action": "ci_log_analysis",
      "input": {"logs_count": 3, "cached": false},
      "output": {
        "failed_steps": ["test"],
        "error_description": "ImportError: cannot import..."
      },
      "success": true,
      "duration": 2.5
    },
    {
      "step": 2,
      "action": "memory_retrieval",
      "input": {"ablation_levels": "L1+L2+L3"},
      "output": {
        "enabled": true,
        "memory_injected": true,
        "weighted_similarity": 0.65,
        "level_scores": {"L1": 0.70, "L2": 0.55, "L3": 0.45},
        "selected_memory_levels": ["L1", "L2"],
        "matches_count": 3,
        "l1_matches": [...],
        "l2_matches": [...],
        "l3_matches": [...],
        "candidate_files": [...],
        "high_level_hints": [...]
      },
      "success": true
    },
    {
      "step": 3,
      "action": "fault_localization",
      "input": {"memory_enabled": true},
      "output": {
        "predicted_files": ["src/foo.py", "src/bar.py"],
        "files_count": 2,
        "memory_was_injected": true
      },
      "success": true,
      "duration": 15.3
    },
    {
      "step": 4,
      "action": "patch_generation",
      "input": {"fl_files_count": 2},
      "output": {
        "patch_generated": true,
        "patched_files": ["src/foo.py"],
        "patched_files_count": 1
      },
      "success": true,
      "duration": 24.2
    }
  ]
}
```

## Key Features for Investigation

### Memory Retrieval Details

The `memory_retrieval` step contains:

```json
{
  "enabled": true,                          // Was memory plugin enabled?
  "memory_injected": true,                  // Was memory actually used?
  "weighted_similarity": 0.65,              // Overall similarity score
  "level_scores": {                         // Scores per level
    "L1": 0.70,
    "L2": 0.55,
    "L3": 0.45
  },
  "selected_memory_levels": ["L1", "L2"],  // Which levels passed threshold?
  "matches_count": 3,                       // How many matches total?
  "l1_matches": [...],                      // Actual L1 matches
  "l2_matches": [...],                      // Actual L2 matches
  "l3_matches": [...],                      // Actual L3 matches
  "candidate_files": [...],                 // Files suggested by memory
  "high_level_hints": [...]                 // High-level repair hints
}
```

**This tells you:**
- ✅ Was memory retrieved? (`memory_injected`)
- ✅ What similarity scores? (`weighted_similarity`, `level_scores`)
- ✅ Which levels were selected? (`selected_memory_levels`)
- ✅ What actual matches? (`l1_matches`, `l2_matches`, `l3_matches`)

### Manual Investigation Made Easy

For any issue, check:

```bash
# View memory retrieval for issue 123
cat results/trajectories/gpt-4o-mini/memory/L1L2L3/123/trajectory.json | \
  jq '.steps[] | select(.action=="memory_retrieval") | .output'

# Compare FL predictions
echo "Baseline:"
cat results/trajectories/gpt-4o-mini/baseline/123/trajectory.json | jq '.fl_predictions'

echo "Memory:"
cat results/trajectories/gpt-4o-mini/memory/L1L2L3/123/trajectory.json | jq '.fl_predictions'

# Check if patch was generated
cat results/trajectories/gpt-4o-mini/baseline/123/trajectory.json | jq '.patch_generated'
```

### Using the Viewer

```bash
# View single trajectory
python view_simple_trajectory.py --model gpt-4o-mini --condition baseline --task-id 123

# Compare across all conditions
python view_simple_trajectory.py --model gpt-4o-mini --task-id 123 --compare
```

## Usage

Just run your baseline as normal:

```bash
cd baselines
python main.py
```

**Trajectories are saved automatically** for every issue!

## Next Steps for Investigation

1. **Run 5 test issues** for all conditions (baseline, L1, L1L2, L1L2L3)

2. **Check memory retrieval** for each:
   ```bash
   # Quick check script
   for task in 123 124 125 126 127; do
     echo "=== Issue $task ==="
     cat results/trajectories/gpt-4o-mini/memory/L1L2L3/$task/trajectory.json | \
       jq '{memory: .steps[] | select(.action=="memory_retrieval") | .output | {injected, similarity: .weighted_similarity}}'
   done
   ```

3. **Compare outcomes**:
   ```bash
   python view_simple_trajectory.py --model gpt-4o-mini --task-id 123 --compare
   ```

4. **Manual verification**:
   - Open trajectory JSON
   - Check `l1_matches`, `l2_matches`, `l3_matches`
   - Verify if matched issues are actually relevant

## Summary

✅ **Integrated into main.py**
✅ **Saves per issue: `results/trajectories/{model}/{condition}/{level}/{task_id}/trajectory.json`**
✅ **Includes detailed memory retrieval info**
✅ **Ready for manual investigation**

Now you can answer the professor's questions with concrete data!
