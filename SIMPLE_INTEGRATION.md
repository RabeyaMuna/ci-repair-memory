# Simple Trajectory Integration

## Overview

**One file per issue**: `trajectory.json`

Similar to mini-swe-agent - clean and simple.

## Integration in main.py

### 1. Import

```python
from utilities.trajectory_logger import create_logger
```

### 2. Initialize (per issue)

```python
# After extracting issue data
trajectory_logger = create_logger(
    result_dir=config.project_result_dir,
    model=model_key,
    condition="baseline",  # or "memory"
    level="L1L2L3",  # only if memory
    task_id=task_id,
    sha_fail=sha_fail,
    repo_name=repo_name
)
```

### 3. Log steps (one line per action)

```python
# CI Log Analysis
start = time.time()
log_analysis_result = CILogAnalyzerLLM(...).run()
trajectory_logger.log(
    "ci_log_analysis",
    {"logs_count": len(logs)},
    {"failed_steps": log_analysis_result.get("failed_steps", [])},
    duration=time.time() - start
)

# Memory Retrieval
if memory_plugin.is_enabled():
    start = time.time()
    memory_context = memory_plugin.retrieve(memory_query)
    trajectory_logger.log(
        "memory_retrieval",
        {"ablation_levels": str(config.get("memory_ablation_levels"))},
        {
            "memory_injected": bool(memory_context.get("matches")),
            "similarity": memory_context.get("weighted_similarity", 0.0)
        },
        duration=time.time() - start
    )

# Fault Localization
start = time.time()
fault_localizer = FaultLocalization(...).run()
fl_files = [f.get("file_path") for f in fault_localizer.get("fault_localization_data", [])]
trajectory_logger.log(
    "fault_localization",
    {"memory_enabled": bool(config.get("memory_enabled"))},
    {"predicted_files": fl_files},
    duration=time.time() - start
)

# Patch Generation
start = time.time()
patch_generator = PatchGeneration(...).run()
trajectory_logger.log(
    "patch_generation",
    {},
    {"patch_generated": bool(patch_generator.get("diff"))},
    duration=time.time() - start
)
```

### 4. Set results and save

```python
# At the end of issue processing
trajectory_logger.set_result(
    patch_generated=bool(patch_generator.get("diff")),
    fl_predictions=fl_files,
    total_tokens={"input": 50000, "output": 5000},  # from tracker
    cost=0.15  # from tracker
)

trajectory_logger.save()
```

## Output Format

### trajectory.json

```json
{
  "task_id": "123",
  "sha_fail": "abc123",
  "repo_name": "example-repo",
  "model": "gpt-4o-mini",
  "condition": "memory",
  "level": "L1L2L3",
  "start_time": 1234567890.123,
  "end_time": 1234567935.456,
  "duration_sec": 45.33,
  "patch_generated": true,
  "fl_predictions": ["src/foo.py", "src/bar.py"],
  "total_tokens": {
    "input": 50000,
    "output": 5000
  },
  "cost_usd": 0.15,
  "steps": [
    {
      "step": 1,
      "action": "ci_log_analysis",
      "timestamp": 1234567890.123,
      "input": {
        "logs_count": 3
      },
      "output": {
        "failed_steps": ["test"]
      },
      "success": true,
      "error": null,
      "duration": 2.5
    },
    {
      "step": 2,
      "action": "memory_retrieval",
      "timestamp": 1234567892.623,
      "input": {
        "ablation_levels": "L1+L2+L3"
      },
      "output": {
        "memory_injected": true,
        "similarity": 0.65
      },
      "success": true,
      "error": null,
      "duration": 3.2
    },
    {
      "step": 3,
      "action": "fault_localization",
      "timestamp": 1234567895.823,
      "input": {
        "memory_enabled": true
      },
      "output": {
        "predicted_files": ["src/foo.py", "src/bar.py"]
      },
      "success": true,
      "error": null,
      "duration": 15.3
    },
    {
      "step": 4,
      "action": "patch_generation",
      "timestamp": 1234567911.123,
      "input": {},
      "output": {
        "patch_generated": true
      },
      "success": true,
      "error": null,
      "duration": 24.2
    }
  ]
}
```

## Directory Structure

```
results/
  trajectories/
    gpt-4o-mini/
      baseline/
        123/
          trajectory.json
        124/
          trajectory.json
      memory/
        L1/
          123/
            trajectory.json
        L1L2/
          123/
            trajectory.json
        L1L2L3/
          123/
            trajectory.json
```

## Complete Example

```python
# In main.py, inside the datapoint loop:

for datapoint in subset:
    task_id = datapoint["id"]
    sha_fail = datapoint["sha_fail"]
    repo_name = datapoint["repo_name"]
    # ...

    # Determine condition and level
    memory_enabled = bool(config.get("memory_enabled", False))
    if memory_enabled:
        condition = "memory"
        ablation = str(config.get("memory_ablation_levels", "L1+L2+L3"))
        level = ablation.replace("+", "")  # "L1L2L3"
    else:
        condition = "baseline"
        level = None

    # Create logger
    trajectory_logger = create_logger(
        result_dir=config.project_result_dir,
        model=model_key,
        condition=condition,
        level=level,
        task_id=task_id,
        sha_fail=sha_fail,
        repo_name=repo_name
    )

    try:
        # Step 1: CI Log Analysis
        start = time.time()
        log_analysis_result = CILogAnalyzerLLM(...).run()
        trajectory_logger.log(
            "ci_log_analysis",
            {"logs_count": len(logs), "cached": cached_entry is not None},
            {"failed_steps": log_analysis_result.get("failed_steps", [])},
            success=not log_analysis_result.get("error"),
            duration=time.time() - start
        )

        # Step 2: Memory Retrieval (if enabled)
        if memory_plugin.is_enabled():
            start = time.time()
            memory_query = memory_plugin.build_query(...)
            memory_context = memory_plugin.retrieve(memory_query)
            trajectory_logger.log(
                "memory_retrieval",
                {"ablation_levels": ablation},
                {
                    "memory_injected": bool(memory_context.get("matches")),
                    "weighted_similarity": memory_context.get("weighted_similarity", 0.0),
                    "selected_levels": memory_context.get("selected_memory_levels", [])
                },
                duration=time.time() - start
            )

        # Step 3: Fault Localization
        start = time.time()
        fault_localizer = FaultLocalization(...).run()
        fl_files = [f.get("file_path") for f in fault_localizer.get("fault_localization_data", [])]
        trajectory_logger.log(
            "fault_localization",
            {"memory_enabled": memory_enabled},
            {"predicted_files": fl_files, "count": len(fl_files)},
            duration=time.time() - start
        )

        # Step 4: Patch Generation
        start = time.time()
        patch_generator = PatchGeneration(...).run()
        has_patch = bool(patch_generator.get("diff"))
        trajectory_logger.log(
            "patch_generation",
            {"fl_files_count": len(fl_files)},
            {"patch_generated": has_patch},
            success=has_patch,
            duration=time.time() - start
        )

        # Set final results
        task_summary = tracker.get_task_summary(task_id) if tracker else {}
        trajectory_logger.set_result(
            patch_generated=has_patch,
            fl_predictions=fl_files,
            total_tokens=task_summary.get("total_tokens", {"input": 0, "output": 0}),
            cost=task_summary.get("total_cost_usd", 0.0)
        )

    except Exception as e:
        # Log error
        trajectory_logger.log(
            "error",
            {},
            {},
            success=False,
            error=str(e)
        )

    finally:
        # Always save
        trajectory_logger.save()
```

## That's It!

**Simple, clean, production-ready.**

- One `trajectory.json` per issue
- Minimal integration (~10 lines)
- All data in one place
- Easy to read and analyze
