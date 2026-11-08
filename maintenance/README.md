# CI-Repair-Bench Maintenance System

This directory contains the automated maintenance infrastructure for CI-Repair-Bench, designed to support the monthly health check procedure described in Section 3.3 of the paper.

## Overview

As described in the paper:
> *"We perform monthly health checks by re-executing each retained instance with its ground-truth patch. Failed or inconsistent executions are flagged for manual inspection and repair. All changes are documented through versioned benchmark releases with changelogs recording modified or removed instances and updates to workflows or ground-truth patches."*

## Components

### 1. Monthly Maintenance Orchestrator (`run_monthly_maintenance.py`)
The main script that automates the entire monthly maintenance workflow:
- Runs health checks on all benchmark instances
- Analyzes results and compares with previous runs
- Generates changelog entries automatically
- Creates versioned releases with documentation
- Provides actionable recommendations

**Usage:**
```bash
# Run full monthly maintenance
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/benchmark.parquet

# Dry run (health check only, no version update)
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/benchmark.parquet \
  --dry-run

# Create minor version update
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/benchmark.parquet \
  --version-level minor

# Use custom version
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/benchmark.parquet \
  --version 1.2.0
```

### 2. Version Manager (`version_manager.py`)
Manages semantic versioning and changelog updates:
- Tracks current version in `VERSION.txt`
- Increments versions following semantic versioning
- Updates `CHANGELOG.md` with structured entries
- Supports major, minor, and patch releases

**Version scheme:**
- **MAJOR**: Breaking changes (e.g., dataset format changes, API changes)
- **MINOR**: Instance additions/removals, significant repairs, workflow updates
- **PATCH**: Minor fixes, small workflow updates, documentation

**Usage:**
```bash
# Show current version
python3 maintenance/version_manager.py current

# Initialize changelog
python3 maintenance/version_manager.py init-changelog

# Create manual release
python3 maintenance/version_manager.py release patch
```

### 3. Changelog Generator (`changelog_generator.py`)
Automatically generates changelog entries from health check results:
- Analyzes health check outcomes
- Compares with previous runs to detect changes
- Categorizes changes (Added, Changed, Fixed, Removed, Flagged)
- Generates human-readable maintenance summaries

**Usage:**
```bash
# Generate changelog from health check results
python3 maintenance/changelog_generator.py \
  --health-check maintenance/reports/health_check/health_check_202610_full.json \
  --previous maintenance/reports/health_check/health_check_202609_full.json \
  --output maintenance/reports/maintenance_summary_202610.md
```

### 4. Health Check System (`run_health_check.py`)
Re-executes all benchmark instances with ground-truth patches:
- Validates that ground-truth patches still work
- Flags instances that fail, timeout, or are cancelled
- Generates detailed reports for manual inspection
- Tracks workflow execution status
- Monitors execution time and queue delays via GitHub Actions polling

### 5. Flaky Workflow Detector (`detect_flaky_workflows.py`)
Analyzes workflow runs to detect flakiness:
- Identifies inconsistent outcomes across multiple runs
- Detects timing variations exceeding threshold
- Flags intermittent failures
- Helps distinguish real failures from flaky tests

**Usage:**
```bash
# Analyze workflow runs for flakiness
python3 maintenance/detect_flaky_workflows.py \
  --runs 3 \
  --threshold 0.5
```

### 6. Hugging Face Dataset Comparator (`compare_with_huggingface.py`)
Compares local dataset with official Hugging Face version:
- Downloads latest dataset from Hugging Face
- Identifies added, removed, and modified instances
- Generates detailed comparison reports
- Documents changes for version releases

**Usage:**
```bash
# Compare with HuggingFace dataset
python3 maintenance/compare_with_huggingface.py \
  --local-dataset data/ci_repair_bench_v1.parquet \
  --hf-repo RabeyaMuna/ci-repair-bench
```

## Monthly Maintenance Workflow

### Automated Steps

1. **Health Check Execution**
   - Re-executes all retained instances with ground-truth patches
   - Monitors workflow completion (pass/fail/timeout/cancelled)
   - Tracks execution time and queue delays
   - Generates detailed results in `maintenance/reports/health_check/`

2. **Result Analysis**
   - Compares with previous month's results
   - Identifies newly broken or fixed instances
   - Categorizes issues (failed, timeout, invalid, etc.)

3. **Documentation Generation**
   - Creates maintenance summary with recommendations
   - Generates changelog entries automatically
   - Saves detailed reports for manual review

4. **Version Release**
   - Increments version number appropriately
   - Updates CHANGELOG.md with structured entries
   - Records all changes in VERSION.txt

### Manual Steps (After Automation)

5. **Manual Inspection**
   - Review flagged instances in generated reports
   - Investigate root causes of failures
   - Check for upstream changes (Actions versions, dependencies, runner environments)

6. **Instance Repair**
   - Update ground-truth patches for broken instances
   - Remove invalid instances from dataset
   - Update workflow files if needed

7. **Verification**
   - Re-run health check on repaired instances
   - Verify fixes before final release

8. **Release Documentation**
   - Review and enhance auto-generated changelog
   - Document specific changes made during repair
   - Update VERSION.txt if needed

## Output Files

### Health Check Reports (`maintenance/reports/health_check/`)
- `health_check_YYYYMM_full.json`: Complete results for all instances
- `health_check_YYYYMM_flagged.json`: Instances requiring attention

### Maintenance Reports (`maintenance/reports/`)
- `maintenance_report_YYYYMM.json`: Detailed report with recommendations
- `maintenance_summary_YYYYMM.md`: Human-readable summary

### Version Control
- `VERSION.txt`: Current benchmark version
- `CHANGELOG.md`: Structured changelog following Keep a Changelog format

## Maintenance Schedule

As stated in the paper:
> *"Based on our experience maintaining CI-Repair-Bench, failures occurred approximately three times over a ten-month period."*

**Recommended schedule:**
- **Monthly health checks**: First week of each month
- **Quarterly detailed reviews**: Every 3 months, deep inspection of all flagged instances
- **Immediate repairs**: When breakage rate exceeds 5%

## Example: Typical Maintenance Run

```bash
# 1. Run monthly maintenance (first week of November 2026)
python3 maintenance/run_monthly_maintenance.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet

# Output will show:
# - 250/250 instances passed (100% pass rate) ✅
# - OR flagged instances requiring attention ⚠️

# 2. If instances are flagged, review reports
cat maintenance/reports/maintenance_summary_202611.md
cat maintenance/reports/health_check/health_check_202611_flagged.json

# 3. Manually repair flagged instances
# (Update ground-truth patches, fix workflows, etc.)

# 4. Re-run health check on repaired instances
python3 maintenance/run_health_check.py \
  --config config.yaml \
  --dataset data/ci_repair_bench_v1.parquet \
  --ids 42 67 89

# 5. If all pass, maintenance is complete
# VERSION.txt and CHANGELOG.md are already updated!
```

## Changelog Format

Entries follow [Keep a Changelog](https://keepachangelog.com/) format:

- **Added**: New instances or features
- **Changed**: Updates to existing instances or workflows
- **Fixed**: Repaired instances or corrected patches
- **Removed**: Instances removed from benchmark
- **Maintenance Notes**: Flagged issues and observations

## Integration with Paper

This maintenance system directly implements the procedure described in Section 3.3:

| Paper Statement | Implementation |
|----------------|----------------|
| "Monthly health checks" | `run_monthly_maintenance.py` |
| "Re-executing each retained instance with ground-truth patch" | `health_check/run_health_check.py` |
| "Flagged for manual inspection and repair" | Flagged instance reports |
| "Versioned benchmark releases with changelogs" | `version_manager.py` + `CHANGELOG.md` |
| "Recording modified or removed instances" | Structured changelog entries |
| "Updates to workflows or ground-truth patches" | Change categorization and documentation |

## Troubleshooting

### No GitHub Token
```bash
export GITHUB_TOKEN=your_token_here
```

### Health Check Timeouts
Adjust timeout in `health_check/run_health_check.py`:
```python
max_wait = 600  # Increase from 10 minutes
```

### High Failure Rate
1. Check GitHub Actions status: https://www.githubstatus.com/
2. Review recent upstream changes (Actions versions, runner images)
3. Consider temporary pause if widespread outage

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- CI-Repair-Bench Paper, Section 3.3: Maintenance and Evolution
