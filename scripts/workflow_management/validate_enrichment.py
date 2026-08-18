"""
Validate the enriched dataset to ensure all metadata was added correctly.
"""

import pandas as pd
import json
import os
from pathlib import Path


def validate_enriched_dataset(enriched_path: str, failed_jobs_path: str):
    """
    Validate the enriched dataset structure and data quality.

    Args:
        enriched_path: Path to enriched parquet file
        failed_jobs_path: Path to failed_jobs_all.json for cross-validation
    """
    print("="*70)
    print("DATASET ENRICHMENT VALIDATION")
    print("="*70)

    # Load enriched dataset
    if not os.path.exists(enriched_path):
        print(f"❌ ERROR: Enriched dataset not found at {enriched_path}")
        return False

    print(f"\n📂 Loading enriched dataset from: {enriched_path}")
    df = pd.read_parquet(enriched_path)
    print(f"✓ Loaded {len(df)} records")

    # Check required columns exist
    print("\n" + "="*70)
    print("1. CHECKING REQUIRED COLUMNS")
    print("="*70)

    required_new_columns = [
        'total_jobs',
        'num_failed_jobs',
        'failed_jobs',
        'lines_inserted',
        'lines_deleted',
        'total_lines_changed'
    ]

    missing_columns = []
    for col in required_new_columns:
        if col in df.columns:
            print(f"✓ {col:25s} - present")
        else:
            print(f"❌ {col:25s} - MISSING")
            missing_columns.append(col)

    if missing_columns:
        print(f"\n❌ VALIDATION FAILED: Missing columns {missing_columns}")
        return False

    # Check existing columns
    print("\n" + "="*70)
    print("2. CHECKING EXISTING COLUMNS")
    print("="*70)

    existing_columns = ['id', 'repo_name', 'diff', 'changed_files']
    for col in existing_columns:
        if col in df.columns:
            print(f"✓ {col:25s} - present")
        else:
            print(f"⚠ {col:25s} - missing (may affect enrichment)")

    # Validate data types
    print("\n" + "="*70)
    print("3. VALIDATING DATA TYPES")
    print("="*70)

    type_checks = {
        'total_jobs': 'int',
        'num_failed_jobs': 'int',
        'failed_jobs': 'object',  # List stored as object
        'lines_inserted': 'int',
        'lines_deleted': 'int',
        'total_lines_changed': 'int'
    }

    for col, expected_type in type_checks.items():
        actual_type = str(df[col].dtype)
        if expected_type in actual_type or (expected_type == 'int' and 'int' in actual_type):
            print(f"✓ {col:25s} - {actual_type}")
        else:
            print(f"❌ {col:25s} - Expected {expected_type}, got {actual_type}")

    # Validate data ranges
    print("\n" + "="*70)
    print("4. VALIDATING DATA RANGES")
    print("="*70)

    # Non-negative integers
    for col in ['total_jobs', 'num_failed_jobs', 'lines_inserted', 'lines_deleted', 'total_lines_changed']:
        min_val = df[col].min()
        max_val = df[col].max()

        if min_val >= 0:
            print(f"✓ {col:25s} - range [{min_val:,} to {max_val:,}]")
        else:
            print(f"❌ {col:25s} - Contains negative values: min = {min_val}")

    # Validate total_lines_changed = lines_inserted + lines_deleted
    print("\n" + "="*70)
    print("5. VALIDATING COMPUTED FIELDS")
    print("="*70)

    df['computed_total'] = df['lines_inserted'] + df['lines_deleted']
    mismatches = df[df['total_lines_changed'] != df['computed_total']]

    if len(mismatches) == 0:
        print("✓ total_lines_changed = lines_inserted + lines_deleted (all records)")
    else:
        print(f"❌ {len(mismatches)} records have incorrect total_lines_changed")
        print("  First mismatch:")
        print(mismatches[['id', 'lines_inserted', 'lines_deleted', 'total_lines_changed', 'computed_total']].head(1))

    # Validate failed_jobs structure
    print("\n" + "="*70)
    print("6. VALIDATING FAILED_JOBS STRUCTURE")
    print("="*70)

    sample_failed_jobs = df['failed_jobs'].dropna().head(5)
    valid_structure = True

    for idx, jobs in sample_failed_jobs.items():
        if isinstance(jobs, list):
            if all(isinstance(job, str) for job in jobs):
                print(f"✓ Record {df.loc[idx, 'id']:5s} - failed_jobs: {jobs}")
            else:
                print(f"❌ Record {df.loc[idx, 'id']:5s} - failed_jobs contains non-string items")
                valid_structure = False
        else:
            print(f"❌ Record {df.loc[idx, 'id']:5s} - failed_jobs is not a list: {type(jobs)}")
            valid_structure = False

    # Validate num_failed_jobs matches failed_jobs length
    print("\n" + "="*70)
    print("7. VALIDATING FAILED JOB COUNTS")
    print("="*70)

    df['computed_num_failed'] = df['failed_jobs'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    count_mismatches = df[df['num_failed_jobs'] != df['computed_num_failed']]

    if len(count_mismatches) == 0:
        print("✓ num_failed_jobs matches length of failed_jobs list (all records)")
    else:
        print(f"❌ {len(count_mismatches)} records have mismatched failed job counts")
        print("  First mismatch:")
        print(count_mismatches[['id', 'failed_jobs', 'num_failed_jobs', 'computed_num_failed']].head(1))

    # Cross-validate with failed_jobs_all.json
    if os.path.exists(failed_jobs_path):
        print("\n" + "="*70)
        print("8. CROSS-VALIDATING WITH FAILED_JOBS_ALL.JSON")
        print("="*70)

        with open(failed_jobs_path, 'r') as f:
            failed_jobs_data = json.load(f)

        failed_jobs_map = {}
        for item in failed_jobs_data:
            issue_id = str(item.get('id'))
            jobs = [job.get('step_name') for job in item.get('failed_jobs', [])]
            failed_jobs_map[issue_id] = jobs

        # Sample validation
        sample_ids = df['id'].head(10).astype(str).tolist()
        all_match = True

        for issue_id in sample_ids:
            expected = failed_jobs_map.get(issue_id, [])
            actual = df[df['id'] == issue_id]['failed_jobs'].iloc[0] if len(df[df['id'] == issue_id]) > 0 else []

            if set(expected) == set(actual if actual else []):
                print(f"✓ ID {issue_id:5s} - failed_jobs match: {expected}")
            else:
                print(f"❌ ID {issue_id:5s} - MISMATCH")
                print(f"  Expected: {expected}")
                print(f"  Got:      {actual}")
                all_match = False

        if all_match:
            print("\n✓ Sample validation passed (first 10 records)")
    else:
        print(f"\n⚠ Skipping cross-validation: {failed_jobs_path} not found")

    # Summary statistics
    print("\n" + "="*70)
    print("9. SUMMARY STATISTICS")
    print("="*70)

    stats = {
        'Total Records': len(df),
        'Records with failed_jobs data': df['failed_jobs'].notna().sum(),
        'Records with total_jobs > 0': (df['total_jobs'] > 0).sum(),
        'Records with line changes': (df['total_lines_changed'] > 0).sum(),
        'Avg total_jobs': df['total_jobs'].mean(),
        'Avg num_failed_jobs': df['num_failed_jobs'].mean(),
        'Avg lines_inserted': df['lines_inserted'].mean(),
        'Avg lines_deleted': df['lines_deleted'].mean(),
        'Avg total_lines_changed': df['total_lines_changed'].mean(),
        'Max total_jobs': df['total_jobs'].max(),
        'Max num_failed_jobs': df['num_failed_jobs'].max(),
        'Max total_lines_changed': df['total_lines_changed'].max(),
    }

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:35s}: {value:,.2f}")
        else:
            print(f"  {key:35s}: {value:,}")

    # Data completeness
    print("\n" + "="*70)
    print("10. DATA COMPLETENESS")
    print("="*70)

    completeness = {
        'total_jobs': (df['total_jobs'] > 0).sum() / len(df) * 100,
        'failed_jobs': df['failed_jobs'].notna().sum() / len(df) * 100,
        'lines_changed': (df['total_lines_changed'] > 0).sum() / len(df) * 100,
    }

    for field, percentage in completeness.items():
        status = "✓" if percentage > 95 else "⚠" if percentage > 50 else "❌"
        print(f"{status} {field:20s}: {percentage:6.2f}% complete")

    # Final verdict
    print("\n" + "="*70)
    print("VALIDATION RESULT")
    print("="*70)

    if not missing_columns and valid_structure:
        print("✅ VALIDATION PASSED - Dataset enrichment successful!")
        print("\nYou can now use the enriched dataset:")
        print(f"  {enriched_path}")
        return True
    else:
        print("❌ VALIDATION FAILED - Please review errors above")
        return False


if __name__ == "__main__":
    BASE_DIR = "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH"

    ENRICHED_PATH = os.path.join(BASE_DIR, "dataset", "lca_dataset_enriched.parquet")
    FAILED_JOBS_PATH = os.path.join(BASE_DIR, "dataset", "failed_jobs_all.json")

    success = validate_enriched_dataset(ENRICHED_PATH, FAILED_JOBS_PATH)

    print("\n" + "="*70)
    if success:
        print("Next steps:")
        print("  1. Use lca_dataset_enriched.parquet in your evaluation")
        print("  2. Analyze metadata using the examples in README_METADATA_ENRICHMENT.md")
        print("  3. Update your benchmark scripts to leverage new fields")
    else:
        print("Please fix the issues above and re-run the enrichment script.")
    print("="*70)
