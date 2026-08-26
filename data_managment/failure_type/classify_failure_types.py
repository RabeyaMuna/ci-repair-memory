#!/usr/bin/env python3
"""
Classify failure types using backward engineering approach.

Uses ground truth (diff + CI logs + validation steps) to determine:
- failure_type: High-level category
- failure_subtype: Specific issue that was fixed
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from data_managment.failure_type.dependency_chunker import chunk_diff_with_dependencies

# Load environment variables
load_dotenv()

# 12-type taxonomy for failure classification
FAILURE_TAXONOMY = {
    "Code Formatting": "Issues with code style, indentation, line length, etc.",
    "Linting": "Code quality issues detected by linters (unused imports, undefined vars, etc.)",
    "Syntax Error": "Python syntax errors, invalid code structure",
    "Runtime Error": "Errors that occur during execution (AttributeError, KeyError, etc.)",
    "Test Failure": "Unit/integration tests that fail",
    "Assertion Error": "Failed assertions in tests",
    "Type Checking": "Type annotation errors, mypy issues",
    "Dependency Issues": "Missing dependencies, version conflicts",
    "Package Install Error": "Failed package installation, missing requirements",
    "Configuration Error": "Invalid config files, wrong settings",
    "Environment Error": "Platform-specific issues, missing env vars",
    "Doc/Docstring": "Documentation format issues, missing docstrings"
}


def build_classification_prompt(ci_logs: str, diff_chunk: str, validation_steps: list,
                               log_details: dict = None, workflow_info: dict = None) -> str:
    """Build classification prompt using backward engineering approach from mini-swe-agent.

    Analyzes ground truth diff + CI context to identify ALL failure types fixed.
    One issue can have MULTIPLE failure types across different jobs/steps.
    """

    taxonomy_str = "\n".join([f"- {k}: {v}" for k, v in FAILURE_TAXONOMY.items()])

    # Build CI failure context based on available data
    # Priority: log_details > ci_logs > validation_steps only

    has_ci_logs = ci_logs and len(ci_logs.strip()) > 0

    if log_details:
        # BEST: Structured log details available
        ci_failure_context = {
            "workflow": workflow_info.get('workflow', 'CI validation') if workflow_info else "CI validation",
            "error_context": log_details.get('error_context', []),
            "failure_signals": log_details.get('failure_signals', []),
            "relevant_files": log_details.get('relevant_files', []),
            "error_types": log_details.get('error_types', []),
            "failed_jobs": log_details.get('failed_job', []),
            "has_ci_logs": True
        }
    elif has_ci_logs:
        # MEDIUM: Raw CI logs available
        formatted_validations = []
        for step in validation_steps:
            job_name = step.get('job_name', 'unknown')
            failed_steps = step.get('failed_steps', [])
            formatted_validations.append({
                "job_name": job_name,
                "failed_steps": failed_steps,
                "effective_cmd": f"{job_name}: {', '.join(failed_steps[:3])}"
            })

        ci_failure_context = {
            "workflow": workflow_info.get('workflow', 'CI validation') if workflow_info else "CI validation",
            "validations": formatted_validations,
            "error_excerpt": ci_logs[:2000],
            "has_ci_logs": True
        }
    else:
        # FALLBACK: Only validation steps + workflow (NO CI logs)
        formatted_validations = []
        for step in validation_steps:
            job_name = step.get('job_name', 'unknown')
            failed_steps = step.get('failed_steps', [])
            formatted_validations.append({
                "job_name": job_name,
                "failed_steps": failed_steps,
                "effective_cmd": f"{job_name}: {', '.join(failed_steps[:3])}"
            })

        ci_failure_context = {
            "workflow": workflow_info.get('workflow', 'CI validation') if workflow_info else "CI validation",
            "validations": formatted_validations,
            "note": "CI logs not available - analyzing based on workflow validation steps and ground truth changes only",
            "has_ci_logs": False
        }

    has_logs = ci_failure_context.get('has_ci_logs', False)

    # Adjust prompt based on available evidence
    if has_logs:
        evidence_note = """CLASSIFICATION BASIS - Use all three evidence sources together:
1. CI failure context (logs/structured details): identifies visible/primary failures,
   but may stop at the FIRST failure and may not show later broken steps.
2. Ground-truth diff: shows the COMPLETE repair, including hidden setup,
   dependency, tooling, config, source, docs, test, build, and workflow fixes.
3. Full workflow validation sequence: shows all CI validation steps.

**CRITICAL: CI logs are INCOMPLETE - they show only the FIRST failure.**
**Ground truth diff fixes ALL problems, including HIDDEN failures after the first.**"""
    else:
        evidence_note = """CLASSIFICATION BASIS - CI logs NOT available, use these two evidence sources:
1. Workflow validation steps: shows which jobs/steps failed (but not detailed error logs)
2. Ground-truth diff: shows the COMPLETE repair - analyze what was changed to infer
   what problems were fixed

**IMPORTANT: Without CI logs, rely heavily on ground truth diff analysis.**
**Examine BEFORE/AFTER changes carefully to determine what failures were fixed.**
**Use validation step names (e.g., "mypy", "pytest", "black") as hints about failure types.**"""

    prompt = f"""Classify each changed file by the CI step that would catch or require the fixed issue.

## CLASSIFICATION TAXONOMY (USE THESE 12 TYPES FOR failure_type)
{taxonomy_str}

## INPUT

CI failure context:
{json.dumps(ci_failure_context, indent=2)}

Changed files (ground truth diff):
{diff_chunk}

## TASK

{evidence_note}

Do NOT classify only from CI logs. A changed file absent from logs can still be
a required hidden fix for setup, installation, dependency, formatting, linting,
typing, tests, docs, build, or workflow execution.

For every distinct failure fixed in the diff:
1. Inspect before/after changes
2. Decide what CI validation failure the change fixes
3. Choose failure_type from the 12-type TAXONOMY above
4. Set failure_subtype to be SPECIFIC (e.g., "missing import", "wrong indentation")
5. Determine confidence based on clarity of the fix

Use two levels:
- failure_type: broad category from TAXONOMY (Type Checking, Linting, etc.)
- failure_subtype: specific issue (missing annotation, unused import, etc.)

IMPORTANT CONTEXT:
- One issue can fix MULTIPLE distinct failure types across different files/steps
- CI logs show FIRST failure → diff fixes FIRST + HIDDEN failures
- Files absent from CI logs can be hidden fixes
- Analyze COMPLETE diff to find ALL failures, not just first visible one

## OUTPUT FORMAT

Return ONLY a JSON object with this exact format:
{{
  "failures": [
    {{
      "failure_type": "<one of the 12 taxonomy types above>",
      "failure_subtype": "<specific issue, e.g., 'missing import', 'wrong indentation'>",
      "confidence": "<high|medium|low>"
    }},
    ... (return ALL distinct failure types, not just primary)
  ]
}}

REQUIREMENTS:
- failure_type MUST be exactly one of the 12 taxonomy types
- Return ALL distinct failure types found (CI shows FIRST, diff fixes ALL)
- If only one failure type, return array with one element
- Base on what diff shows was fixed (complete repair), not just CI error
- Return valid JSON only, no markdown
"""
    return prompt


def chunk_diff(diff: str, max_chars: int = 8000) -> list:
    """Chunk large diffs by file to keep complete file changes together."""
    if len(diff) <= max_chars:
        return [diff]

    chunks = []
    current_chunk = ""

    # Split by file (diff --git lines)
    files = re.split(r'(diff --git.*?\n)', diff)

    for i in range(1, len(files), 2):
        if i+1 < len(files):
            file_header = files[i]
            file_content = files[i+1]
            file_diff = file_header + file_content

            # If adding this file exceeds limit, save current chunk and start new
            if current_chunk and len(current_chunk) + len(file_diff) > max_chars:
                chunks.append(current_chunk)
                current_chunk = file_diff
            else:
                current_chunk += file_diff

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [diff]


def classify_chunk_with_llm(issue_id: str, ci_logs: str, diff_chunk: str,
                            validation_steps: list, chunk_idx: int, total_chunks: int,
                            client: OpenAI, log_details: dict = None, workflow_info: dict = None) -> list:
    """Classify failure types for one chunk of the diff."""

    # CRITICAL: Truncate CI logs to prevent token overflow
    # Some issues have 300k+ token logs, which exceed model limits
    MAX_CI_LOG_CHARS = 8000  # ~2k tokens, leaving room for diff + prompt
    if ci_logs and len(ci_logs) > MAX_CI_LOG_CHARS:
        # Take first and last parts to capture both initial failure and final context
        half = MAX_CI_LOG_CHARS // 2
        ci_logs_truncated = (
            ci_logs[:half] +
            f"\n\n... [TRUNCATED {len(ci_logs) - MAX_CI_LOG_CHARS} chars] ...\n\n" +
            ci_logs[-half:]
        )
        ci_logs = ci_logs_truncated

    prompt = build_classification_prompt(ci_logs, diff_chunk, validation_steps, log_details, workflow_info)

    # Add chunk info to prompt
    if total_chunks > 1:
        chunk_note = f"\n\nNOTE: This is chunk {chunk_idx}/{total_chunks} of the complete diff. Classify failures in THIS chunk only."
        prompt += chunk_note

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a CI/CD failure analysis expert. Identify ALL failure types in the ground truth diff, as one CI workflow can have multiple failed jobs/steps with different failure types. CI logs only show FIRST failure, but the diff fixes MULTIPLE problems."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )

        result = json.loads(response.choices[0].message.content)
        failures = result.get('failures', [])
        return failures

    except Exception as e:
        print(f"   ⚠️  Error in chunk {chunk_idx}: {str(e)}")
        return []


def classify_with_llm(issue_id: str, ci_logs: str, diff: str, validation_steps: list,
                     client: OpenAI, log_details: dict = None, workflow_info: dict = None) -> Dict[str, Any]:
    """Classify failure types using GPT-4o-mini with DEPENDENCY-AWARE chunking.

    Analyzes COMPLETE ground truth (not truncated) to find ALL hidden failures.
    Keeps related files (with import dependencies) together for better analysis.
    CI logs only show FIRST failure, but diff fixes MULTIPLE problems.
    """

    # Dependency-aware chunking (keeps related files together)
    try:
        chunk_data = chunk_diff_with_dependencies(diff, max_chars=8000)
    except Exception as e:
        print(f"   ⚠️  Dependency chunking failed, using simple chunking: {str(e)}")
        # Fallback to simple chunking
        simple_chunks = chunk_diff(diff, max_chars=8000)
        chunk_data = [{'content': c, 'files': [], 'has_dependencies': False} for c in simple_chunks]

    all_failures = []

    # Classify each chunk
    for idx, chunk_info in enumerate(chunk_data, 1):
        chunk_content = chunk_info['content']
        has_deps = chunk_info['has_dependencies']

        # Add dependency context to prompt if files are related
        chunk_note = ""
        if has_deps:
            files = chunk_info['files']
            chunk_note = f"\n\nNOTE: Files in this chunk have dependencies: {', '.join(files[:3])}"
            if len(files) > 3:
                chunk_note += f" and {len(files)-3} more"

        chunk_failures = classify_chunk_with_llm(
            issue_id, ci_logs, chunk_content + chunk_note,
            validation_steps, idx, len(chunk_data), client,
            log_details, workflow_info
        )
        all_failures.extend(chunk_failures)

    if not all_failures:
        return {
            "issue_id": issue_id,
            "failure_types": ["Unknown"],
            "failure_subtypes": ["classification_failed"],
            "confidences": ["low"],
            "num_failure_types": 1,
            "num_chunks": len(chunk_data)
        }

    # Deduplicate failures (same type + subtype)
    seen = set()
    unique_failures = []
    for f in all_failures:
        key = (f['failure_type'], f['failure_subtype'])
        if key not in seen:
            seen.add(key)
            unique_failures.append(f)

    # Extract as arrays
    failure_types = [f['failure_type'] for f in unique_failures]
    failure_subtypes = [f['failure_subtype'] for f in unique_failures]
    confidences = [f['confidence'] for f in unique_failures]

    return {
        "issue_id": issue_id,
        "failure_types": failure_types,
        "failure_subtypes": failure_subtypes,
        "confidences": confidences,
        "num_failure_types": len(unique_failures),
        "num_chunks": len(chunk_data)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Classify failure types using backward engineering')
    parser.add_argument('--dataset', type=str, default='dataset/lca_dataset.parquet',
                        help='Path to dataset')
    parser.add_argument('--preds', type=str, default='results/preds.json',
                        help='Predictions file with selected IDs (optional, use --all to classify all issues)')
    parser.add_argument('--all', action='store_true',
                        help='Classify all issues in dataset (ignore preds.json)')
    parser.add_argument('--log-details', type=str, default='results/log_details.json',
                        help='Structured log details JSON file')
    parser.add_argument('--output', type=str, default='results/failure_classifications.json',
                        help='Output file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of issues to classify (for testing)')
    args = parser.parse_args()

    print("="*80)
    print("FAILURE TYPE CLASSIFICATION (Backward Engineering)")
    print("="*80)
    print()

    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return 1

    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized")

    # Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_parquet(args.dataset)
    print(f"   ✓ Loaded {len(df)} issues")

    # Load structured log details if available
    print("\n📋 Loading structured log details...")
    log_details_map = {}
    if Path(args.log_details).exists():
        with open(args.log_details, 'r') as f:
            log_details_list = json.load(f)
            log_details_map = {str(item['id']): item for item in log_details_list if 'id' in item}
        print(f"   ✓ Loaded {len(log_details_map)} log details")
    else:
        print(f"   ⚠️  Log details not found, using raw logs")

    # Load selected IDs from predictions or use all
    print("\n📋 Loading selected IDs...")
    if args.all:
        print("   ✓ Using ALL issues from dataset")
        df_filtered = df
    elif Path(args.preds).exists():
        with open(args.preds, 'r') as f:
            selected_ids = list(json.load(f).keys())
        print(f"   ✓ {len(selected_ids)} IDs from predictions")
        df_filtered = df[df['id'].astype(str).isin(selected_ids)]
        print(f"   ✓ Filtered to {len(df_filtered)} issues")
    else:
        print("   ⚠️  No predictions file, using all issues")
        df_filtered = df

    if args.limit:
        df_filtered = df_filtered.head(args.limit)
        print(f"   ⚠️  Limited to {args.limit} issues for testing")

    # Classify each issue
    print("\n🔍 Classifying failure types...")
    classifications = []

    for idx, row in df_filtered.iterrows():
        issue_id = str(row['id'])
        print(f"   [{len(classifications)+1}/{len(df_filtered)}] Processing {issue_id}...", end='', flush=True)

        # Extract data (handle potential arrays/None/pandas types)
        def safe_str(value):
            """Convert value to string safely."""
            if value is None:
                return ''
            if isinstance(value, str):
                return value
            # Handle pandas/numpy arrays
            if hasattr(value, 'tolist'):
                return str(value.tolist()) if len(value) > 0 else ''
            return str(value)

        ci_logs = safe_str(row.get('ci_logs'))
        if not ci_logs:
            ci_logs = safe_str(row.get('logs'))

        diff = safe_str(row.get('diff'))

        # Get validation steps (failed jobs with their steps)
        validation_steps = []
        failed_jobs = row.get('failed_jobs', [])
        if isinstance(failed_jobs, list):
            for job in failed_jobs:
                if isinstance(job, dict):
                    validation_steps.append({
                        'job_name': job.get('job_name', 'unknown'),
                        'failed_steps': job.get('failed_steps', [])
                    })

        # Get structured log details if available
        log_details = log_details_map.get(issue_id)

        # Get workflow info
        workflow_info = {
            'workflow': safe_str(row.get('workflow', '')),
            'repo_name': safe_str(row.get('repo_name', ''))
        }

        # Classify
        result = classify_with_llm(issue_id, ci_logs, diff, validation_steps, client,
                                   log_details, workflow_info)
        classifications.append(result)

        # Show all failure types found
        types_str = ", ".join(result['failure_types'][:3])
        if result['num_failure_types'] > 3:
            types_str += f" +{result['num_failure_types']-3} more"
        print(f" ✓ [{result['num_failure_types']}] {types_str}")

    # Add classifications to dataset
    print(f"\n📝 Adding classifications to dataset...")

    # Make a copy to avoid SettingWithCopyWarning
    df_filtered = df_filtered.copy()

    # Create classification lookup
    classification_map = {c['issue_id']: c for c in classifications}

    # Add columns to dataframe (as lists)
    df_filtered['failure_types'] = df_filtered['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('failure_types', ['Unknown'])
    )
    df_filtered['failure_subtypes'] = df_filtered['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('failure_subtypes', ['unknown'])
    )
    df_filtered['num_failure_types'] = df_filtered['id'].astype(str).map(
        lambda x: classification_map.get(x, {}).get('num_failure_types', 1)
    )

    # Get unique lists (flatten arrays)
    all_types = []
    all_subtypes = []
    for c in classifications:
        all_types.extend(c['failure_types'])
        all_subtypes.extend(c['failure_subtypes'])

    unique_types = sorted(set(all_types))
    unique_subtypes = sorted(set(all_subtypes))

    # Save enriched dataset
    enriched_dataset_path = args.dataset.replace('.parquet', '_with_failure_types.parquet')
    df_filtered.to_parquet(enriched_dataset_path, index=False)
    print(f"   ✓ Saved enriched dataset: {enriched_dataset_path}")

    # Save detailed results
    print(f"\n💾 Saving detailed results to {args.output}...")
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)

    output_data = {
        "total_classified": len(classifications),
        "taxonomy": FAILURE_TAXONOMY,
        "unique_failure_types": unique_types,
        "unique_failure_subtypes": unique_subtypes,
        "classifications": classifications
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("CLASSIFICATION SUMMARY")
    print("="*80)

    # Count total failure types (can be > number of issues)
    type_counts = {}
    multi_type_issues = sum(1 for c in classifications if c['num_failure_types'] > 1)
    total_failure_instances = sum(c['num_failure_types'] for c in classifications)

    for c in classifications:
        for ftype in c['failure_types']:
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

    print(f"\nTotal issues classified: {len(classifications)}")
    print(f"Issues with multiple failure types: {multi_type_issues}")
    print(f"Total failure type instances: {total_failure_instances}")

    print("\nFailure type distribution (total occurrences):")
    for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {ftype}: {count}")

    print(f"\n📊 Unique failure types ({len(unique_types)}):")
    print(f"   {unique_types}")

    print(f"\n📊 Unique failure subtypes ({len(unique_subtypes)}):")
    for subtype in unique_subtypes[:10]:
        print(f"   - {subtype}")
    if len(unique_subtypes) > 10:
        print(f"   ... and {len(unique_subtypes) - 10} more")

    print("\n" + "="*80)
    print(f"✓ Enriched dataset: {enriched_dataset_path}")
    print(f"✓ Detailed results: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
