#!/usr/bin/env python3
"""
Poll pending triggered instances from triggered_waiting.json.

For each instance's pending commit(s) (every commit between sha_fail and
sha_success, inclusive), check whether the pushed branch's workflow run has
completed. Once ALL commits are resolved for an instance, merge the completed
instance into all_instances_metadata.json and remove it from
triggered_waiting.json.

Run this after fetch_and_trigger_metadata.py --trigger has pushed the
workflows and given them time to run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fetch_and_trigger_metadata import (  # noqa: E402
    OUTPUT_DIR, OUTPUT_FILE, TRIGGER_FILE,
    get_workflow_runs_for_branch, fetch_jobs_for_run, process_jobs_to_json_format,
    load_trigger_entries, save_trigger_entries, save_all_metadata, is_instance_valid
)


def resolve_commit_item(entry: dict, item: dict) -> bool:
    """Check a single pending commit item; update it in place. Returns True if state changed."""
    if not item.get('pending'):
        return False

    branch_name = item.get('triggered_branch')
    triggered_in_fork = item.get('triggered_in_fork')
    if not branch_name or not triggered_in_fork:
        return False

    # The branch was pushed to the fork (triggered_in_fork), not the original repo
    fork_owner, fork_repo = triggered_in_fork.split('/', 1)

    runs = get_workflow_runs_for_branch(fork_owner, fork_repo, branch_name, entry['workflow_file'])
    if not runs or runs[0]['status'] != 'completed':
        return False

    run = runs[0]
    jobs = fetch_jobs_for_run(fork_owner, fork_repo, run['id'])
    jobs_data = process_jobs_to_json_format(jobs) if jobs else {
        'overall_jobs': [], 'overall_jobs_count': 0, 'overall_steps_count': 0,
        'failed_jobs': [], 'no_failed_jobs': 0
    }

    item.update({
        'workflow_run_id': run['id'],
        'workflow_conclusion': run['conclusion'],
        'run_html_url': run.get('html_url'),
        'run_number': run.get('run_number'),
        'pending': False,
        **jobs_data
    })
    return True


def build_completed_metadata(entry: dict) -> dict:
    """Assemble the final all_instances_metadata.json entry from a resolved trigger entry."""
    return {
        'id': entry['id'],
        'repo': f"{entry['repo_owner']}/{entry['repo_name']}",
        'repo_owner': entry['repo_owner'],
        'repo_name': entry['repo_name'],
        'workflow_name': entry['workflow_name'],
        'workflow_file': entry['workflow_file'],
        'sha_fail': entry['sha_fail'],
        'sha_success': entry['sha_success'],
        'compare': entry['compare'],
        'commit_metadata': entry['commit_metadata'],
    }


def main():
    trigger_path = os.path.join(OUTPUT_DIR, TRIGGER_FILE)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    trigger_entries = load_trigger_entries(trigger_path)
    if not trigger_entries:
        print(f"No pending instances in {trigger_path}")
        return

    metadata_by_id = {}
    if os.path.exists(output_path):
        import json
        with open(output_path, 'r') as f:
            for item in json.load(f):
                if is_instance_valid(item):
                    metadata_by_id[item['id']] = item

    print(f"Polling {len(trigger_entries)} pending instance(s)...")
    print('=' * 80)

    resolved_count = 0

    for entry in trigger_entries:
        if entry['id'] in metadata_by_id:
            continue

        print(f"\n  ID {entry['id']}: {entry['repo_owner']}/{entry['repo_name']}")
        for item in entry['commit_metadata']:
            if resolve_commit_item(entry, item):
                print(f"    {item['commit_type'].upper()} ({item['original_commit'][:7]})... ✓ {item['workflow_conclusion']}")
            elif item.get('pending'):
                print(f"    {item['commit_type'].upper()} ({item['original_commit'][:7]})... ⏳ still running")

        completed = build_completed_metadata(entry)
        if is_instance_valid(completed):
            metadata_by_id[entry['id']] = completed
            resolved_count += 1
            save_all_metadata(list(metadata_by_id.values()), output_path)

        # Persist progress after every instance so an interruption loses nothing.
        # trigger_entries items are mutated in place, so re-filtering resolved
        # ids reflects the latest state of everything processed so far.
        remaining = [e for e in trigger_entries if e['id'] not in metadata_by_id]
        save_trigger_entries(remaining, trigger_path)

    still_pending = [e for e in trigger_entries if e['id'] not in metadata_by_id]

    print("\n" + "=" * 80)
    print(f"✓ Resolved: {resolved_count} instance(s) → merged into {output_path}")
    print(f"⏳ Still pending: {len(still_pending)} instance(s) in {trigger_path}")


if __name__ == '__main__':
    main()
