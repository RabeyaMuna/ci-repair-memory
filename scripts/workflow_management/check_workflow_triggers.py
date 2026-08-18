#!/usr/bin/env python3
"""
Check which workflows in the dataset have branch restrictions
that will prevent them from triggering on custom branches.
"""

import pandas as pd
import yaml
from io import StringIO

# Load dataset
df = pd.read_parquet('dataset/lca_dataset.parquet')

problematic = []
ok = []

for idx, row in df.iterrows():
    id = row['id']
    repo = row['repo_name']
    workflow_content = row.get('workflow', '')

    if not workflow_content:
        continue

    try:
        # Parse YAML
        wf = yaml.safe_load(StringIO(workflow_content))

        if not wf or 'on' not in wf:
            continue

        on_config = wf['on']

        # Check if it's restricted
        restricted = False
        reason = ""

        if isinstance(on_config, dict) and 'push' in on_config:
            push_config = on_config['push']

            if isinstance(push_config, dict) and 'branches' in push_config:
                branches = push_config['branches']
                if branches and not ('**' in branches or '*' in ' '.join(map(str, branches))):
                    restricted = True
                    reason = f"push.branches = {branches}"

        elif isinstance(on_config, str):
            if on_config == 'push':
                restricted = False
                reason = "on: push (OK)"
        elif isinstance(on_config, list):
            if 'push' in on_config:
                restricted = False
                reason = "on: [push, ...] (OK)"

        if restricted:
            problematic.append({
                'id': id,
                'repo': repo,
                'workflow': row['workflow_path'],
                'reason': reason
            })
        else:
            ok.append({'id': id, 'repo': repo})

    except Exception as e:
        print(f"Error parsing ID {id}: {e}")

print("="*80)
print("WORKFLOWS WITH BRANCH RESTRICTIONS (Will Not Trigger!)")
print("="*80)
print(f"\nTotal problematic: {len(problematic)}")

if problematic:
    print("\nFirst 20:")
    for p in problematic[:20]:
        print(f"  ID {p['id']:4s} | {p['repo']:20s} | {p['reason']}")

    print(f"\n... and {len(problematic) - 20} more" if len(problematic) > 20 else "")

print(f"\n✓ OK workflows: {len(ok)}")
print("="*80)

# Save problematic IDs to file
if problematic:
    with open('results/problematic_workflow_ids.txt', 'w') as f:
        for p in problematic:
            f.write(f"{p['id']}\n")
    print(f"\nProblematic IDs saved to: results/problematic_workflow_ids.txt")
