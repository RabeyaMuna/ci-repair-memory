#!/bin/bash
# Fix all repos with Actions blocked

# Read repos from jobs_awaiting_diff.jsonl
if [ ! -f "results/jobs_awaiting_diff.jsonl" ]; then
    echo "Error: results/jobs_awaiting_diff.jsonl not found"
    exit 1
fi

echo "Finding repos with 'repository invocation blocked' errors..."
echo ""

# Extract unique repo names from awaiting jobs
repos=$(python3 -c "
import json
repos = set()
with open('results/jobs_awaiting_diff.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            try:
                job = json.loads(line)
                repo = job.get('repo', '')
                if repo:
                    repos.add(repo)
            except:
                pass
for repo in sorted(repos):
    print(repo)
")

if [ -z "$repos" ]; then
    echo "No repos found in jobs_awaiting_diff.jsonl"
    exit 0
fi

echo "Found repos to fix:"
echo "$repos"
echo ""
echo "Enabling Actions for each repo..."
echo "========================================"

for repo in $repos; do
    echo ""
    echo "Fixing: Muna4029/$repo"
    
    gh api -X PUT "/repos/Muna4029/${repo}/actions/permissions" \
        -f enabled=true \
        -f allowed_actions=all 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Enabled Actions for $repo"
    else
        echo "  ✗ Failed to enable Actions for $repo"
    fi
done

echo ""
echo "========================================"
echo "Done! Retry the benchmark jobs now."
