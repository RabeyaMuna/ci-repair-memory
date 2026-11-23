#!/bin/bash
# Fix "repository invocation blocked" error

REPO_NAME="$1"

if [ -z "$REPO_NAME" ]; then
    echo "Usage: ./fix_actions_blocked.sh <repo-name>"
    echo "Example: ./fix_actions_blocked.sh DocsGPT"
    exit 1
fi

REPO_OWNER="Muna4029"
REPO_FULL="${REPO_OWNER}/${REPO_NAME}"

echo "Fixing Actions for: $REPO_FULL"
echo "========================================"

# Enable GitHub Actions for the repository
gh api -X PUT "/repos/${REPO_FULL}/actions/permissions" \
    -f enabled=true \
    -f allowed_actions=all

echo "✓ Enabled GitHub Actions"

# Set workflow permissions to read and write
gh api -X PUT "/repos/${REPO_FULL}/actions/permissions/workflow" \
    -f default_workflow_permissions=write \
    -f can_approve_pull_request_reviews=false

echo "✓ Set workflow permissions to write"

# Enable Actions for all workflows
gh api -X PUT "/repos/${REPO_FULL}/actions/permissions/selected-actions" \
    -f github_owned_allowed=true \
    -f verified_allowed=true \
    -f patterns_allowed='["*"]'

echo "✓ Enabled all actions"

echo ""
echo "========================================"
echo "Actions should now be enabled for $REPO_FULL"
echo "Try pushing again to trigger workflows"
