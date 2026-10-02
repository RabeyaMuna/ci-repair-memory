#!/bin/bash
# Cleanup corrupted repository directories

REPO_DIR="/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/baselines/repo_cloned"

echo "Cleaning up corrupted repositories in: $REPO_DIR"
echo ""

# Find directories without .git subdirectory
for dir in "$REPO_DIR"/*; do
    if [ -d "$dir" ]; then
        repo_name=$(basename "$dir")
        if [ ! -d "$dir/.git" ]; then
            echo "Found corrupted repo: $repo_name (no .git directory)"
            echo "  Removing: $dir"
            rm -rf "$dir"
            echo "  ✓ Removed"
        else
            echo "OK: $repo_name"
        fi
    fi
done

echo ""
echo "Cleanup complete!"
