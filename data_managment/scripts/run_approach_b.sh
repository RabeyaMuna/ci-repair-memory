#!/bin/bash
# Quick Start Script for Approach B
# Complete data collection: Fetch OR Trigger for ALL commits

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════"
echo "  CI Benchmark - Approach B: Complete Data Collection"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if we're in the right directory
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: Must run from data_managment directory"
    exit 1
fi

# Check for GitHub token
if [ -z "$GH_TOKEN" ] && [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Error: No GitHub token found"
    echo "   Set GH_TOKEN in .env file"
    exit 1
fi

echo "✓ GitHub token found"
echo ""

# Step 1: Setup branches (if not done)
if [ ! -f "results/branches/benchmark_branches.json" ]; then
    echo "════════════════════════════════════════════════════════════════"
    echo "Step 1: Setting up benchmark branches"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    python setup_benchmark_branches.py
    echo ""
else
    echo "✓ Benchmark branches already setup"
    echo ""
fi

# Step 2: Run complete collection
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Running complete data collection (Approach B)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Strategy: Fetch if exists, Trigger if missing"
echo "Expected time: 10-20 hours"
echo ""
echo "Starting collection..."
echo ""

python fetch_and_trigger_metadata.py

# Step 3: Summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Collection Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check results
if [ -f "results/metadata/commit_job_metadata.json" ]; then
    TOTAL=$(jq 'length' results/metadata/commit_job_metadata.json)
    echo "✓ Metadata collected for $TOTAL issues"
fi

if [ -f "results/metadata/failed_jobs_overall.json" ]; then
    FAILED=$(jq 'length' results/metadata/failed_jobs_overall.json)
    echo "✓ Collected $FAILED failed job/step entries"
fi

if [ -f "results/metadata/missing_metadata_ids.json" ]; then
    MISSING=$(jq 'length' results/metadata/missing_metadata_ids.json)
    echo "⚠️  Still missing metadata for $MISSING issues"
fi

echo ""
echo "Results saved in: results/metadata/"
echo ""
echo "Next steps:"
echo "  1. Verify: python workflow_manager.py status"
echo "  2. Update dataset: python update_failed_logs.py"
echo "  3. Analyze: jq . results/metadata/failed_jobs_overall.json"
echo ""
echo "════════════════════════════════════════════════════════════════"
