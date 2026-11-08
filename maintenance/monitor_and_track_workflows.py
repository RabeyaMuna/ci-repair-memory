#!/usr/bin/env python3
"""
Real-time workflow monitor for CI-REPAIR-BENCH.
Tracks workflows from push to completion, calculating timing metrics:
1. Execution duration (workflow start to completion) - from GitHub API
2. Queueing delay (submission to start) - from GitHub API

Generates Table 9 format with three categories:
- Failing Commits (no patch applied)
- Ground-Truth Repairs (patch applied, success)
- Candidate Patches (patch applied, failure)

Usage:
  # Start monitoring
  python3 monitor_and_track_workflows.py --monitor results/jobs_results_diff.jsonl --loop

  # Generate report
  python3 monitor_and_track_workflows.py --report
"""

import json
import requests
import time
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Optional
import os
import sys

# Configuration
CHECKPOINT_FILE = "monitoring_checkpoint.json"
POLLING_INTERVAL_SEC = 15 * 60  # 15 minutes
MAX_CYCLES = 25
MAX_MONITORING_MIN = 15 * MAX_CYCLES  # 375 minutes
RATE_LIMIT_DELAY = 0.5  # seconds between API calls

class WorkflowMonitor:
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token
        self.tracking = {}
        self.completed = []
        self.failed_fetch = []
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load previous monitoring state"""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r') as f:
                    data = json.load(f)
                    self.tracking = data.get('tracking', {})
                    self.completed = data.get('completed', [])
                    self.failed_fetch = data.get('failed', [])
                print(f"✓ Resumed: {len(self.tracking)} tracking, {len(self.completed)} completed")
            except Exception as e:
                print(f"⚠ Could not load checkpoint: {e}")

    def save_checkpoint(self):
        """Save current monitoring state"""
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump({
                    'tracking': self.tracking,
                    'completed': self.completed,
                    'failed': self.failed_fetch,
                    'last_update': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"⚠ Could not save checkpoint: {e}")

    def parse_github_url(self, url: str) -> Optional[tuple]:
        """Extract owner, repo, run_id from GitHub Actions URL"""
        parts = url.split('/')
        if 'runs' in parts and len(parts) >= 7:
            owner = parts[3]
            repo = parts[4]
            run_id = parts[parts.index('runs') + 1]
            return (owner, repo, run_id)
        return None

    def fetch_workflow_timing(self, owner: str, repo: str, run_id: str) -> Optional[Dict]:
        """Fetch workflow timing data from GitHub Actions API"""
        headers = {'Accept': 'application/vnd.github.v3+json'}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'

        url = f'https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}'

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
                updated_at = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))

                run_started_at = data.get('run_started_at')
                if run_started_at:
                    started_at = datetime.fromisoformat(run_started_at.replace('Z', '+00:00'))
                    queue_delay_min = (started_at - created_at).total_seconds() / 60
                else:
                    started_at = created_at
                    queue_delay_min = 0

                execution_duration_min = (updated_at - started_at).total_seconds() / 60

                return {
                    'created_at': created_at.isoformat(),
                    'started_at': started_at.isoformat(),
                    'updated_at': updated_at.isoformat(),
                    'execution_duration_min': execution_duration_min,
                    'queue_delay_min': queue_delay_min,
                    'status': data.get('status'),
                    'conclusion': data.get('conclusion')
                }

            elif response.status_code in [403, 429]:
                print(f"⚠ Rate limited ({response.status_code})")
                return None
            else:
                return None

        except Exception as e:
            return None

    def start_tracking_from_file(self, jobs_file: str):
        """Load workflows from jobs file and start tracking"""
        if not os.path.exists(jobs_file):
            print(f"❌ File not found: {jobs_file}")
            return

        print(f"📂 Loading workflows from {jobs_file}...")

        records = []
        with open(jobs_file, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        new_count = 0
        for record in records:
            url = record.get('url', '')
            parsed = self.parse_github_url(url)

            if not parsed:
                continue

            owner, repo, run_id = parsed

            if run_id in self.tracking or any(c['run_id'] == run_id for c in self.completed):
                continue

            pushed_at_str = record.get('pushed_at')
            if pushed_at_str:
                push_time = datetime.fromisoformat(pushed_at_str.replace('Z', '+00:00'))
            else:
                push_time = datetime.now(timezone.utc)

            self.tracking[run_id] = {
                'run_id': run_id,
                'owner': owner,
                'repo': repo,
                'url': url,
                'repo_name': record.get('repo_name'),
                'commit': record.get('commit'),
                'patch_applied': record.get('patch_applied'),
                'workflow': record.get('workflow'),
                'pushed_at': push_time.isoformat(),
                'first_check_at': datetime.now(timezone.utc).isoformat(),
                'poll_count': 0,
                'status': 'tracking'
            }
            new_count += 1

        self.save_checkpoint()
        print(f"✓ Now tracking {new_count} new workflows ({len(self.tracking)} total)")

    def poll_workflows(self):
        """Poll all tracked workflows"""
        if not self.tracking:
            print("ℹ No workflows currently being tracked")
            return

        print(f"\n🔄 Polling {len(self.tracking)} workflows...")
        completed_count = 0

        for run_id in list(self.tracking.keys()):
            track_info = self.tracking[run_id]
            timing = self.fetch_workflow_timing(track_info['owner'], track_info['repo'], run_id)

            if timing:
                track_info['poll_count'] += 1
                track_info['last_status'] = timing['status']
                track_info['last_conclusion'] = timing.get('conclusion')

                if timing['status'] == 'completed':
                    self.complete_workflow(run_id, timing)
                    completed_count += 1
                elif track_info['poll_count'] >= MAX_CYCLES:
                    self.timeout_workflow(run_id)
                    completed_count += 1

            time.sleep(RATE_LIMIT_DELAY)

        self.save_checkpoint()
        print(f"✓ Completed {completed_count} workflows | Still tracking: {len(self.tracking)} | Total: {len(self.completed)}")

    def complete_workflow(self, run_id: str, timing: Dict):
        """Mark workflow as completed"""
        if run_id not in self.tracking:
            return

        track_info = self.tracking[run_id]
        push_time = datetime.fromisoformat(track_info['pushed_at'])
        detection_time = datetime.now(timezone.utc)
        monitoring_delay_min = (detection_time - push_time).total_seconds() / 60

        completed_workflow = {
            'run_id': run_id,
            'owner': track_info['owner'],
            'repo': track_info['repo'],
            'repo_name': track_info['repo_name'],
            'url': track_info['url'],
            'commit': track_info['commit'],
            'patch_applied': track_info['patch_applied'],
            'workflow': track_info['workflow'],
            'execution_duration_min': timing['execution_duration_min'],
            'queue_delay_min': timing['queue_delay_min'],
            'monitoring_delay_min': monitoring_delay_min,
            'pushed_at': track_info['pushed_at'],
            'created_at': timing['created_at'],
            'started_at': timing['started_at'],
            'updated_at': timing['updated_at'],
            'detected_at': detection_time.isoformat(),
            'status': timing['status'],
            'conclusion': timing.get('conclusion'),
            'poll_count': track_info['poll_count'],
            'timed_out': False
        }

        self.completed.append(completed_workflow)
        del self.tracking[run_id]

        status_icon = "✅" if timing.get('conclusion') == 'success' else "❌"
        print(f"{status_icon} {track_info['repo_name']}: exec={timing['execution_duration_min']:.1f}min, queue={timing['queue_delay_min']:.1f}min")

    def timeout_workflow(self, run_id: str):
        """Mark workflow as timed out"""
        if run_id not in self.tracking:
            return

        track_info = self.tracking[run_id]
        self.completed.append({**track_info, 'timed_out': True, 'poll_count': track_info['poll_count']})
        del self.tracking[run_id]

        print(f"⏱ TIMEOUT: {track_info['repo_name']} (after {track_info['poll_count']} polls)")

    def generate_final_report(self):
        """Generate Table 9 format report"""
        if not self.completed:
            print("ℹ No completed workflows to report")
            return

        # Three categories:
        # 1. Failing commits (no patch)
        # 2. Ground-truth repairs (patch + success)
        # 3. Candidate patches (patch + failure)

        failing_exec = []
        failing_queue = []
        failing_timeouts = 0

        gt_exec = []
        gt_queue = []
        gt_timeouts = 0

        cand_exec = []
        cand_queue = []
        cand_timeouts = 0

        for wf in self.completed:
            if wf.get('timed_out'):
                patch = wf.get('patch_applied', False)
                if not patch:
                    failing_timeouts += 1
                elif wf.get('conclusion') == 'success':
                    gt_timeouts += 1
                else:
                    cand_timeouts += 1
                continue

            exec_dur = wf['execution_duration_min']
            queue_del = wf['queue_delay_min']

            patch = wf.get('patch_applied', False)
            conclusion = wf.get('conclusion', '')

            if not patch:
                failing_exec.append(exec_dur)
                failing_queue.append(queue_del)
            elif conclusion == 'success':
                gt_exec.append(exec_dur)
                gt_queue.append(queue_del)
            else:
                cand_exec.append(exec_dur)
                cand_queue.append(queue_del)

        # Calculate mean and max only
        def calc(vals):
            if not vals:
                return {'mean': 0, 'max': 0, 'count': 0}
            return {'mean': statistics.mean(vals), 'max': max(vals), 'count': len(vals)}

        f_exec, f_queue = calc(failing_exec), calc(failing_queue)
        g_exec, g_queue = calc(gt_exec), calc(gt_queue)
        c_exec, c_queue = calc(cand_exec), calc(cand_queue)

        print("\n" + "="*80)
        print("TABLE 9: CI WORKFLOW EXECUTION DURATION AND QUEUEING DELAY")
        print("="*80)
        print(f"\nTotal: {len(self.completed)} workflows")
        print(f"  Failing commits: {f_exec['count']} (timeouts: {failing_timeouts})")
        print(f"  Ground-truth repairs: {g_exec['count']} (timeouts: {gt_timeouts})")
        print(f"  Candidate patches: {c_exec['count']} (timeouts: {cand_timeouts})")

        print("\n" + "="*80)
        print("MARKDOWN TABLE:")
        print("="*80)
        print()
        print("|                      | Failing Commits |            | Ground-Truth Repairs |            | Candidate Patches |            |")
        print("|----------------------|-----------------|------------|----------------------|------------|-------------------|------------|")
        print("|                      | Mean (min)      | Max (min)  | Mean (min)           | Max (min)  | Mean (min)        | Max (min)  |")
        print(f"| Execution duration   | {f_exec['mean']:15.1f} | {f_exec['max']:10.1f} | {g_exec['mean']:20.1f} | {g_exec['max']:10.1f} | {c_exec['mean']:17.1f} | {c_exec['max']:10.1f} |")
        print(f"| Queueing delay       | {f_queue['mean']:15.1f} | {f_queue['max']:10.1f} | {g_queue['mean']:20.1f} | {g_queue['max']:10.1f} | {c_queue['mean']:17.1f} | {c_queue['max']:10.1f} |")
        print(f"| Timeouts             | {failing_timeouts:15} | —          | {gt_timeouts:20} | —          | {cand_timeouts:17} | —          |")
        print()

        print("\n" + "="*80)
        print("LATEX TABLE:")
        print("="*80)
        print()
        print("\\begin{table}[t]")
        print("\\centering")
        print("\\caption{CI workflow execution duration and queueing delay across benchmark")
        print("instances. Execution duration is measured from workflow start to completion")
        print("using GitHub Actions timestamps. Queueing delay is measured from workflow")
        print("submission to execution start. Timeout instances exceeded the 375-minute")
        print("monitoring window and are excluded.}")
        print("\\label{tab:workflow_execution}")
        print("\\begin{tabular}{lcccccc}")
        print("\\toprule")
        print("& \\multicolumn{2}{c}{\\textbf{Failing Commits}} & \\multicolumn{2}{c}{\\textbf{Ground-Truth Repairs}} & \\multicolumn{2}{c}{\\textbf{Candidate Patches}} \\\\")
        print("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}")
        print("& Mean (min) & Max (min) & Mean (min) & Max (min) & Mean (min) & Max (min) \\\\")
        print("\\midrule")
        print(f"Execution duration & {f_exec['mean']:.1f} & {f_exec['max']:.1f} & {g_exec['mean']:.1f} & {g_exec['max']:.1f} & {c_exec['mean']:.1f} & {c_exec['max']:.1f} \\\\")
        print(f"Queueing delay     & {f_queue['mean']:.1f} & {f_queue['max']:.1f} & {g_queue['mean']:.1f} & {g_queue['max']:.1f} & {c_queue['mean']:.1f} & {c_queue['max']:.1f} \\\\")
        print(f"Timeouts           & {failing_timeouts} & — & {gt_timeouts} & — & {cand_timeouts} & — \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print()

        # Save
        with open("workflow_timing_table9.txt", 'w') as f:
            f.write("TABLE 9: CI WORKFLOW EXECUTION DURATION AND QUEUEING DELAY\n\n")
            f.write("EXECUTION DURATION (minutes):\n")
            f.write(f"  Failing Commits:      Mean {f_exec['mean']:6.1f}, Max {f_exec['max']:6.1f}\n")
            f.write(f"  Ground-Truth Repairs: Mean {g_exec['mean']:6.1f}, Max {g_exec['max']:6.1f}\n")
            f.write(f"  Candidate Patches:    Mean {c_exec['mean']:6.1f}, Max {c_exec['max']:6.1f}\n\n")
            f.write("QUEUEING DELAY (minutes):\n")
            f.write(f"  Failing Commits:      Mean {f_queue['mean']:6.1f}, Max {f_queue['max']:6.1f}\n")
            f.write(f"  Ground-Truth Repairs: Mean {g_queue['mean']:6.1f}, Max {g_queue['max']:6.1f}\n")
            f.write(f"  Candidate Patches:    Mean {c_queue['mean']:6.1f}, Max {c_queue['max']:6.1f}\n\n")
            f.write("TIMEOUTS:\n")
            f.write(f"  Failing Commits:      {failing_timeouts}\n")
            f.write(f"  Ground-Truth Repairs: {gt_timeouts}\n")
            f.write(f"  Candidate Patches:    {cand_timeouts}\n")

        print("✓ Report saved to: workflow_timing_table9.txt")
        print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Monitor GitHub Actions workflows')
    parser.add_argument('--monitor', help='Load workflows from jobs file')
    parser.add_argument('--poll', action='store_true', help='Poll once')
    parser.add_argument('--loop', action='store_true', help='Continuous polling')
    parser.add_argument('--report', action='store_true', help='Generate Table 9 report')

    args = parser.parse_args()

    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("⚠ WARNING: GITHUB_TOKEN not set. Will be rate-limited.\n")

    monitor = WorkflowMonitor(github_token)

    if args.report:
        monitor.generate_final_report()
        return

    if args.monitor:
        monitor.start_tracking_from_file(args.monitor)

    if args.poll:
        monitor.poll_workflows()
    elif args.loop:
        print(f"\n🔄 Continuous monitoring (every {POLLING_INTERVAL_SEC/60:.0f} minutes)")
        print("   Press Ctrl+C to stop\n")

        try:
            while monitor.tracking:
                monitor.poll_workflows()
                if not monitor.tracking:
                    print("\n✓ All workflows completed!")
                    break
                print(f"\n💤 Sleeping {POLLING_INTERVAL_SEC/60:.0f} minutes...")
                time.sleep(POLLING_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("\n\n⏸ Stopped. Progress saved.")
            monitor.save_checkpoint()
    else:
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} --monitor results/jobs_results_diff.jsonl --loop")
        print(f"  python3 {sys.argv[0]} --report")


if __name__ == '__main__':
    main()
