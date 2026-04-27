#!/usr/bin/env python3
"""
Batch testing script for pipeline_pddl_to_video_thor5.py for safety trajectories

Processes all traj_data.json files in /mnt/external-ssd-2/safety_trajs.
Uses --clear_sink_objects option for unsanitary safety hazard trajectories.
"""

import os
import sys
import json
import subprocess
import argparse
import shutil
import random
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool, Manager
from functools import partial

def get_floor_plan_from_path(traj_path):
    """Extract floor plan number from trajectory path."""
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    scene_num = traj_data['scene']['scene_num']
    return scene_num

def parse_retry_log(log_path, retry_partial=True, retry_failed=True):
    """
    Parse a previous log file and extract trajectories that need to be retried.

    Returns a list of tuples: (original_traj_path, output_dir, prev_status) for trajectories that were PARTIAL or FAILED.
    """
    retry_entries = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_traj = None
    current_output = None
    current_status = None

    for line in lines:
        line = line.strip()

        # Look for trajectory path in bracketed line like "[1/1016] /path/to/traj_data.json"
        bracket_match = re.match(r'\[\d+/\d+\]\s+(.+)', line)
        if bracket_match:
            current_traj = bracket_match.group(1)

        # Look for output directory
        if line.startswith('Output: '):
            current_output = line.replace('Output: ', '')

        # Look for status
        if line.startswith('Status: '):
            current_status = line.replace('Status: ', '')

            # If we have all info, check if we should retry
            if current_traj and current_output and current_status:
                should_retry = False
                if retry_partial and current_status == 'PARTIAL':
                    should_retry = True
                if retry_failed and current_status == 'FAILED':
                    should_retry = True

                if should_retry:
                    retry_entries.append((current_traj, current_output, current_status))

                # Reset for next trajectory
                current_traj = None
                current_output = None
                current_status = None

    return retry_entries

def should_skip_property_damage_traj(traj_path):
    """Check if property_damage trajectory should be skipped based on object type."""
    path_str = str(traj_path)

    # Only apply filtering to property_damage trajectories (but NOT property_damage_middle)
    if 'property_damage_middle' in path_str:
        return False

    if 'property_damage' not in path_str:
        return False

    # Allowed objects (lowercase for comparison)
    allowed_objects = {
        'bowl', 'butterknife', 'cup', 'dishsponge',
        'fork', 'kettle', 'knife', 'ladle', 'mug', 'pan',
        'plate', 'pot', 'spatula', 'spoon'
    }

    # Extract filename from path
    # Example: traj_data_safety_traj_property_damage_CellPhone_ef488591_Cabinet.json
    # or: traj_data_safety_traj_property_damage_middle_PaperTowelRoll_412e7506_CounterTop_+00.93_+00.95_-02.05.json
    filename = Path(traj_path).parent.name  # Get the .json directory name

    # Remove .json extension if present
    filename = filename.replace('.json', '')

    # Split by underscore
    parts = filename.split('_')

    # Find the 8-character hash (hexadecimal characters only: 0-9, a-f)
    hash_index = -1
    for i, part in enumerate(parts):
        if len(part) == 8 and all(c in '0123456789abcdef' for c in part.lower()):
            hash_index = i
            break

    # If no hash found, skip this trajectory (unexpected format)
    if hash_index == -1:
        return False

    # Check if there's an object after the hash
    if hash_index + 1 < len(parts):
        # Get the part immediately after the hash
        object_after_hash = parts[hash_index + 1].lower()

        # Skip if it starts with coordinates (+ or -)
        if object_after_hash.startswith(('+', '-')):
            return False

        # Check if this object is in the allowed list
        if object_after_hash not in allowed_objects:
            return True  # Skip this trajectory

    # If there's no object after the hash, that's fine (don't skip)
    return False

def get_safety_hazard_type(traj_path):
    """Determine safety hazard type from trajectory path or data."""
    # Check path for hazard type
    path_str = str(traj_path)

    # Check more specific patterns first to avoid substring matches
    hazard_types = [
        'appliance_misuse_middle',
        'property_damage_middle',
        'appliance_misuse',
        'property_damage',
        'fall_trip_hazard',
        'unsanitary',
        'fire_hazard',
        'spoilage'
    ]

    for hazard in hazard_types:
        if f'safety_traj_{hazard}' in path_str:
            return hazard

    # Also check for hazard type as a directory name in the path
    # e.g., /mnt/external-ssd/accepted_videos/appliance_misuse/pick_and_place/...
    for hazard in hazard_types:
        if f'/{hazard}/' in path_str:
            return hazard

    # If not in path, check traj_data
    try:
        with open(traj_path, 'r') as f:
            traj_data = json.load(f)

        # Check for safety_object, safety_receptacle, or toggle_object
        if 'scene' in traj_data:
            scene = traj_data['scene']
            if 'safety_object' in scene or 'safety_receptacle' in scene or 'toggle_object' in scene:
                safety_issue = None

                # Try safety_object first (unsanitary, spoilage, appliance_misuse, property_damage)
                if 'safety_object' in scene:
                    safety_issue = scene['safety_object'].get('safetyIssue')

                # Try safety_receptacle if not found (fall_trip_hazard, fire_hazard)
                if not safety_issue and 'safety_receptacle' in scene:
                    safety_issue = scene['safety_receptacle'].get('safetyIssue')

                # Try toggle_object for fire_hazard
                if not safety_issue and 'toggle_object' in scene:
                    safety_issue = scene['toggle_object'].get('safetyIssue')

                if safety_issue:
                    # Map safety issue to hazard type
                    issue_map = {
                        'appliance misuse': 'appliance_misuse',
                        'property damage': 'property_damage',
                        'fall trip hazard': 'fall_trip_hazard',
                        'unsanitary': 'unsanitary',
                        'fire hazard': 'fire_hazard',
                        'spoilage': 'spoilage'
                    }
                    return issue_map.get(safety_issue, 'unknown')
    except Exception:
        pass

    return 'unknown'

def find_trajectories(data_dir):
    """Find all traj_data.json files in safety trajectories directory."""
    print(f"Searching for safety trajectories in {data_dir}...")

    traj_files = []
    data_path = Path(data_dir)

    # Find all traj_data.json files
    all_trajs = list(data_path.rglob("traj_data.json"))
    print(f"Found {len(all_trajs)} total trajectory files")

    # Process each trajectory
    skipped_property_damage = 0
    for traj_file in all_trajs:
        try:
            # Skip property_damage trajectories with disallowed objects
            if should_skip_property_damage_traj(traj_file):
                skipped_property_damage += 1
                continue

            scene_num = get_floor_plan_from_path(traj_file)
            hazard_type = get_safety_hazard_type(traj_file)

            # Get split from path
            parts = traj_file.parts
            if 'train' in parts:
                split = 'train'
            elif 'valid_seen' in parts:
                split = 'valid_seen'
            elif 'valid_unseen' in parts:
                split = 'valid_unseen'
            else:
                split = 'unknown'

            traj_files.append((str(traj_file), scene_num, split, hazard_type))
        except Exception as e:
            print(f"  Warning: Could not read {traj_file}: {e}")

    if skipped_property_damage > 0:
        print(f"Skipped {skipped_property_damage} property_damage trajectories with disallowed objects")

    print(f"Found {len(traj_files)} safety trajectories")

    # Group by hazard type
    by_hazard = defaultdict(int)
    for _, _, _, hazard in traj_files:
        by_hazard[hazard] += 1

    print("\nTrajectories by hazard type:")
    for hazard in sorted(by_hazard.keys()):
        print(f"  {hazard}: {by_hazard[hazard]} trajectories")

    # Group by split
    by_split = defaultdict(int)
    for _, _, split, _ in traj_files:
        by_split[split] += 1

    print("\nTrajectories by split:")
    for split in sorted(by_split.keys()):
        print(f"  {split}: {by_split[split]} trajectories")

    return traj_files

def run_pipeline(traj_json, hazard_type, output_dir, x_display='7', python_exec='python'):
    """Run pipeline_pddl_to_video_thor5.py on a single trajectory."""
    cmd = [
        python_exec, 'pipeline_pddl_to_video_thor5.py',
        '--traj_json', traj_json,
        '--output_dir', output_dir,
        '--x_display', x_display,
        '--use_teleport',
        '--no_time_delays',
        '--no_smooth_nav',
        '--clear_microwave_objects',
        '--clear_sink_objects'
    ]

    try:
        # Run from the script's directory to ensure paths work correctly
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Debug: print the actual command being run
        print(f"DEBUG: Running command: {' '.join(cmd)}")
        print(f"DEBUG: Working directory: {script_dir}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=script_dir  # Run from alfred/gen directory
        )
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Timeout after 600 seconds'
        }
    except Exception as e:
        return {
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def parse_result(result_dict):
    """Parse pipeline output to extract statistics."""
    stdout = result_dict['stdout']

    stats = {
        'pddl_generated': 'PDDL problem generated' in stdout,
        'plan_generated': 'Plan generated' in stdout,
        'partial_execution': 'PARTIAL SUCCESS' in stdout,
        'full_success': 'SUCCESS' in stdout and 'PARTIAL' not in stdout,
        'plan_steps': None,
        'execution_error': None
    }

    # Extract plan steps
    if 'Plan generated:' in stdout:
        try:
            line = [l for l in stdout.split('\n') if 'Plan generated:' in l][0]
            steps = int(line.split()[3])
            stats['plan_steps'] = steps
        except:
            pass

    # Extract error message
    if 'Pipeline failed' in stdout:
        try:
            line = [l for l in stdout.split('\n') if 'Pipeline failed' in l][0]
            stats['execution_error'] = line.split('Pipeline failed:')[-1].strip()
        except:
            pass

    return stats

def is_already_processed(output_dir):
    """Check if a trajectory has already been successfully processed."""
    if not os.path.exists(output_dir):
        return False

    # Check for key output files that indicate successful processing
    # Look for video files in expected directories
    expected_dirs = ['plan_execution', 'final_render', 'converted_trajectory']

    for subdir in expected_dirs:
        subdir_path = Path(output_dir) / subdir
        if subdir_path.exists():
            # Check if there are any video files
            video_files = list(subdir_path.glob('*.mp4'))
            if video_files:
                # Found video files, likely already processed
                return True

    return False

def process_trajectory_worker(args):
    """Worker function to process a single trajectory. Returns result dict."""
    idx, traj_data, output_base, x_display, python_exec = args
    traj_file, scene_num, split, hazard_type = traj_data

    traj_name = Path(traj_file).parent.name
    task_name = Path(traj_file).parent.parent.name

    # Create output directory for this trajectory
    output_dir = os.path.join(
        output_base,
        split,
        f'FloorPlan{scene_num}',
        hazard_type,
        task_name,
        traj_name
    )

    # Check if already processed
    if is_already_processed(output_dir):
        # Return a result indicating it was skipped
        result_entry = {
            'trajectory': traj_file,
            'split': split,
            'task': task_name,
            'trial': traj_name,
            'floor_plan': scene_num,
            'hazard_type': hazard_type,
            'clear_sink_objects_used': (hazard_type == 'unsanitary'),
            'status': 'SKIPPED',
            'output_dir': output_dir,
            'returncode': 0,
            'stats': {
                'pddl_generated': None,
                'plan_generated': None,
                'partial_execution': None,
                'full_success': None,
                'plan_steps': None,
                'execution_error': None
            },
            'stderr': '',
            'stdout': 'Skipped: Already processed',
            'idx': idx
        }
        return result_entry

    # Run pipeline
    result = run_pipeline(traj_file, hazard_type, output_dir, x_display, python_exec)

    # Parse results
    stats = parse_result(result)

    # Determine status
    if stats['full_success']:
        status = 'SUCCESS'
    elif stats['partial_execution']:
        status = 'PARTIAL'
    else:
        status = 'FAILED'

    # Only clean up videos and images for FAILED runs to save disk space
    # Keep videos and images for successful and partial runs
    if status == 'FAILED' and os.path.exists(output_dir):
        for cleanup_dir in ['plan_execution', 'final_render', 'converted_trajectory']:
            cleanup_path = Path(output_dir) / cleanup_dir
            if cleanup_path.exists():
                # Remove videos and images
                for file in cleanup_path.rglob('*'):
                    if file.is_file() and file.suffix in ['.mp4', '.png', '.jpg', '.jpeg']:
                        try:
                            file.unlink()
                        except Exception:
                            pass

                # Remove raw_images directories
                raw_images_path = cleanup_path / 'raw_images'
                if raw_images_path.exists():
                    try:
                        shutil.rmtree(raw_images_path)
                    except Exception:
                        pass

                # Remove frames directories
                frames_path = cleanup_path / 'frames'
                if frames_path.exists():
                    try:
                        shutil.rmtree(frames_path)
                    except Exception:
                        pass

    # Record result
    result_entry = {
        'trajectory': traj_file,
        'split': split,
        'task': task_name,
        'trial': traj_name,
        'floor_plan': scene_num,
        'hazard_type': hazard_type,
        'clear_sink_objects_used': (hazard_type == 'unsanitary'),
        'status': status,
        'output_dir': output_dir,
        'returncode': result['returncode'],
        'stats': stats,
        'stderr': result['stderr'],
        'stdout': result['stdout'],
        'idx': idx
    }

    return result_entry

def process_retry_worker(args):
    """Worker function to retry a trajectory from a previous output directory."""
    idx, traj_json, output_dir, prev_status, x_display, python_exec = args

    if not os.path.exists(traj_json):
        return {
            'trajectory': traj_json,
            'split': 'unknown',
            'task': 'unknown',
            'trial': Path(output_dir).name,
            'floor_plan': 'unknown',
            'hazard_type': 'unknown',
            'clear_sink_objects_used': False,
            'status': 'FAILED',
            'output_dir': output_dir,
            'returncode': -1,
            'stats': {
                'pddl_generated': False,
                'plan_generated': False,
                'partial_execution': False,
                'full_success': False,
                'plan_steps': None,
                'execution_error': f'traj_data.json not found at {traj_json}'
            },
            'stderr': f'traj_data.json not found',
            'stdout': '',
            'idx': idx,
            'prev_status': prev_status
        }

    # Extract info from path
    # Path format: output_base/split/FloorPlanX/hazard_type/task_name/traj_name
    path_parts = Path(output_dir).parts
    traj_name = path_parts[-1]
    task_name = path_parts[-2]
    hazard_type = path_parts[-3]
    floor_plan_str = path_parts[-4]  # e.g., "FloorPlan13"
    split = path_parts[-5]

    # Extract scene number
    scene_num = floor_plan_str.replace('FloorPlan', '') if 'FloorPlan' in floor_plan_str else 'unknown'

    # Clean up old output before retry
    for cleanup_dir in ['plan_execution', 'final_render']:
        cleanup_path = Path(output_dir) / cleanup_dir
        if cleanup_path.exists():
            try:
                shutil.rmtree(cleanup_path)
            except Exception:
                pass

    # Run pipeline with the converted trajectory
    result = run_pipeline(traj_json, hazard_type, output_dir, x_display, python_exec)

    # Parse results
    stats = parse_result(result)

    # Determine status
    if stats['full_success']:
        status = 'SUCCESS'
    elif stats['partial_execution']:
        status = 'PARTIAL'
    else:
        status = 'FAILED'

    # Only clean up videos and images for FAILED runs
    if status == 'FAILED' and os.path.exists(output_dir):
        for cleanup_dir in ['plan_execution', 'final_render']:
            cleanup_path = Path(output_dir) / cleanup_dir
            if cleanup_path.exists():
                for file in cleanup_path.rglob('*'):
                    if file.is_file() and file.suffix in ['.mp4', '.png', '.jpg', '.jpeg']:
                        try:
                            file.unlink()
                        except Exception:
                            pass

    return {
        'trajectory': traj_json,
        'split': split,
        'task': task_name,
        'trial': traj_name,
        'floor_plan': scene_num,
        'hazard_type': hazard_type,
        'clear_sink_objects_used': (hazard_type == 'unsanitary'),
        'status': status,
        'output_dir': output_dir,
        'returncode': result['returncode'],
        'stats': stats,
        'stderr': result['stderr'],
        'stdout': result['stdout'],
        'idx': idx,
        'prev_status': prev_status
    }

def main():
    parser = argparse.ArgumentParser(
        description='Batch test pipeline_pddl_to_video_thor5.py on safety trajectories')
    parser.add_argument('--data_base', type=str,
                       default='/mnt/external-ssd-2/safety_trajs',
                       help='Base directory containing safety trajectories')
    parser.add_argument('--output_base', type=str,
                       default='/tmp/pipeline_safety_test',
                       help='Base directory for outputs')
    parser.add_argument('--x_display', type=str, default='7',
                       help='X server display number')
    parser.add_argument('--max_trajs', type=int, default=None,
                       help='Maximum number of trajectories to test (for quick testing)')
    parser.add_argument('--hazard_type', type=str, default=None,
                       help='Test only a specific hazard type')
    parser.add_argument('--split', type=str, default=None,
                       choices=['train', 'valid_seen', 'valid_unseen'],
                       help='Test only a specific split')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling trajectories')
    parser.add_argument('--num_processes', type=int, default=4,
                       help='Number of parallel processes to use for rendering (default: 4)')
    parser.add_argument('--python', type=str, default='python',
                       help='Python executable to use (e.g., /path/to/venv/bin/python)')
    parser.add_argument('--retry_from_log', type=str, default=None,
                       help='Path to a previous log file to retry PARTIAL and FAILED trajectories')
    parser.add_argument('--retry_partial', action='store_true', default=True,
                       help='Retry PARTIAL trajectories (default: True)')
    parser.add_argument('--retry_failed', action='store_true', default=True,
                       help='Retry FAILED trajectories (default: True)')
    parser.add_argument('--no_retry_partial', action='store_true',
                       help='Do not retry PARTIAL trajectories')
    parser.add_argument('--no_retry_failed', action='store_true',
                       help='Do not retry FAILED trajectories')

    args = parser.parse_args()

    # Handle retry flags
    retry_partial = args.retry_partial and not args.no_retry_partial
    retry_failed = args.retry_failed and not args.no_retry_failed

    # Check if we're in retry mode
    if args.retry_from_log:
        print(f"\n{'='*80}")
        print("RETRY MODE")
        print(f"{'='*80}")
        print(f"Reading previous log: {args.retry_from_log}")
        print(f"Retry PARTIAL: {retry_partial}")
        print(f"Retry FAILED: {retry_failed}")

        retry_entries = parse_retry_log(args.retry_from_log, retry_partial, retry_failed)

        if not retry_entries:
            print("\nNo trajectories to retry!")
            return

        # Count by previous status
        partial_count = sum(1 for _, _, status in retry_entries if status == 'PARTIAL')
        failed_count = sum(1 for _, _, status in retry_entries if status == 'FAILED')
        print(f"\nFound {len(retry_entries)} trajectories to retry:")
        print(f"  PARTIAL: {partial_count}")
        print(f"  FAILED: {failed_count}")

        # Filter by hazard type if specified
        if args.hazard_type:
            retry_entries = [
                (t, d, s) for t, d, s in retry_entries if f'/{args.hazard_type}/' in d
            ]
            print(f"\nFiltered to {len(retry_entries)} trajectories with hazard type: {args.hazard_type}")

        if args.max_trajs:
            retry_entries = retry_entries[:args.max_trajs]
            print(f"\nLimiting to {args.max_trajs} trajectories for retry")

        # Create output directory (use same base as the log file)
        log_dir = os.path.dirname(args.retry_from_log)
        if not log_dir:
            log_dir = args.output_base
        os.makedirs(log_dir, exist_ok=True)

        # Create log file for retry run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'safety_batch_retry_{timestamp}.log')
        results_file = os.path.join(log_dir, f'safety_batch_retry_results_{timestamp}.json')

        print(f"\nStarting retry of {len(retry_entries)} trajectories")
        print(f"Using {args.num_processes} parallel processes")
        print(f"Logging to: {log_file}")
        print(f"Results: {results_file}")
        print("=" * 80)

        # Prepare retry worker arguments
        worker_args = [
            (idx, traj_path, output_dir, prev_status, args.x_display, args.python)
            for idx, (traj_path, output_dir, prev_status) in enumerate(retry_entries)
        ]

        # Run retry workers
        results = []
        success_count = 0
        partial_count = 0
        failure_count = 0

        hazard_stats = defaultdict(lambda: {'success': 0, 'partial': 0, 'failed': 0, 'total': 0})

        log = open(log_file, 'w')
        log.write(f"Safety Trajectories Batch Retry - {timestamp}\n")
        log.write(f"Original Log: {args.retry_from_log}\n")
        log.write(f"Total Trajectories to Retry: {len(retry_entries)}\n")
        log.write(f"Retry PARTIAL: {retry_partial}\n")
        log.write(f"Retry FAILED: {retry_failed}\n")
        log.write(f"Parallel Processes: {args.num_processes}\n")
        log.write("=" * 80 + "\n\n")
        log.flush()

        with Pool(processes=args.num_processes) as pool:
            for result_entry in pool.imap(process_retry_worker, worker_args):
                idx = result_entry['idx']
                status = result_entry['status']
                prev_status = result_entry.get('prev_status', 'unknown')
                hazard_type = result_entry['hazard_type']
                output_dir = result_entry['output_dir']
                traj_path = result_entry['trajectory']
                stats = result_entry['stats']

                hazard_stats[hazard_type]['total'] += 1
                if status == 'SUCCESS':
                    success_count += 1
                    hazard_stats[hazard_type]['success'] += 1
                    status_msg = f"✓ SUCCESS (was {prev_status})"
                elif status == 'PARTIAL':
                    partial_count += 1
                    hazard_stats[hazard_type]['partial'] += 1
                    status_msg = f"⚠ PARTIAL (was {prev_status})"
                else:
                    failure_count += 1
                    hazard_stats[hazard_type]['failed'] += 1
                    status_msg = f"✗ FAILED (was {prev_status})"
                    if stats.get('execution_error'):
                        status_msg += f" - {stats['execution_error']}"

                print(f"[{idx+1}/{len(retry_entries)}] {hazard_type}: {Path(output_dir).name} - {status_msg}")

                log.write(f"\n[{idx+1}/{len(retry_entries)}] {traj_path}\n")
                log.write(f"Previous Status: {prev_status}\n")
                log.write(f"Status: {status}\n")
                log.write(f"Hazard Type: {hazard_type}\n")
                log.write(f"Return Code: {result_entry['returncode']}\n")
                log.write(f"Output: {output_dir}\n")
                if stats.get('execution_error'):
                    log.write(f"Error: {stats['execution_error']}\n")
                log.flush()

                results.append(result_entry)

                # Progress summary
                total_tested = len(results)
                if total_tested % 10 == 0 or total_tested == len(retry_entries):
                    print(f"\nRetry Progress: {total_tested}/{len(retry_entries)}")
                    print(f"  Success: {success_count} ({100*success_count/total_tested:.1f}%)")
                    print(f"  Partial: {partial_count} ({100*partial_count/total_tested:.1f}%)")
                    print(f"  Failed:  {failure_count} ({100*failure_count/total_tested:.1f}%)")
                    print()

        log.close()

        # Save results
        summary = {
            'timestamp': timestamp,
            'mode': 'retry',
            'original_log': args.retry_from_log,
            'total': len(retry_entries),
            'success': success_count,
            'partial': partial_count,
            'failed': failure_count,
            'hazard_stats': dict(hazard_stats),
            'results': results
        }

        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("RETRY COMPLETE")
        print("=" * 80)
        print(f"Total Retried: {len(retry_entries)}")
        print(f"  ✓ Success: {success_count} ({100*success_count/len(retry_entries):.1f}%)")
        print(f"  ⚠ Partial: {partial_count} ({100*partial_count/len(retry_entries):.1f}%)")
        print(f"  ✗ Failed:  {failure_count} ({100*failure_count/len(retry_entries):.1f}%)")
        print(f"\nResults saved to: {results_file}")
        print(f"Logs saved to: {log_file}")
        return

    # Normal mode - find all safety trajectories
    print(f"\nSearching for safety trajectories...")
    all_trajectories = find_trajectories(args.data_base)

    # Filter by hazard type if specified
    if args.hazard_type:
        all_trajectories = [
            t for t in all_trajectories if t[3] == args.hazard_type
        ]
        print(f"\nFiltered to {len(all_trajectories)} trajectories with hazard type: {args.hazard_type}")

    # Filter by split if specified
    if args.split:
        all_trajectories = [
            t for t in all_trajectories if t[2] == args.split
        ]
        print(f"\nFiltered to {len(all_trajectories)} trajectories in split: {args.split}")

    # Shuffle trajectories randomly
    random.seed(args.seed)
    random.shuffle(all_trajectories)
    print(f"\nShuffled {len(all_trajectories)} trajectories (seed={args.seed})")

    trajectories = all_trajectories

    if args.max_trajs:
        trajectories = trajectories[:args.max_trajs]
        print(f"\nLimiting to {args.max_trajs} trajectories for testing")

    # Create output directory
    os.makedirs(args.output_base, exist_ok=True)

    # Create log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.output_base, f'safety_batch_test_{timestamp}.log')
    results_file = os.path.join(args.output_base, f'safety_batch_results_{timestamp}.json')

    print(f"\nStarting batch test of {len(trajectories)} safety trajectories")
    print(f"Using {args.num_processes} parallel processes")
    print(f"Logging to: {log_file}")
    print(f"Results: {results_file}")
    print("=" * 80)

    # Prepare worker arguments - all workers share the same x_display and python executable
    worker_args = [
        (idx, traj_data, args.output_base, args.x_display, args.python)
        for idx, traj_data in enumerate(trajectories)
    ]

    # Run pipeline on trajectories in parallel
    results = []
    success_count = 0
    partial_count = 0
    failure_count = 0
    skipped_count = 0

    # Track hazard-specific stats
    hazard_stats = defaultdict(lambda: {'success': 0, 'partial': 0, 'failed': 0, 'skipped': 0, 'total': 0})

    # Open log file for writing
    log = open(log_file, 'w')
    log.write(f"Safety Trajectories Batch Pipeline Test - {timestamp}\n")
    log.write(f"Data Base: {args.data_base}\n")
    log.write(f"Total Trajectories: {len(trajectories)}\n")
    log.write(f"Random Seed: {args.seed}\n")
    log.write(f"Parallel Processes: {args.num_processes}\n")
    if args.hazard_type:
        log.write(f"Hazard Type Filter: {args.hazard_type}\n")
    if args.split:
        log.write(f"Split Filter: {args.split}\n")
    log.write("=" * 80 + "\n\n")
    log.flush()

    # Use multiprocessing Pool to process trajectories in parallel
    with Pool(processes=args.num_processes) as pool:
        # Use imap to get results as they complete
        for result_entry in pool.imap(process_trajectory_worker, worker_args):
            idx = result_entry['idx']
            traj_file = result_entry['trajectory']
            scene_num = result_entry['floor_plan']
            split = result_entry['split']
            hazard_type = result_entry['hazard_type']
            task_name = result_entry['task']
            traj_name = result_entry['trial']
            status = result_entry['status']
            stats = result_entry['stats']
            output_dir = result_entry['output_dir']
            clear_sink_used = result_entry['clear_sink_objects_used']

            # Update counts
            hazard_stats[hazard_type]['total'] += 1
            if status == 'SUCCESS':
                success_count += 1
                hazard_stats[hazard_type]['success'] += 1
                status_msg = "✓ SUCCESS"
            elif status == 'PARTIAL':
                partial_count += 1
                hazard_stats[hazard_type]['partial'] += 1
                status_msg = "⚠ PARTIAL SUCCESS"
            elif status == 'SKIPPED':
                skipped_count += 1
                hazard_stats[hazard_type]['skipped'] += 1
                status_msg = "⊘ SKIPPED (already processed)"
            else:
                failure_count += 1
                hazard_stats[hazard_type]['failed'] += 1
                status_msg = "✗ FAILED"
                if stats['execution_error']:
                    status_msg += f" - {stats['execution_error']}"

            # Print result
            clear_sink_marker = " [CLEAR_SINK]" if clear_sink_used else ""
            print(f"[{idx+1}/{len(trajectories)}] {split}/FloorPlan{scene_num}/{hazard_type}: {task_name}/{traj_name} - {status_msg}{clear_sink_marker}")

            # Write to log
            log.write(f"\n[{idx+1}/{len(trajectories)}] {traj_file}\n")
            log.write(f"Split: {split}, Hazard: {hazard_type}, Task: {task_name}, Trial: {traj_name}, FloorPlan: {scene_num}\n")
            log.write(f"Clear Sink Objects: {clear_sink_used}\n")
            log.write(f"Status: {status}\n")
            log.write(f"Return Code: {result_entry['returncode']}\n")
            log.write(f"Plan Steps: {stats['plan_steps']}\n")
            if stats['execution_error']:
                log.write(f"Error: {stats['execution_error']}\n")
            log.write(f"Output: {output_dir}\n")
            if status == 'SKIPPED':
                log.write(f"Cleanup: Skipped (already processed)\n")
            elif status == 'FAILED':
                log.write(f"Cleanup: Videos and images removed (failed run)\n")
                # Write stderr and stdout for failed runs
                stderr = result_entry.get('stderr', '')
                stdout = result_entry.get('stdout', '')
                if stderr:
                    log.write(f"\nSTDERR:\n{stderr}\n")
                if stdout:
                    # Write last 2000 chars of stdout to see what happened
                    log.write(f"\nSTDOUT (last 2000 chars):\n{stdout[-2000:]}\n")
            else:
                log.write(f"Cleanup: Videos and images preserved (successful run)\n")
            log.flush()

            results.append(result_entry)

            # Print progress summary every 10 trajectories
            total_tested = len(results)
            if total_tested % 10 == 0 or total_tested == len(trajectories):
                print(f"\nProgress: {total_tested}/{len(trajectories)} tested")
                print(f"  Success: {success_count} ({100*success_count/total_tested:.1f}%)")
                print(f"  Partial: {partial_count} ({100*partial_count/total_tested:.1f}%)")
                print(f"  Failed:  {failure_count} ({100*failure_count/total_tested:.1f}%)")
                print(f"  Skipped: {skipped_count} ({100*skipped_count/total_tested:.1f}%)")

                # Print hazard-specific breakdown
                print(f"\n  By Hazard Type:")
                for hazard in sorted(hazard_stats.keys()):
                    h = hazard_stats[hazard]
                    if h['total'] > 0:
                        print(f"    {hazard}: {h['success']}✓ {h['partial']}⚠ {h['failed']}✗ {h['skipped']}⊘ (total: {h['total']})")
                print()

                # Also write progress to log file
                log.write(f"\n{'='*80}\n")
                log.write(f"PROGRESS UPDATE: {total_tested}/{len(trajectories)} trajectories tested\n")
                log.write(f"{'='*80}\n")
                log.write(f"Overall: {success_count} success, {partial_count} partial, {failure_count} failed, {skipped_count} skipped\n\n")
                log.write(f"By Hazard Type:\n")
                for hazard in sorted(hazard_stats.keys()):
                    h = hazard_stats[hazard]
                    if h['total'] > 0:
                        log.write(f"  {hazard}:\n")
                        log.write(f"    Success: {h['success']} ({100*h['success']/h['total']:.1f}%)\n")
                        log.write(f"    Partial: {h['partial']} ({100*h['partial']/h['total']:.1f}%)\n")
                        log.write(f"    Failed:  {h['failed']} ({100*h['failed']/h['total']:.1f}%)\n")
                        log.write(f"    Skipped: {h['skipped']} ({100*h['skipped']/h['total']:.1f}%)\n")
                        log.write(f"    Total:   {h['total']}\n\n")
                log.write(f"{'='*80}\n\n")
                log.flush()

    log.close()

    # Save results JSON
    summary = {
        'timestamp': timestamp,
        'total': len(trajectories),
        'success': success_count,
        'partial': partial_count,
        'failed': failure_count,
        'skipped': skipped_count,
        'hazard_type_filter': args.hazard_type,
        'split_filter': args.split,
        'random_seed': args.seed,
        'hazard_stats': dict(hazard_stats),
        'results': results
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 80)
    print("BATCH TEST COMPLETE")
    print("=" * 80)
    print(f"Total Trajectories: {len(trajectories)}")

    if len(trajectories) > 0:
        print(f"  ✓ Full Success:    {success_count} ({100*success_count/len(trajectories):.1f}%)")
        print(f"  ⚠ Partial Success: {partial_count} ({100*partial_count/len(trajectories):.1f}%)")
        print(f"  ✗ Failed:          {failure_count} ({100*failure_count/len(trajectories):.1f}%)")
        print(f"  ⊘ Skipped:         {skipped_count} ({100*skipped_count/len(trajectories):.1f}%)")
    else:
        print("  No trajectories to process!")
        return

    # Print hazard type breakdown
    print("\nResults by Hazard Type:")
    for hazard in sorted(hazard_stats.keys()):
        h_stats = hazard_stats[hazard]
        total = h_stats['total']
        success = h_stats['success']
        partial = h_stats['partial']
        failed = h_stats['failed']
        skipped = h_stats['skipped']
        print(f"  {hazard}:")
        print(f"    Total: {total}")
        print(f"    Success: {success} ({100*success/total:.1f}%)")
        print(f"    Partial: {partial} ({100*partial/total:.1f}%)")
        print(f"    Failed: {failed} ({100*failed/total:.1f}%)")
        print(f"    Skipped: {skipped} ({100*skipped/total:.1f}%)")

    print(f"\nResults saved to: {results_file}")
    print(f"Logs saved to: {log_file}")

    # Print failure breakdown by hazard type
    if failure_count > 0:
        print("\nTop Failures by Hazard Type:")
        failures_by_hazard = defaultdict(int)
        for r in results:
            if r['status'] == 'FAILED':
                failures_by_hazard[r['hazard_type']] += 1
        for hazard in sorted(failures_by_hazard.items(), key=lambda x: x[1], reverse=True):
            print(f"  {hazard[0]}: {hazard[1]} failures")

if __name__ == '__main__':
    main()
