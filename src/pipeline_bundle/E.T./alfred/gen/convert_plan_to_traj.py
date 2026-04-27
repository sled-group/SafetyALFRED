#!/usr/bin/env python3
"""
Convert a PDDL plan execution log to ALFRED trajectory format.

This allows the generated plan to be rendered using render_safety_trajs_test.py
"""

import os
import sys
import json
import argparse
import shutil
from termcolor import colored

def convert_plan_to_traj(original_traj_path, execution_log_path, output_dir):
    """
    Convert plan execution log to ALFRED trajectory format.

    Args:
        original_traj_path: Path to original traj_data.json (for scene info)
        execution_log_path: Path to execution_log.json from render_plan_with_navigation.py
        output_dir: Directory to save converted trajectory
    """

    # Load original trajectory for scene information
    with open(original_traj_path, 'r') as f:
        original_traj = json.load(f)

    # Load execution log
    with open(execution_log_path, 'r') as f:
        execution_log = json.load(f)

    # Create new trajectory based on original
    new_traj = {
        'scene': original_traj['scene'],
        'task_id': original_traj.get('task_id', 'generated_plan'),
        'task_type': original_traj.get('task_type', 'pick_and_place_simple'),
        'turk_annotations': original_traj.get('turk_annotations', {}),
        'plan': {
            'high_pddl': [],
            'low_actions': []
        },
        'images': []
    }

    # Convert execution log to low_actions format
    low_idx = 0
    high_idx = 0

    for step in execution_log:
        if 'low_level_actions' not in step:
            continue

        # Add high-level PDDL action
        high_pddl_action = {
            'discrete_action': {
                'action': step['pddl_action'].split()[0],
                'args': step['pddl_action'].split()[1:]
            },
            'high_idx': high_idx,
            'planner_action': {
                'action': step['pddl_action'].split()[0],
                'location': step['pddl_action'].split()[-1] if 'gotolocation' in step['pddl_action'] else None
            }
        }
        new_traj['plan']['high_pddl'].append(high_pddl_action)

        # Convert each low-level action
        for low_action_result in step['low_level_actions']:
            action_name = low_action_result['action']

            # Build api_action
            api_action = {
                'action': action_name,
                'forceAction': True
            }

            # Add objectId for manipulation actions
            if action_name in ['PickupObject', 'OpenObject', 'CloseObject']:
                # Try to extract from thor_action in failed actions
                if 'thor_action' in step and 'objectId' in step['thor_action']:
                    api_action['objectId'] = step['thor_action']['objectId']

            # Add receptacleObjectId for PutObject
            if action_name == 'PutObject':
                if 'thor_action' in step:
                    if 'objectId' in step['thor_action']:
                        api_action['objectId'] = step['thor_action']['objectId']
                    if 'receptacleObjectId' in step['thor_action']:
                        api_action['receptacleObjectId'] = step['thor_action']['receptacleObjectId']
                    api_action['placeStationary'] = True

            # Add discrete_action (simplified version)
            discrete_action = {
                'action': action_name,
                'args': {}
            }

            low_action = {
                'api_action': api_action,
                'discrete_action': discrete_action,
                'high_idx': high_idx
            }

            new_traj['plan']['low_actions'].append(low_action)
            low_idx += 1

        high_idx += 1

    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)

    # Save converted trajectory
    output_traj_path = os.path.join(output_dir, 'traj_data.json')
    with open(output_traj_path, 'w') as f:
        json.dump(new_traj, f, indent=4, sort_keys=True)

    print(colored(f"✓ Converted trajectory saved: {output_traj_path}", 'green'))
    print(f"  - High-level actions: {len(new_traj['plan']['high_pddl'])}")
    print(f"  - Low-level actions: {len(new_traj['plan']['low_actions'])}")

    return output_traj_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDDL plan execution to ALFRED trajectory format')
    parser.add_argument('--original_traj', type=str, required=True,
                       help='Path to original traj_data.json')
    parser.add_argument('--execution_log', type=str, required=True,
                       help='Path to execution_log.json from render_plan_with_navigation.py')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save converted trajectory')

    args = parser.parse_args()

    try:
        convert_plan_to_traj(
            args.original_traj,
            args.execution_log,
            args.output_dir
        )
        return 0
    except Exception as e:
        print(colored(f"✗ Error: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
