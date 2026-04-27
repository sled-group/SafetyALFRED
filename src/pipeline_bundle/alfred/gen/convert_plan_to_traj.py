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
        'pddl_params': original_traj.get('pddl_params', {}),
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

            # Extract THOR action parameters if available
            thor_action = low_action_result.get('thor_action', {})

            # Handle TeleportFull with THOR 5.0 format
            if action_name == 'TeleportFull' and thor_action:
                api_action = {
                    'action': 'TeleportFull',
                    'x': thor_action.get('x'),
                    'y': thor_action.get('y'),
                    'z': thor_action.get('z'),
                    'rotation': thor_action.get('rotation'),  # Already in dict format from render_plan_with_navigation
                    'horizon': thor_action.get('horizon'),
                    'standing': thor_action.get('standing', True)
                }

                # Check if this is a look_at_object teleport
                if low_action_result.get('action_type') == 'look_at_object':
                    api_action['action_type'] = 'look_at_object'
                    api_action['target_object'] = low_action_result.get('target_object')

            # Add objectId for manipulation actions
            elif action_name in ['PickupObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']:
                if 'objectId' in thor_action:
                    api_action['objectId'] = thor_action['objectId']

            # Add receptacleObjectId for PutObject
            elif action_name == 'PutObject':
                if 'objectId' in thor_action:
                    api_action['objectId'] = thor_action['objectId']
                if 'receptacleObjectId' in thor_action:
                    api_action['receptacleObjectId'] = thor_action['receptacleObjectId']
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

    # Filter out redundant put-then-pickup and pickup-then-put patterns
    print(f"\n=== Filtering redundant actions from {len(new_traj['plan']['low_actions'])} low-level actions ===")

    # Apply filters iteratively until no more patterns are found
    actions_to_filter = new_traj['plan']['low_actions'][:]
    previous_length = len(actions_to_filter)
    iteration = 0

    while True:
        iteration += 1
        filtered_low_actions = []
        i = 0
        original_actions = actions_to_filter[:]

        while i < len(original_actions):
            action = original_actions[i]
            action_name = action['api_action']['action']

            # Check for pattern: OpenObject(microwave) → PickupObject → CloseObject(microwave) → [TeleportFull*] → OpenObject(microwave) → PutObject(same object) → CloseObject(microwave)
            if action_name == 'OpenObject' and 'objectId' in action['api_action']:
                receptacle_id = action['api_action']['objectId']

                # Helper function to skip TeleportFull actions
                def skip_teleports(start_idx):
                    idx = start_idx
                    while idx < len(original_actions) and original_actions[idx]['api_action']['action'] == 'TeleportFull':
                        idx += 1
                    return idx

                # First check for microwave cycle pattern
                if 'Microwave' in receptacle_id and i + 1 < len(original_actions):
                    # Look for: OpenObject → PickupObject
                    idx_1 = skip_teleports(i + 1)
                    if idx_1 < len(original_actions) and original_actions[idx_1]['api_action']['action'] == 'PickupObject':
                        pickup_object_id = original_actions[idx_1]['api_action'].get('objectId')

                        # Look for: PickupObject → CloseObject
                        idx_2 = skip_teleports(idx_1 + 1)
                        if (idx_2 < len(original_actions) and
                            original_actions[idx_2]['api_action']['action'] == 'CloseObject' and
                            original_actions[idx_2]['api_action'].get('objectId') == receptacle_id):

                            # Look for: CloseObject → OpenObject (same microwave)
                            idx_3 = skip_teleports(idx_2 + 1)
                            if (idx_3 < len(original_actions) and
                                original_actions[idx_3]['api_action']['action'] == 'OpenObject' and
                                original_actions[idx_3]['api_action'].get('objectId') == receptacle_id):

                                # Look for: OpenObject → PutObject (same object)
                                idx_4 = skip_teleports(idx_3 + 1)
                                if (idx_4 < len(original_actions) and
                                    original_actions[idx_4]['api_action']['action'] == 'PutObject' and
                                    original_actions[idx_4]['api_action'].get('objectId') == pickup_object_id):

                                    # Look for: PutObject → CloseObject (same microwave)
                                    idx_5 = skip_teleports(idx_4 + 1)
                                    if (idx_5 < len(original_actions) and
                                        original_actions[idx_5]['api_action']['action'] == 'CloseObject' and
                                        original_actions[idx_5]['api_action'].get('objectId') == receptacle_id):

                                        # Found the redundant pattern! Skip all actions up to and including the final CloseObject
                                        picked_object = pickup_object_id.split('|')[0] if pickup_object_id else 'unknown'
                                        print(f"  Filtering redundant microwave cycle: Open→Pickup({picked_object})→Close→Open→Put({picked_object})→Close")
                                        i = idx_5 + 1
                                        continue


            # Check for PutObject followed by PickupObject on same object
            if action_name == 'PutObject' and 'objectId' in action['api_action']:
                put_object_id = action['api_action']['objectId']

                # Look ahead for immediate PickupObject
                if i + 1 < len(original_actions):
                    next_action = original_actions[i + 1]
                    if (next_action['api_action']['action'] == 'PickupObject' and
                        'objectId' in next_action['api_action'] and
                        next_action['api_action']['objectId'] == put_object_id):
                        # Skip both actions
                        print(f"  Filtering redundant: PutObject→PickupObject {put_object_id.split('|')[0]}")
                        i += 2
                        continue

            # Check for PickupObject followed by PutObject in same receptacle
            elif action_name == 'PickupObject' and 'objectId' in action['api_action']:
                pickup_object_id = action['api_action']['objectId']

                # Find what receptacle this object was in before pickup
                # Look backwards for the most recent PutObject of this object
                original_receptacle_id = None
                for j in range(i - 1, -1, -1):
                    prev_action = original_actions[j]
                    if (prev_action['api_action']['action'] == 'PutObject' and
                        prev_action['api_action'].get('objectId') == pickup_object_id):
                        original_receptacle_id = prev_action['api_action'].get('receptacleObjectId')
                        break

                # Look ahead for immediate PutObject
                if i + 1 < len(original_actions):
                    next_action = original_actions[i + 1]
                    if (next_action['api_action']['action'] == 'PutObject' and
                        'objectId' in next_action['api_action'] and
                        next_action['api_action']['objectId'] == pickup_object_id):

                        # Get the receptacle where object is being put
                        put_receptacle_id = next_action['api_action'].get('receptacleObjectId')

                        # Only filter if putting back in the same receptacle AND not placed in different receptacle later
                        if original_receptacle_id and put_receptacle_id == original_receptacle_id:
                            # Check if this object is put in a DIFFERENT receptacle later
                            found_different_receptacle = False
                            for j in range(i + 2, len(original_actions)):
                                future_action = original_actions[j]
                                if (future_action['api_action']['action'] == 'PutObject' and
                                    future_action['api_action'].get('objectId') == pickup_object_id):
                                    future_receptacle_id = future_action['api_action'].get('receptacleObjectId')
                                    # Check if receptacle ID is different
                                    if future_receptacle_id != original_receptacle_id:
                                        # Object is placed in a different receptacle later
                                        found_different_receptacle = True
                                        break

                            # Only filter if object is NOT placed in a different receptacle later
                            if not found_different_receptacle:
                                # Skip both actions - completely redundant
                                print(f"  Filtering redundant: PickupObject→PutObject {pickup_object_id.split('|')[0]} (back into {put_receptacle_id.split('|')[0] if put_receptacle_id else 'unknown'})")
                                i += 2
                                continue

            # Add action if not filtered
            filtered_low_actions.append(action)
            i += 1

        # Check if any filtering happened in this iteration
        if len(filtered_low_actions) == previous_length:
            # No changes, we're done
            break

        # Update for next iteration
        actions_to_filter = filtered_low_actions
        previous_length = len(filtered_low_actions)
        print(f"  Iteration {iteration}: Filtered to {len(filtered_low_actions)} actions")

    # Update trajectory with filtered actions and reindex
    if len(filtered_low_actions) != len(new_traj['plan']['low_actions']):
        print(colored(f"  Filtered {len(new_traj['plan']['low_actions']) - len(filtered_low_actions)} redundant low-level actions", 'yellow'))
        new_traj['plan']['low_actions'] = filtered_low_actions

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
