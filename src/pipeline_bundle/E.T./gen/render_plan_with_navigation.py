#!/usr/bin/env python3
"""
Script to render and visualize a plan generated from PDDL with proper navigation.

Takes a trajectory JSON, generates PDDL, runs the planner, and executes
the plan in THOR using smooth navigation (not teleport) while saving video frames.

Usage:
    python render_plan_with_navigation.py --traj_json <path> --domain <path> --output_dir <path>
"""

import os
import sys
import json
import argparse
import numpy as np
from termcolor import colored

# Add ALFRED paths
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'gen'))

from env.thor_env import ThorEnv
from gen import constants
from gen.utils import video_util, game_util
from gen.graph.graph_obj import Graph
from generate_problem_pddl_full import generate_pddl_from_traj_full

# Import DANLI planner
import importlib.util
_RP_BUNDLE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
danli_planner_path = os.path.join(
    _RP_BUNDLE_ROOT, 'alfred_git', 'alfred', 'data', 'DANLI', 'pddl', 'planner.py')
spec = importlib.util.spec_from_file_location("danli_planner", danli_planner_path)
danli_planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(danli_planner)
PDDLPlanner = danli_planner.PDDLPlanner


def correct_slice_id(object_id, env):
    """
    Corrects object ID for sliced objects.

    When an object like Bread is sliced in THOR, it creates new objects with IDs like
    Bread|...|BreadSliced_1, Bread|...|BreadSliced_2, etc.

    This function finds the closest slice to the agent if the object has been sliced.
    Based on ALFRED's game_state_base.py:correct_slice_id()

    Args:
        object_id: Original object ID (e.g., 'Bread|+00.86|+00.99|-00.03')
        env: THOR environment

    Returns:
        Corrected object ID (e.g., 'Bread|+00.86|+00.99|-00.03|BreadSliced_1')
    """
    # Get object metadata
    main_obj = game_util.get_object(object_id, env.last_event.metadata)

    if main_obj is None:
        return object_id

    # Check if object is sliceable and has been sliced
    if (main_obj['objectType'] in constants.VAL_ACTION_OBJECTS['Sliceable'] and
        main_obj.get('isSliced', False)):

        slice_obj_type = main_obj['objectType'] + "Sliced"

        # Find all slice objects and pick the closest one
        min_d = None
        best_slice_id = None
        sidx = 1

        while True:
            slice_obj_id = main_obj['objectId'] + "|" + slice_obj_type + "_%d" % sidx
            slice_obj = game_util.get_object(slice_obj_id, env.last_event.metadata)

            if slice_obj is None:
                break

            if min_d is None or slice_obj['distance'] < min_d:
                min_d = slice_obj['distance']
                best_slice_id = slice_obj_id

            sidx += 1

        if best_slice_id:
            return best_slice_id

    return object_id


def pddl_action_to_navigation_sequence(pddl_action, env, nav_graph, agent_loc_history, pddl_problem_dir=None, next_pddl_action=None, use_teleport=False):
    """
    Convert a PDDL action to a sequence of low-level navigation actions.

    Args:
        pddl_action: Tuple like ('gotolocation', 'agent1', 'loc_start', 'loc_end')
        pddl_problem_dir: Optional directory containing problem.pddl file
        env: THOR environment
        nav_graph: Navigation graph for pathfinding
        agent_loc_history: List tracking agent location history
        next_pddl_action: Optional next PDDL action in sequence (for optimization)
        use_teleport: If True, use TeleportFull instead of navigation graph (allows exact rotation angles)

    Returns:
        list: List of low-level THOR API action dicts
    """
    action_name = pddl_action[0].lower()

    if action_name == 'gotolocation':
        # Extract target location
        target_loc = pddl_action[3]  # loc_end
        # Parse location: loc_bar__minus_14_bar_5_bar_3_bar_30
        # Format: loc|x|y|rotation_index|horizon
        # where rotation_index is 0-3 (0=0°, 1=90°, 2=180°, 3=270°)
        parts = target_loc.replace('loc_bar_', '').split('_bar_')

        # Convert coordinates
        def parse_coord(s):
            s = s.replace('_minus_', '-').replace('_plus_', '+').replace('_dot_', '.')
            return float(s)

        x = int(parse_coord(parts[0]))  # Grid x coordinate
        y = int(parse_coord(parts[1]))  # Grid y coordinate (z in THOR)
        rotation_index = int(parse_coord(parts[2]))  # Rotation index (0, 1, 2, 3)
        horizon = int(parse_coord(parts[3]))  # Horizon angle

        # Get current pose
        current_pose = game_util.get_pose(env.last_event)

        # Target pose in graph coordinates (x, y, rotation_index, horizon)
        target_pose = (x, y, rotation_index, horizon)

        print(f"  Navigation: {current_pose} -> {target_pose}")

        # Choose between teleport or navigation based on use_teleport flag
        if use_teleport:
            # Use TeleportFull for instant positioning with calculated rotation to face target object
            x_pos = x * constants.AGENT_STEP_SIZE
            z_pos = y * constants.AGENT_STEP_SIZE

            # Get current agent Y position (don't hardcode it)
            agent_y = env.last_event.metadata['agent']['position']['y']

            # Calculate rotation to face the next object that will be interacted with
            rotation_deg = rotation_index * 90  # Default to PDDL rotation

            if next_pddl_action:
                next_action_name = next_pddl_action[0].lower()
                target_object_id = None

                # Extract the object/receptacle that will be interacted with next
                if next_action_name == 'pickupobjectinreceptacle1':
                    # Object is at index 3, receptacle at 4
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[4])  # Use receptacle
                elif next_action_name == 'pickupobjectnoreceptacle':
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])
                elif next_action_name == 'putobjectinreceptacle1':
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[5])  # Receptacle
                elif next_action_name in ['openobject', 'closeobject']:
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])

                # Calculate angle to face the target object
                if target_object_id:
                    objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
                    target_obj = objects.get(target_object_id)

                    if target_obj:
                        # Calculate angle from agent position to target object
                        target_pos = target_obj['position']
                        dx = target_pos['x'] - x_pos
                        dz = target_pos['z'] - z_pos

                        # Calculate angle in degrees (arctan2 returns angle in radians)
                        # AI2-THOR: 0° = North (+Z), 90° = East (+X), 180° = South (-Z), 270° = West (-X)
                        angle_rad = np.arctan2(dx, dz)
                        rotation_deg = np.degrees(angle_rad)

                        # Normalize to 0-360 range
                        if rotation_deg < 0:
                            rotation_deg += 360

                        print(f"  Calculated rotation to face {target_object_id}: {rotation_deg:.1f}°")

            # Calculate horizon angle to look at the target object (if next action exists)
            calculated_horizon = horizon  # Default to PDDL horizon

            if next_pddl_action and target_object_id and target_obj:
                # Calculate horizon to look at target object center
                target_pos = target_obj['position']

                # Horizontal distance
                dx = target_pos['x'] - x_pos
                dz = target_pos['z'] - z_pos
                horizontal_dist = np.sqrt(dx**2 + dz**2)

                # Vertical distance (camera height to object center)
                # Camera is at agent_y + camera offset (usually 0.675m for standing)
                camera_height = agent_y + 0.675
                vertical_dist = target_pos['y'] - camera_height

                # Calculate horizon angle (negative = look up, positive = look down)
                if horizontal_dist > 0.01:  # Avoid division by zero
                    horizon_rad = np.arctan2(-vertical_dist, horizontal_dist)  # Negative because THOR convention
                    calculated_horizon = np.degrees(horizon_rad)
                    # Clamp to THOR limits: -30 (up) to +60 (down)
                    calculated_horizon = np.clip(calculated_horizon, -30, 60)
                    print(f"  Calculated horizon to look at object center: {calculated_horizon:.1f}°")

            print(f"  Using TeleportFull to ({x_pos:.2f}, {agent_y:.2f}, {z_pos:.2f}) facing {rotation_deg:.1f}° horizon {calculated_horizon:.1f}°")
            return [{
                'action': 'TeleportFull',
                'x': x_pos,
                'y': agent_y,
                'z': z_pos,
                'rotation': {'x': 0, 'y': rotation_deg, 'z': 0},  # THOR 5.0 format
                'horizon': calculated_horizon,
                'standing': True
            }]
        else:
            # Use navigation graph to get low-level actions (MoveAhead, Rotate, etc.)
            try:
                actions, path = nav_graph.get_shortest_path(current_pose, target_pose)
                print(f"  Using navigation with {len(actions)} low-level actions")
                return actions
            except Exception as e:
                print(colored(f"  Warning: Navigation failed: {e}", 'yellow'))
                print(f"  Falling back to teleport")
                # Fallback to teleport if navigation fails
                x_pos = x * constants.AGENT_STEP_SIZE
                z_pos = y * constants.AGENT_STEP_SIZE
                agent_y = env.last_event.metadata['agent']['position']['y']
                rotation_deg = rotation_index * 90
                return [{
                    'action': 'TeleportFull',
                    'x': x_pos,
                    'y': agent_y,
                    'z': z_pos,
                    'rotation': {'x': 0, 'y': rotation_deg, 'z': 0},  # THOR 5.0 format
                    'horizon': horizon,
                    'standing': True
                }]

    elif action_name == 'pickupobjectinreceptacle1':
        # pickupobjectinreceptacle1 agent1 loc object_id receptacle_id
        # Extract object ID and receptacle ID from PDDL format
        object_id_pddl = pddl_action[3]
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)
        # Correct object ID if it's a sliced object (e.g., Bread → Bread|...|BreadSliced_1)
        object_id_thor = correct_slice_id(object_id_thor, env)

        receptacle_id_pddl = pddl_action[4]
        receptacle_id_thor = convert_pddl_object_to_thor(receptacle_id_pddl)

        actions = []

        # Check if receptacle needs to be opened (e.g., microwave, fridge)
        objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
        receptacle = objects.get(receptacle_id_thor)

        is_openable = receptacle and receptacle.get('openable', False)

        if is_openable and not receptacle.get('isOpen', False):
            actions.append({
                'action': 'OpenObject',
                'objectId': receptacle_id_thor,
                'forceAction': True
            })

        # Pickup the object
        actions.append({
            'action': 'PickupObject',
            'objectId': object_id_thor,
            'forceAction': True,
            'manualInteract': False
        })

        # Check if next action uses the same receptacle
        should_close = is_openable
        if should_close and next_pddl_action:
            next_action_name = next_pddl_action[0].lower()
            # Check if next action is put/pickup on same receptacle
            if next_action_name in ['putobjectinreceptacle1', 'pickupobjectinreceptacle1']:
                # Extract receptacle from next action
                if next_action_name == 'putobjectinreceptacle1':
                    next_receptacle_pddl = next_pddl_action[5]
                else:  # pickupobjectinreceptacle1
                    next_receptacle_pddl = next_pddl_action[4]

                next_receptacle_thor = convert_pddl_object_to_thor(next_receptacle_pddl)
                if next_receptacle_thor == receptacle_id_thor:
                    should_close = False
                    print(f"  Skipping close - next action uses same receptacle")

        # Close receptacle after picking up (matching ALFRED behavior)
        if should_close:
            actions.append({
                'action': 'CloseObject',
                'objectId': receptacle_id_thor,
                'forceAction': True
            })

        return actions

    elif action_name == 'pickupobjectnoreceptacle':
        object_id_pddl = pddl_action[3]
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)
        # Correct object ID if it's a sliced object (e.g., Bread → Bread|...|BreadSliced_1)
        object_id_thor = correct_slice_id(object_id_thor, env)

        return [{
            'action': 'PickupObject',
            'objectId': object_id_thor,
            'forceAction': True,
            'manualInteract': False
        }]

    elif action_name == 'putobjectinreceptacle1':
        # putobjectinreceptacle1 agent1 loc otype object receptacle rtype
        # Extract receptacle ID (now at index 5 with the new rtype parameter at index 6)
        receptacle_id_pddl = pddl_action[5]
        receptacle_id_thor = convert_pddl_object_to_thor(receptacle_id_pddl)

        actions = []

        # Check if receptacle needs to be opened
        objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
        receptacle = objects.get(receptacle_id_thor)

        is_openable = receptacle and receptacle.get('openable', False)
        is_open = receptacle and receptacle.get('isOpen', False)

        if is_openable and not is_open:
            actions.append({
                'action': 'OpenObject',
                'objectId': receptacle_id_thor,
                'forceAction': True
            })

        # Get held object from inventory
        if len(env.last_event.metadata['inventoryObjects']) > 0:
            inv_obj_id = env.last_event.metadata['inventoryObjects'][0]['objectId']
            # Put object
            actions.append({
                'action': 'PutObject',
                'objectId': inv_obj_id,
                'receptacleObjectId': receptacle_id_thor,
                'forceAction': True,
                'placeStationary': True
            })
        else:
            print(colored("  Warning: No object in inventory to put", 'yellow'))

        # Check if next action uses the same receptacle
        should_close = is_openable
        if should_close and next_pddl_action:
            next_action_name = next_pddl_action[0].lower()
            # Check if next action is put/pickup on same receptacle
            if next_action_name in ['putobjectinreceptacle1', 'pickupobjectinreceptacle1']:
                # Extract receptacle from next action
                if next_action_name == 'putobjectinreceptacle1':
                    next_receptacle_pddl = next_pddl_action[5]
                else:  # pickupobjectinreceptacle1
                    next_receptacle_pddl = next_pddl_action[4]

                next_receptacle_thor = convert_pddl_object_to_thor(next_receptacle_pddl)
                if next_receptacle_thor == receptacle_id_thor:
                    should_close = False
                    print(f"  Skipping close - next action uses same receptacle")

        # Close receptacle after putting object (matching ALFRED decomposition)
        if should_close:
            actions.append({
                'action': 'CloseObject',
                'objectId': receptacle_id_thor,
                'forceAction': True
            })

        return actions

    elif action_name == 'putobjectinreceptacleobject1':
        # Put object into a movable receptacle object (e.g., pencil in bowl)
        # putobjectinreceptacleobject1 agent1 loc objtype obj_id recep_id parent_recep_id
        receptacle_id_pddl = pddl_action[5]  # The movable receptacle (e.g., bowl)
        receptacle_id_thor = convert_pddl_object_to_thor(receptacle_id_pddl)

        actions = []

        # Get the target receptacle and check its parent receptacles (matching ALFRED logic)
        objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
        target_receptacle = objects.get(receptacle_id_thor)

        # Check if target receptacle has parent receptacles that need opening
        parent_to_open = None
        if target_receptacle and target_receptacle.get('parentReceptacles'):
            parent_recep_ids = target_receptacle['parentReceptacles']
            # Find first openable parent (matching ALFRED's logic)
            for parent_id in parent_recep_ids:
                parent_obj = objects.get(parent_id)
                if parent_obj and parent_obj.get('openable', False):
                    parent_to_open = parent_obj
                    break

        # Open parent receptacle if needed
        if parent_to_open and not parent_to_open.get('isOpen', False):
            actions.append({
                'action': 'OpenObject',
                'objectId': parent_to_open['objectId'],
                'forceAction': True
            })

        # Get held object from inventory
        if len(env.last_event.metadata['inventoryObjects']) > 0:
            inv_obj_id = env.last_event.metadata['inventoryObjects'][0]['objectId']
            # Put object into movable receptacle
            actions.append({
                'action': 'PutObject',
                'objectId': inv_obj_id,
                'receptacleObjectId': receptacle_id_thor,
                'forceAction': True,
                'placeStationary': True
            })
        else:
            print(colored("  Warning: No object in inventory to put", 'yellow'))

        # Don't close parent receptacle here - we might need to pick up the
        # movable receptacle in the next action. Let pickupobjectinreceptacle1
        # handle opening/closing as needed.

        return actions

    elif action_name == 'openobject':
        receptacle_id_pddl = pddl_action[3]
        receptacle_id_thor = convert_pddl_object_to_thor(receptacle_id_pddl)

        return [{
            'action': 'OpenObject',
            'objectId': receptacle_id_thor,
            'forceAction': True
        }]

    elif action_name == 'closeobject':
        receptacle_id_pddl = pddl_action[3]
        receptacle_id_thor = convert_pddl_object_to_thor(receptacle_id_pddl)

        return [{
            'action': 'CloseObject',
            'objectId': receptacle_id_thor,
            'forceAction': True
        }]

    elif action_name == 'heatobject':
        # heatobject agent1 loc microwave_id object_id
        microwave_id_pddl = pddl_action[3]
        microwave_id_thor = convert_pddl_object_to_thor(microwave_id_pddl)

        actions = []

        # Get held object from inventory
        if len(env.last_event.metadata['inventoryObjects']) > 0:
            inv_obj_id = env.last_event.metadata['inventoryObjects'][0]['objectId']

            # Open microwave
            actions.append({
                'action': 'OpenObject',
                'objectId': microwave_id_thor,
                'forceAction': True
            })

            # Put object in microwave
            actions.append({
                'action': 'PutObject',
                'objectId': inv_obj_id,
                'receptacleObjectId': microwave_id_thor,
                'forceAction': True,
                'placeStationary': True
            })

            # Close microwave
            actions.append({
                'action': 'CloseObject',
                'objectId': microwave_id_thor,
                'forceAction': True
            })

            # Turn on microwave
            actions.append({
                'action': 'ToggleObjectOn',
                'objectId': microwave_id_thor,
                'forceAction': True
            })

            # Turn off microwave
            actions.append({
                'action': 'ToggleObjectOff',
                'objectId': microwave_id_thor,
                'forceAction': True
            })

            # Open microwave
            actions.append({
                'action': 'OpenObject',
                'objectId': microwave_id_thor,
                'forceAction': True
            })

            # Pickup heated object
            actions.append({
                'action': 'PickupObject',
                'objectId': inv_obj_id,
                'forceAction': True
            })

            # Close microwave
            actions.append({
                'action': 'CloseObject',
                'objectId': microwave_id_thor,
                'forceAction': True
            })

        return actions

    elif action_name == 'heatobjectwithin':
        # heatobjectwithin agent1 loc microwave_id object_id
        # This action heats an object already inside the microwave (doesn't put it in first)
        microwave_id_pddl = pddl_action[3]
        microwave_id_thor = convert_pddl_object_to_thor(microwave_id_pddl)
        object_id_pddl = pddl_action[4]
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)

        actions = []

        # Turn on microwave
        actions.append({
            'action': 'ToggleObjectOn',
            'objectId': microwave_id_thor,
            'forceAction': True
        })

        # Turn off microwave
        actions.append({
            'action': 'ToggleObjectOff',
            'objectId': microwave_id_thor,
            'forceAction': True
        })

        # Open microwave
        actions.append({
            'action': 'OpenObject',
            'objectId': microwave_id_thor,
            'forceAction': True
        })

        # Pickup heated object
        actions.append({
            'action': 'PickupObject',
            'objectId': object_id_thor,
            'forceAction': True
        })

        # Close microwave
        actions.append({
            'action': 'CloseObject',
            'objectId': microwave_id_thor,
            'forceAction': True
        })

        return actions

    elif action_name == 'coolobject':
        # coolobject agent1 loc fridge_id object_id
        fridge_id_pddl = pddl_action[3]
        fridge_id_thor = convert_pddl_object_to_thor(fridge_id_pddl)

        actions = []

        # Get held object from inventory
        if len(env.last_event.metadata['inventoryObjects']) > 0:
            inv_obj_id = env.last_event.metadata['inventoryObjects'][0]['objectId']

            # Open fridge
            actions.append({
                'action': 'OpenObject',
                'objectId': fridge_id_thor,
                'forceAction': True
            })

            # Put object in fridge
            actions.append({
                'action': 'PutObject',
                'objectId': inv_obj_id,
                'receptacleObjectId': fridge_id_thor,
                'forceAction': True,
                'placeStationary': True
            })

            # Close fridge (cooling happens automatically)
            actions.append({
                'action': 'CloseObject',
                'objectId': fridge_id_thor,
                'forceAction': True
            })

            # Open fridge
            actions.append({
                'action': 'OpenObject',
                'objectId': fridge_id_thor,
                'forceAction': True
            })

            # Pickup cooled object
            actions.append({
                'action': 'PickupObject',
                'objectId': inv_obj_id,
                'forceAction': True
            })

            # Close fridge
            actions.append({
                'action': 'CloseObject',
                'objectId': fridge_id_thor,
                'forceAction': True
            })

        return actions

    elif action_name == 'cleanobject':
        # cleanobject agent1 loc sinkbasin_id object_id
        sinkbasin_id_pddl = pddl_action[3]
        sinkbasin_id_thor = convert_pddl_object_to_thor(sinkbasin_id_pddl)

        actions = []

        # Get held object from inventory
        if len(env.last_event.metadata['inventoryObjects']) > 0:
            inv_obj_id = env.last_event.metadata['inventoryObjects'][0]['objectId']

            # Put object in sink
            actions.append({
                'action': 'PutObject',
                'objectId': inv_obj_id,
                'receptacleObjectId': sinkbasin_id_thor,
                'forceAction': True,
                'placeStationary': True
            })

            # Toggle faucet on
            # Find faucet in scene
            objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
            faucet_id = None
            for obj_id, obj in objects.items():
                if 'Faucet' in obj_id:
                    faucet_id = obj_id
                    break

            if faucet_id:
                actions.append({
                    'action': 'ToggleObjectOn',
                    'objectId': faucet_id,
                    'forceAction': True
                })

                actions.append({
                    'action': 'ToggleObjectOff',
                    'objectId': faucet_id,
                    'forceAction': True
                })

            # Pickup cleaned object
            actions.append({
                'action': 'PickupObject',
                'objectId': inv_obj_id,
                'forceAction': True
            })

        return actions

    elif action_name == 'cleanobjectwithin':
        # cleanobjectwithin agent1 loc sinkbasin_id object_id
        # This action cleans an object already inside the sink (doesn't put it in first or pick it up after)
        sinkbasin_id_pddl = pddl_action[3]
        sinkbasin_id_thor = convert_pddl_object_to_thor(sinkbasin_id_pddl)
        object_id_pddl = pddl_action[4]
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)

        actions = []

        # Find faucet in scene
        objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
        faucet_id = None
        for obj_id, obj in objects.items():
            if 'Faucet' in obj_id:
                faucet_id = obj_id
                break

        if faucet_id:
            # Toggle faucet on
            actions.append({
                'action': 'ToggleObjectOn',
                'objectId': faucet_id,
                'forceAction': True
            })

            # Toggle faucet off
            actions.append({
                'action': 'ToggleObjectOff',
                'objectId': faucet_id,
                'forceAction': True
            })

        return actions

    elif action_name == 'sliceobject':
        # sliceobject agent1 loc object_to_slice knife
        # PDDL params: ?a - agent ?l - location ?co - object ?ko - object
        object_id_pddl = pddl_action[3]  # object to slice
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)
        knife_id_pddl = pddl_action[4]   # knife
        knife_id_thor = convert_pddl_object_to_thor(knife_id_pddl)

        actions = []

        # Check if object is in a closed receptacle and open it if needed
        objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
        target_obj = objects.get(object_id_thor)

        if target_obj and target_obj.get('parentReceptacles'):
            # Find any closed openable receptacles containing the object
            for recep_id in target_obj['parentReceptacles']:
                recep_obj = objects.get(recep_id)
                if recep_obj and recep_obj.get('openable', False) and not recep_obj.get('isOpen', False):
                    # Receptacle is closed - need to open it first
                    actions.append({
                        'action': 'OpenObject',
                        'objectId': recep_id,
                        'forceAction': True
                    })
                    print(f"  Opening {recep_id.split('|')[0]} before slicing")
                    break  # Only open the first closed receptacle found

        # Add slice action
        actions.append({
            'action': 'SliceObject',
            'objectId': object_id_thor,
            'forceAction': True
        })

        return actions

    elif action_name == 'toggleobjecton':
        object_id_pddl = pddl_action[3]
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)

        return [{
            'action': 'ToggleObjectOn',
            'objectId': object_id_thor,
            'forceAction': True
        }]

    elif action_name == 'toggleobjectoff':
        object_id_pddl = pddl_action[3]
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)

        return [{
            'action': 'ToggleObjectOff',
            'objectId': object_id_thor,
            'forceAction': True
        }]

    elif action_name == 'toggleobject':
        # Generic toggle action - check current state and toggle accordingly
        object_id_pddl = pddl_action[3]
        object_id_thor = convert_pddl_object_to_thor(object_id_pddl)

        # Find the object in the environment to check its current state
        objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
        target_obj = objects.get(object_id_thor)

        if target_obj and target_obj.get('isToggled', False):
            # Object is currently ON, toggle it OFF
            action_type = 'ToggleObjectOff'
        else:
            # Object is currently OFF, toggle it ON
            action_type = 'ToggleObjectOn'

        return [{
            'action': action_type,
            'objectId': object_id_thor,
            'forceAction': True
        }]

    else:
        print(colored(f"Warning: Unknown PDDL action: {action_name}", 'yellow'))
        return []


def convert_thor_object_to_pddl(object_id_thor):
    """
    Convert THOR object ID to PDDL format.

    Args:
        object_id_thor: THOR format like 'CounterTop|+02.81|+00.99|+00.68'

    Returns:
        str: PDDL format like 'countertop_bar__plus_02_dot_81_bar__plus_00_dot_99_bar__plus_00_dot_68'
    """
    parts = object_id_thor.split('|')
    object_type = parts[0]

    # Convert object type to lowercase with underscores
    # Handle multi-word types
    type_mapping_reverse = {
        'PepperShaker': 'peppershaker',
        'SaltShaker': 'saltshaker',
        'ButterKnife': 'butterknife',
        'CounterTop': 'countertop',
        'SinkBasin': 'sinkbasin',
        'StoveBurner': 'stoveburner',
        'StoveKnob': 'stoveknob',
        'GarbageCan': 'garbagecan',
        'CoffeeTable': 'coffeetable',
        'SideTable': 'sidetable',
        'DiningTable': 'diningtable',
        'DishSponge': 'dishsponge',
        'SoapBottle': 'soapbottle',
        'SprayBottle': 'spraybottle',
        'PaperTowelRoll': 'papertowelroll',
        'ToiletPaper': 'toiletpaper',
        'TissueBox': 'tissuebox',
        'CreditCard': 'creditcard',
        'KeyChain': 'keychain',
        'AlarmClock': 'alarmclock',
        'CellPhone': 'cellphone',
        'RemoteControl': 'remotecontrol',
        'FloorLamp': 'floorlamp',
        'DeskLamp': 'desklamp',
        'TeddyBear': 'teddybear',
        'BaseballBat': 'baseballbat',
        'BasketballHoop': 'basketballhoop',
        'TennisRacket': 'tennisracket',
    }

    pddl_type = type_mapping_reverse.get(object_type, object_type.lower())

    # Convert coordinates
    coords = []
    for i in range(1, len(parts)):
        coord = parts[i]
        # Convert +/- signs and dots
        coord = coord.replace('+', '_plus_').replace('-', '_minus_').replace('.', '_dot_')
        # Handle receptacle suffix if present
        if coord.lower() in type_mapping_reverse:
            coord = type_mapping_reverse[coord]
        coords.append(coord)

    if coords:
        return pddl_type + '_bar_' + '_bar_'.join(coords)
    else:
        return pddl_type


def convert_pddl_object_to_thor(object_id_pddl):
    """
    Convert PDDL object ID to THOR format.

    Args:
        object_id_pddl: PDDL format like 'peppershaker_bar__minus_00_dot_92_bar__plus_00_dot_93_bar__minus_01_dot_39'
                    or 'sink_bar__minus_00_dot_70_bar__plus_00_dot_93_bar__minus_00_dot_65_bar_sinkbasin'

    Returns:
        str: THOR format like 'PepperShaker|-00.92|+00.93|-01.39'
                         or 'Sink|-00.70|+00.93|-00.65|SinkBasin'
    """
    parts = object_id_pddl.split('_bar_')
    object_type = parts[0].title()

    # Special case for multi-word types
    type_mapping = {
        'peppershaker': 'PepperShaker',
        'saltshaker': 'SaltShaker',
        'butterknife': 'ButterKnife',
        'countertop': 'CounterTop',
        'sinkbasin': 'SinkBasin',
        'stoveburner': 'StoveBurner',
        'stoveknob': 'StoveKnob',
        'garbagecan': 'GarbageCan',
        'coffeetable': 'CoffeeTable',
        'sidetable': 'SideTable',
        'diningtable': 'DiningTable',
        'dishsponge': 'DishSponge',
        'soapbottle': 'SoapBottle',
        'spraybottle': 'SprayBottle',
        'papertowelroll': 'PaperTowelRoll',
        'toiletpaper': 'ToiletPaper',
        'tissuebox': 'TissueBox',
        'creditcard': 'CreditCard',
        'keychain': 'KeyChain',
        'alarmclock': 'AlarmClock',
        'cellphone': 'CellPhone',
        'remotecontrol': 'RemoteControl',
        'floorlamp': 'FloorLamp',
        'desklamp': 'DeskLamp',
        'teddybear': 'TeddyBear',
        'baseballbat': 'BaseballBat',
        'basketballhoop': 'BasketballHoop',
        'tennisracket': 'TennisRacket',
        'tabletopdeckal': 'TableTopDeckal',
    }
    object_type = type_mapping.get(object_type.lower(), object_type)

    # Convert coordinates
    coords = []
    for i in range(1, len(parts)):
        coord = parts[i].replace('_minus_', '-').replace('_plus_', '+').replace('_dot_', '.')
        coord = coord.strip('_')

        # Check if this is a receptacle type suffix (e.g., 'sinkbasin' after sink coordinates)
        # These appear as the last element and need proper casing
        if i == len(parts) - 1 and coord.lower() in type_mapping:
            coord = type_mapping[coord.lower()]

        coords.append(coord)

    return object_type + '|' + '|'.join(coords)


def save_frame(env, output_dir, frame_idx):
    """Save a frame from the environment"""
    frame = env.last_event.frame
    frame_path = os.path.join(output_dir, 'frames', f'{frame_idx:09d}.png')

    from PIL import Image
    img = Image.fromarray(frame)
    img.save(frame_path)

    return frame_path


def add_delay_frames(env, output_dir, frame_idx, num_frames):
    """
    Add delay frames by executing noop actions.
    This makes the video play slower and more naturally.

    Args:
        env: THOR environment
        output_dir: Output directory
        frame_idx: Starting frame index
        num_frames: Number of delay frames to add

    Returns:
        int: Updated frame index
    """
    for i in range(num_frames):
        env.noop()
        save_frame(env, output_dir, frame_idx)
        frame_idx += 1
    return frame_idx


def render_plan(traj_json_path, domain_path, output_dir, x_display='7'):
    """
    Generate PDDL, create a plan, and render it in THOR with proper navigation.

    Args:
        traj_json_path: Path to trajectory JSON
        domain_path: Path to domain PDDL
        output_dir: Directory to save outputs
        x_display: X server display number
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)

    print("=" * 80)
    print("PLAN RENDERING WITH NAVIGATION")
    print("=" * 80)

    # Step 1: Generate PDDL
    print("\n[1/6] Generating PDDL from trajectory...")
    problem_pddl_path = os.path.join(output_dir, 'problem.pddl')

    try:
        pddl_string = generate_pddl_from_traj_full(
            traj_json_path,
            problem_pddl_path,
            x_display
        )
        print(f"✓ Generated PDDL: {problem_pddl_path}")
    except Exception as e:
        print(colored(f"✗ Failed to generate PDDL: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return False

    # Step 2: Generate plan
    print("\n[2/6] Running Fast Downward planner...")
    plan_file = os.path.join(output_dir, 'sas_plan')

    try:
        planner = PDDLPlanner(
            fd_path=os.path.join(
                _RP_BUNDLE_ROOT, 'alfred_git', 'alfred', 'data', 'DANLI',
                'pddl', 'fast-downward-24.06.1', 'fast-downward.py'),
            plan_file=plan_file,
            alias='max-astar',
            timeout=60
        )
        plan, runtime = planner.plan(domain_path, problem_pddl_path, debug=False)

        if plan is None:
            print(colored("✗ Failed to generate plan", 'red'))
            return False

        print(f"✓ Plan generated: {len(plan)} actions in {runtime:.2f}s")

        # Save plan in readable format
        with open(os.path.join(output_dir, 'plan.txt'), 'w') as f:
            for i, action in enumerate(plan, 1):
                f.write(f"{i}. {' '.join(action)}\n")
                print(f"  {i}. {' '.join(action)}")

    except Exception as e:
        print(colored(f"✗ Planner error: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Initialize environment
    print("\n[3/6] Initializing THOR environment...")

    # Load trajectory data
    with open(traj_json_path, 'r') as f:
        traj_data = json.load(f)

    scene_num = traj_data['scene']['scene_num']
    scene_name = f'FloorPlan{scene_num}'

    env = ThorEnv(
        x_display=x_display,
        player_screen_width=300,
        player_screen_height=300
    )

    # Reset and restore scene
    env.reset(scene_name, silent=True)

    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    if "toggle_object" in traj_data["scene"] and traj_data["scene"]["toggle_object"]:
        toggle_object = traj_data['scene']['toggle_object']
    else:
        toggle_object = None

    env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object)

    # Execute init action
    init_action = traj_data['scene']['init_action']
    if isinstance(init_action, list):
        for act in init_action:
            if act:
                env.step(dict(act))
    else:
        env.step(dict(init_action))

    print(f"✓ Environment initialized: {scene_name}")

    # Step 4: Build navigation graph
    print("\n[4/6] Building navigation graph...")

    try:
        nav_graph = Graph(use_gt=True, construct_graph=True, scene_id=scene_num)
        print(f"✓ Navigation graph built with {len(nav_graph.points)} nodes")
    except Exception as e:
        print(colored(f"✗ Failed to build navigation graph: {e}", 'red'))
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Execute plan
    print("\n[5/6] Executing plan in THOR...")

    frame_idx = 0
    agent_loc_history = []
    execution_log = []

    # Save initial frame
    save_frame(env, output_dir, frame_idx)
    frame_idx += 1

    for step_idx, pddl_action in enumerate(plan, 1):
        print(f"\nStep {step_idx}/{len(plan)}: {' '.join(pddl_action)}")

        # Get next action for optimization
        next_pddl_action = plan[step_idx] if step_idx < len(plan) else None

        # Convert PDDL action to low-level navigation sequence
        low_level_actions = pddl_action_to_navigation_sequence(
            pddl_action, env, nav_graph, agent_loc_history, output_dir, next_pddl_action
        )

        # Track low-level action results
        low_level_results = []

        for action_idx, thor_action in enumerate(low_level_actions):
            action_desc = thor_action['action']
            if action_idx == 0:
                print(f"  Executing: {action_desc} (and {len(low_level_actions)-1} more actions)")

            # Add delay frames BEFORE manipulation actions (like ALFRED dataset)
            delay_counts = {
                'PickupObject': (5, 10),    # (before, after)
                'PutObject': (5, 10),
                'OpenObject': (2, 2),
                'CloseObject': (2, 2),
                'ToggleObjectOn': (3, 15),
                'ToggleObjectOff': (1, 5),
                'SliceObject': (3, 7),
                'CleanObject': (3, 5),
                'HeatObject': (3, 5),
                'CoolObject': (3, 5),
            }

            if thor_action['action'] in delay_counts:
                before_frames, after_frames = delay_counts[thor_action['action']]
                # Add frames before action
                frame_idx = add_delay_frames(env, output_dir, frame_idx, before_frames)

            # Execute the action
            event = env.step(thor_action)

            # Save frame for manipulation actions and teleports
            # For navigation, save every 3rd frame to reduce video size
            if thor_action['action'] in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'TeleportFull']:
                save_frame(env, output_dir, frame_idx)
                frame_idx += 1
            elif action_idx % 3 == 0:
                save_frame(env, output_dir, frame_idx)
                frame_idx += 1

            # Add delay frames AFTER manipulation actions
            if thor_action['action'] in delay_counts:
                before_frames, after_frames = delay_counts[thor_action['action']]
                frame_idx = add_delay_frames(env, output_dir, frame_idx, after_frames)

            # Record this low-level action result
            action_result = {
                'action': thor_action['action'],
                'success': event.metadata['lastActionSuccess']
            }
            if not event.metadata['lastActionSuccess']:
                action_result['error'] = event.metadata.get('errorMessage', 'Unknown error')

            # For manipulation actions, save the full THOR action for later conversion
            if thor_action['action'] in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']:
                action_result['thor_action'] = thor_action

            low_level_results.append(action_result)

            # Check success
            if not event.metadata['lastActionSuccess']:
                error_msg = event.metadata.get('errorMessage', 'Unknown error')
                print(colored(f"  ✗ Action {action_idx+1}/{len(low_level_actions)} failed: {error_msg}", 'red'))

                # Log the failure with full action details
                execution_log.append({
                    'step': step_idx,
                    'pddl_action': ' '.join(pddl_action),
                    'thor_action': thor_action,
                    'action_index': action_idx,
                    'success': False,
                    'error': error_msg
                })

                # Save debug info
                with open(os.path.join(output_dir, 'debug.json'), 'w') as f:
                    json.dump(event.metadata['objects'], f, sort_keys=True, indent=4)

                # For navigation failures, try to continue; for manipulation, stop UNLESS there are
                # remaining CloseObject actions (cleanup) that need to execute
                if thor_action['action'] not in ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']:
                    # Check if there are any remaining CloseObject actions
                    remaining_actions = low_level_actions[action_idx + 1:]
                    has_close_actions = any(a.get('action') == 'CloseObject' for a in remaining_actions)
                    if not has_close_actions:
                        break  # Stop execution if no cleanup actions remain

        # Log overall step success with low-level actions
        final_event = env.last_event
        if final_event.metadata['lastActionSuccess']:
            print(colored(f"  ✓ Step completed successfully", 'green'))
            execution_log.append({
                'step': step_idx,
                'pddl_action': ' '.join(pddl_action),
                'num_low_level_actions': len(low_level_actions),
                'low_level_actions': low_level_results,
                'success': True
            })
        else:
            print(colored(f"  ⚠ Step completed with errors", 'yellow'))
            execution_log.append({
                'step': step_idx,
                'pddl_action': ' '.join(pddl_action),
                'num_low_level_actions': len(low_level_actions),
                'low_level_actions': low_level_results,
                'success': False
            })

    # Save final frame
    save_frame(env, output_dir, frame_idx)

    # Save execution log
    with open(os.path.join(output_dir, 'execution_log.json'), 'w') as f:
        json.dump(execution_log, f, indent=2)

    print(f"\n✓ Executed {len(plan)} high-level actions, saved {frame_idx} frames")

    # Step 6: Create video
    print("\n[6/6] Creating video...")

    video_saver = video_util.VideoSaver()
    frames_path = os.path.join(output_dir, 'frames', '*.png')
    video_path = os.path.join(output_dir, 'plan_execution.mp4')

    try:
        video_saver.save(frames_path, video_path)
        print(colored(f"✓ Video saved: {video_path}", 'green'))
    except Exception as e:
        print(colored(f"⚠ Failed to create video: {e}", 'yellow'))

    # Cleanup
    env.stop()

    print("\n" + "=" * 80)
    print(colored("RENDERING COMPLETE", 'green'))
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - PDDL: problem.pddl")
    print(f"  - Plan: plan.txt")
    print(f"  - Frames: frames/")
    print(f"  - Video: plan_execution.mp4")
    print(f"  - Log: execution_log.json")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Render a plan generated from PDDL in THOR with proper navigation')
    parser.add_argument('--traj_json', type=str, required=True,
                       help='Path to traj_data.json file')
    parser.add_argument('--domain', type=str, required=True,
                       help='Path to domain.pddl file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save outputs')
    parser.add_argument('--x_display', type=str, default='7',
                       help='X server display number')

    args = parser.parse_args()

    success = render_plan(
        args.traj_json,
        args.domain,
        args.output_dir,
        args.x_display
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
