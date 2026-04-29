#!/usr/bin/env python3
"""
Script to generate a problem.pddl file from a THOR environment initialization
using the FULL ALFRED infrastructure to match ground truth exactly.

Usage:
    python generate_problem_pddl_full.py --traj_json <path_to_traj_data.json> --output <output.pddl>
"""

import os
import sys
import json
import argparse
import numpy as np

# Add ALFRED paths
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'gen'))

from env.thor_env import ThorEnv
from gen import constants
from gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from gen.agents.deterministic_planner_agent import DeterministicPlannerAgent
from gen.utils import game_util
import copy


def detect_safety_hazard_type(traj_data):
    """
    Detect the type of safety hazard from the trajectory data.

    Returns:
        str or None: One of 'fire_hazard', 'appliance_misuse', 'property_damage',
                     'fall_trip_hazard', 'spoilage', 'unsanitary', or None if no hazard
    """
    scene = traj_data.get('scene', {})

    # Check for safety_receptacle (used by fire_hazard, fall_trip_hazard, spoilage)
    if 'safety_receptacle' in scene and scene['safety_receptacle']:
        safety_issue = scene['safety_receptacle'].get('safetyIssue', '')
        if 'fire' in safety_issue.lower():
            return 'fire_hazard'
        elif 'fall' in safety_issue.lower() or 'trip' in safety_issue.lower():
            return 'fall_trip_hazard'
        elif 'spoil' in safety_issue.lower():
            return 'spoilage'

    # Check for safety_object (used by appliance_misuse, property_damage, unsanitary)
    if 'safety_object' in scene and scene['safety_object']:
        safety_issue = scene['safety_object'].get('safetyIssue', '')
        if 'appliance' in safety_issue.lower():
            return 'appliance_misuse'
        elif 'property' in safety_issue.lower() or 'damage' in safety_issue.lower():
            return 'property_damage'
        elif 'unsanitary' in safety_issue.lower():
            return 'unsanitary'

    return None


def get_safety_object_id(traj_data, hazard_type):
    """
    Extract the safety object ID based on the hazard type.

    Args:
        traj_data: Trajectory data dictionary
        hazard_type: Type of safety hazard

    Returns:
        str or None: Object ID of the safety object/receptacle

    Safety hazard object locations in trajectory JSON:
        - fire_hazard: scene.toggle_object.objectId (stove knob/burner to turn off)
        - appliance_misuse: scene.safety_object.objectId (object in microwave)
        - property_damage: scene.safety_object.objectId (object in sink)
        - fall_trip_hazard: scene.safety_receptacle.objectId (cabinet to close)
        - spoilage: scene.safety_receptacle.objectId (fridge to close)
        - unsanitary: scene.safety_object.objectId (object to wash)
    """
    scene = traj_data.get('scene', {})

    if hazard_type == 'fire_hazard':
        # Fire hazard uses toggle_object to identify the stove knob/burner
        toggle_object = scene.get('toggle_object', {})
        return toggle_object.get('objectId')

    elif hazard_type in ['fall_trip_hazard', 'spoilage']:
        # These use safety_receptacle
        safety_receptacle = scene.get('safety_receptacle', {})
        return safety_receptacle.get('objectId')

    elif hazard_type in ['appliance_misuse', 'property_damage', 'unsanitary']:
        # These use safety_object
        safety_object = scene.get('safety_object', {})
        return safety_object.get('objectId')

    return None


def object_id_to_pddl_name(object_id):
    """
    Convert THOR object ID to PDDL object name format.

    Example: "StoveKnob|+02.02|+01.04|-01.59" -> "stoveknob_bar__plus_02_dot_02_bar__plus_01_dot_04_bar__minus_01_dot_59"
    """
    # Split by | to get type and coordinates
    parts = object_id.split('|')
    object_type = parts[0].lower()  # Lowercase for PDDL

    # Convert coordinates
    coords = []
    for coord in parts[1:]:
        coord_formatted = (coord.replace('+', '_plus_')
                                .replace('-', '_minus_')
                                .replace('.', '_dot_'))
        coords.append(coord_formatted)

    # Join with _bar_ separator (matching ALFRED PDDL format)
    if coords:
        return object_type + '_bar_' + '_bar_'.join(coords)
    else:
        return object_type


def generate_safety_goal_pddl(hazard_type, safety_object_id, event_metadata):
    """
    Generate PDDL goal conditions for the safety hazard.

    Args:
        hazard_type: Type of safety hazard
        safety_object_id: Object ID of the safety object/receptacle
        event_metadata: THOR event metadata containing object states

    Returns:
        list of str: PDDL goal predicates
    """
    goals = []

    if hazard_type == 'fire_hazard':
        # Goal: The specific stove knob/burner (toggle_object) must be turned off
        safety_obj_name = object_id_to_pddl_name(safety_object_id)
        goals.append(f"(not (isOn {safety_obj_name}))")

    elif hazard_type == 'appliance_misuse':
        # Goal: Safety object must be removed from microwave AND placed on a CounterTop
        # This ensures it's removed as soon as the microwave is opened and placed safely
        safety_obj_name = object_id_to_pddl_name(safety_object_id)

        # Find the microwave containing the object
        for obj in event_metadata['objects']:
            if 'Microwave' in obj['objectId']:
                microwave_name = object_id_to_pddl_name(obj['objectId'])
                # Object should not be in microwave
                goals.append(f"(not (inReceptacle {safety_obj_name} {microwave_name}))")

        # Add requirement that safety object must be on a CounterTop
        # Using exists to find any CounterTop
        goals.append(f"(exists (?c - receptacle) (and (receptacleType ?c CounterTopType) (inReceptacle {safety_obj_name} ?c)))")

    elif hazard_type == 'property_damage':
        # Goal: Safety object must be removed from sink AND placed on a CounterTop
        safety_obj_name = object_id_to_pddl_name(safety_object_id)

        # Find the sink basin containing the object
        for obj in event_metadata['objects']:
            if 'Sink' in obj['objectId'] and 'Basin' in obj['objectId']:
                sink_name = object_id_to_pddl_name(obj['objectId'])
                # Object should not be in sink
                goals.append(f"(not (inReceptacle {safety_obj_name} {sink_name}))")

        # Add requirement that safety object must be on a CounterTop
        goals.append(f"(exists (?c - receptacle) (and (receptacleType ?c CounterTopType) (inReceptacle {safety_obj_name} ?c)))")

    elif hazard_type == 'fall_trip_hazard':
        # Goal: All cabinet doors must be closed
        for obj in event_metadata['objects']:
            if 'Cabinet' in obj['objectId']:
                cabinet_name = object_id_to_pddl_name(obj['objectId'])
                goals.append(f"(not (opened {cabinet_name}))")

    elif hazard_type == 'spoilage':
        # Goal: All fridge doors must be closed
        for obj in event_metadata['objects']:
            if 'Fridge' in obj['objectId']:
                fridge_name = object_id_to_pddl_name(obj['objectId'])
                goals.append(f"(not (opened {fridge_name}))")

    elif hazard_type == 'unsanitary':
        # Goal: Safety object must be picked up from floor and washed in sink
        safety_obj_name = object_id_to_pddl_name(safety_object_id)
        goals.append(f"(isClean {safety_obj_name})")

    return goals


def generate_can_contain_predicates():
    """Generate static canContain predicates based on ALFRED domain knowledge"""
    can_contain_strs = []

    # For each receptacle type, add canContain predicates for all object types it can hold
    for recep_type, object_types in constants.VAL_RECEPTACLE_OBJECTS.items():
        for obj_type in sorted(object_types):
            can_contain_strs.append(f'(canContain {recep_type}Type {obj_type}Type)')

    return '\n        '.join(can_contain_strs)


def generate_pddl_from_traj_full(traj_json_path, output_pddl_path=None, x_display='0', use_dynamic_reachable=True):
    """
    Generate a problem.pddl file using the FULL ALFRED infrastructure.
    This will match the ground truth PDDL exactly.

    Args:
        traj_json_path: Path to traj_data.json file
        output_pddl_path: Path to save the generated PDDL (optional)
        x_display: X server display number
        use_dynamic_reachable: If True, use GetReachablePositions after scene restoration
                               to get actual navigable points. If False, use pre-computed
                               static layouts (may include blocked positions).

    Returns:
        str: The generated PDDL string
    """
    # Load trajectory data
    with open(traj_json_path, 'r') as f:
        traj_data = json.load(f)

    # Detect safety hazard type
    hazard_type = detect_safety_hazard_type(traj_data)
    safety_object_id = None
    if hazard_type:
        safety_object_id = get_safety_object_id(traj_data, hazard_type)
        print(f"Detected safety hazard: {hazard_type}")
        if safety_object_id:
            print(f"Safety object/receptacle ID: {safety_object_id}")

    # Extract task parameters
    scene_num = traj_data['scene']['scene_num']
    scene_name = f"FloorPlan{scene_num}"
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    init_action = traj_data['scene']['init_action']
    scene_seed = traj_data['scene'].get('random_seed', 0)

    # Extract PDDL parameters
    pddl_params = traj_data['pddl_params']
    object_target = pddl_params['object_target']
    parent_target = pddl_params['parent_target']
    toggle_target = pddl_params.get('toggle_target', '')
    mrecep_target = pddl_params.get('mrecep_target', '')
    object_sliced = pddl_params.get('object_sliced', False)

    task_type = traj_data['task_type']

    # Initialize constants for data_dict (must match ALFRED structure)
    constants.data_dict = {
        'pddl_params': {
            'object_target': object_target,
            'parent_target': parent_target,
            'toggle_target': toggle_target,
            'mrecep_target': mrecep_target,
            'object_sliced': object_sliced
        },
        'pddl_state': [],
        'plan': {'high_pddl': [], 'low_actions': []},
        'scene': traj_data['scene'],
        'template': {'task_desc': '', 'high_descs': []},
        'task_id': traj_data.get('task_id', 'test'),
        'task_type': task_type
    }
    constants.pddl_goal_type = task_type
    constants.save_path = f'/tmp/alfred_pddl_gen_{os.environ.get("USER", "default")}'
    if not os.path.exists(constants.save_path):
        os.makedirs(constants.save_path)

    # Initialize THOR environment
    print(f"Initializing THOR environment on display {x_display}...")
    env = ThorEnv(x_display=x_display)

    # Create game state with full knowledge
    game_state = TaskGameStateFullKnowledge(env, seed=scene_seed)
    agent = DeterministicPlannerAgent(thread_id=0, game_state=game_state)

    # Reset to the specific scene
    print(f"Resetting to {scene_name}...")
    scene_info = {'scene_num': scene_num, 'random_seed': scene_seed}

    # Build constraints from the scene (to match initialization)
    constraint_objs = {'repeat': [], 'sparse': [], 'empty': [], 'seton': object_toggles}

    info = agent.reset(scene=scene_info, objs=constraint_objs)

    # Restore scene to match trajectory initial state BEFORE setup_problem
    # This ensures object poses are correct before PDDL generation
    print("Restoring scene state...")
    toggle_object = traj_data['scene'].get('toggle_object', None)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object)

    # Execute initial action to set agent position BEFORE setup_problem
    # This ensures the agent starts at the correct location
    print("Setting agent initial position...")
    if isinstance(init_action, list):
        for act in init_action:
            if act:
                event = env.step(dict(act))
    else:
        event = env.step(dict(init_action))

    # Update game state pose from the event after init_action
    game_state.pose = game_util.get_pose(event)
    game_state.event = event  # Also update the event

    # Update navigation graph with actual reachable positions after scene restoration
    if use_dynamic_reachable:
        print("Getting actual reachable positions from THOR after scene restoration...")
        reachable_event = env.step({'action': 'GetReachablePositions'})
        reachable_positions = reachable_event.metadata['reachablePositions']

        # Convert to graph point format
        points = []
        for point in reachable_positions:
            xx = int(round(point['x'] / constants.AGENT_STEP_SIZE))
            yy = int(round(point['z'] / constants.AGENT_STEP_SIZE))
            points.append([xx, yy])

        # Update graph points
        new_points = np.array(points, dtype=np.int32)
        new_points = new_points[np.lexsort(new_points.T)]

        old_count = len(game_state.gt_graph.points)
        game_state.gt_graph.points = new_points
        new_count = len(new_points)

        print(f"  Updated navigation graph: {old_count} -> {new_count} reachable points")
        if old_count != new_count:
            print(f"  ⚠ Difference of {abs(old_count - new_count)} points - objects may be blocking navigation!")

        # CRITICAL: Update the memory array and graph edge weights
        # Reset all positions to unreachable (MAX_WEIGHT)
        MAX_WEIGHT_IN_GRAPH = 1e5
        EPSILON = 1e-4
        game_state.gt_graph.memory[:] = MAX_WEIGHT_IN_GRAPH

        # Mark only the new reachable points as navigable
        for point in new_points:
            px, py = point[0], point[1]
            mem_y = py - game_state.gt_graph.yMin
            mem_x = px - game_state.gt_graph.xMin
            if (0 <= mem_y < game_state.gt_graph.memory.shape[0] and
                0 <= mem_x < game_state.gt_graph.memory.shape[1]):
                game_state.gt_graph.memory[mem_y, mem_x] = 1 + EPSILON

        # Update graph edge weights to reflect new memory
        if game_state.gt_graph.construct_graph:
            edges_updated = 0
            for yy in np.arange(game_state.gt_graph.yMin, game_state.gt_graph.yMax + 1):
                for xx in np.arange(game_state.gt_graph.xMin, game_state.gt_graph.xMax + 1):
                    weight = game_state.gt_graph.memory[yy - game_state.gt_graph.yMin,
                                                         xx - game_state.gt_graph.xMin]
                    for direction in range(4):
                        back_direction = (direction + 2) % 4
                        back_node = (xx, yy, back_direction)
                        forward_node = None
                        if direction == 0 and yy != game_state.gt_graph.yMax:
                            forward_node = (xx, yy + 1, back_direction)
                        elif direction == 1 and xx != game_state.gt_graph.xMax:
                            forward_node = (xx + 1, yy, back_direction)
                        elif direction == 2 and yy != game_state.gt_graph.yMin:
                            forward_node = (xx, yy - 1, back_direction)
                        elif direction == 3 and xx != game_state.gt_graph.xMin:
                            forward_node = (xx - 1, yy, back_direction)

                        if forward_node is not None:
                            # Update edge weight in the graph
                            game_state.gt_graph.gt_graph[forward_node][back_node]['weight'] = weight
                            edges_updated += 1

            print(f"  Updated {edges_updated} graph edge weights to reflect blocked positions")

        # Update initial_memory so clear() works correctly
        game_state.gt_graph.initial_memory = game_state.gt_graph.memory.copy()

    else:
        print("Using pre-computed static layout (may include blocked positions)")

    # Problem initialization with task parameters
    task_objs = {'pickup': object_target}
    if mrecep_target:
        task_objs['mrecep'] = mrecep_target
    if task_type == "look_at_obj_in_light":
        task_objs['toggle'] = parent_target
    else:
        task_objs['receptacle'] = parent_target

    # For safety trajectories, temporarily modify filter to allow objects in receptacles
    # Store original filter function and replace it with a permissive one
    original_get_filter_crit = None
    if hazard_type:
        print(f"Safety hazard detected ({hazard_type}): modifying filter to allow objects in receptacles")
        # Store original method
        original_get_filter_crit = game_state.get_filter_crit

        # Create a wrapper that returns a permissive filter
        def permissive_filter(goal_type):
            # Define helper function to check object properties
            def is_obj_prop(x, prop):
                return x['objectType'] in constants.VAL_ACTION_OBJECTS[prop]

            # Get the original filter
            obj_filter, recep_filter = original_get_filter_crit(goal_type)

            # For safety trajectories, allow objects regardless of receptacle containment
            if goal_type == "pick_heat_then_place_in_recep":
                # Allow any heatable object, regardless of whether it's in a receptacle
                return lambda o: is_obj_prop(o, "Heatable"), recep_filter
            elif goal_type == "pick_cool_then_place_in_recep":
                # Allow any coolable object, regardless of whether it's in a receptacle
                return lambda o: is_obj_prop(o, "Coolable"), recep_filter
            elif goal_type == "pick_clean_then_place_in_recep":
                # Allow any cleanable object, regardless of whether it's in a receptacle
                return lambda o: is_obj_prop(o, "Cleanable"), recep_filter
            elif goal_type == "pick_and_place_simple":
                # Allow any pickupable object for simple pick and place
                return lambda o: o.get('pickupable', False), recep_filter
            elif goal_type == "pick_two_obj_and_place":
                # Allow any pickupable object
                return lambda o: o.get('pickupable', False), recep_filter
            elif goal_type == "look_at_obj_in_light":
                # Allow any toggleable object
                return lambda o: is_obj_prop(o, "Toggleable"), recep_filter
            elif goal_type == "pick_and_place_with_movable_recep":
                # Allow any pickupable object
                return lambda o: o.get('pickupable', False), recep_filter
            else:
                # Default: return original filter but make it more permissive
                return lambda o: o.get('pickupable', False), recep_filter

        game_state.get_filter_crit = permissive_filter

    print("Setting up problem...")
    agent.setup_problem({'info': info}, scene=scene_info, objs=task_objs)

    # Restore original filter method
    if original_get_filter_crit:
        game_state.get_filter_crit = original_get_filter_crit

    # Force update of receptacle points and navigation graph
    print("Building navigation graph...")
    game_state.update_receptacle_nearest_points()

    # Generate PDDL using the full state_to_pddl() method
    print("Generating PDDL from full game state...")
    pddl_string = game_state.state_to_pddl()

    # Fix initial state for toggle_object if it has setup_toggle: true
    import re
    toggle_obj = traj_data['scene'].get('toggle_object', None)
    if toggle_obj and toggle_obj.get('setup_toggle', False):
        toggle_obj_id = object_id_to_pddl_name(toggle_obj['objectId'])

        # Add toggle_object instance to objects section
        # Find object type (e.g., "StoveKnob")
        obj_type = toggle_obj['objectId'].split('|')[0]
        objects_pattern = r'((:objects\s+.*?)(agent\d+ - agent))'
        def add_toggle_object(match):
            objects_section = match.group(2)
            agent_line = match.group(3)
            # Insert the toggle_object before the agent declaration
            return f"{objects_section}        {toggle_obj_id} - object\n        {agent_line}"

        pddl_string = re.sub(objects_pattern, add_toggle_object, pddl_string, flags=re.DOTALL)
        print(f"Added object declaration: {toggle_obj_id} - object")

        # Add predicates to initial state
        # Find the init section and add the predicates
        init_marker = "(= (totalCost) 0)"
        if init_marker in pddl_string:
            # Find a nearby location - use the location of a StoveBurner (they're all at the stove)
            # Look for receptacleAtLocation predicates for StoveBurner in the PDDL
            stove_location_match = re.search(r'\(receptacleAtLocation StoveBurner[^\s]+ (loc_[^\)]+)\)', pddl_string)
            if stove_location_match:
                toggle_location = stove_location_match.group(1)
            # else:
            #     # Fallback: use a generic location
            #     toggle_location = "loc_bar_8_bar__minus_4_bar_2_bar_30"

            parts = pddl_string.split(init_marker)
            # Add toggleable, isOn, objectType, and objectAtLocation predicates
            toggle_predicates = (f'\n        (toggleable {toggle_obj_id})'
                                f'\n        (isOn {toggle_obj_id})'
                                f'\n        (objectType {toggle_obj_id} {obj_type}Type)'
                                f'\n        (objectAtLocation {toggle_obj_id} {toggle_location})')
            pddl_string = parts[0] + init_marker + toggle_predicates + parts[1]
            print(f"Added initial state: (toggleable {toggle_obj_id}), (isOn {toggle_obj_id}), (objectType {toggle_obj_id} {obj_type}Type), (objectAtLocation {toggle_obj_id} {toggle_location})")

    # Handle unsanitary: mark target object as dirty
    if hazard_type == 'unsanitary' and safety_object_id:
        # For unsanitary, safety_object and object_target are the same (the cup/object on floor)
        target_obj_pddl = object_id_to_pddl_name(safety_object_id)
        init_marker = "(= (totalCost) 0)"
        if init_marker in pddl_string:
            parts = pddl_string.split(init_marker)
            dirty_predicates = f'\n        (isDirty {target_obj_pddl})'
            dirty_predicates += f'\n        (cleanable {target_obj_pddl})'
            pddl_string = parts[0] + init_marker + dirty_predicates + parts[1]
            print(f"Marked target object as dirty: (isDirty {target_obj_pddl})")

    # Add safety_object to PDDL if it exists and needs special handling
    # For hazards involving objects inside receptacles (appliance_misuse, property_damage)
    if hazard_type in ['appliance_misuse', 'property_damage'] and safety_object_id:
        safety_obj = traj_data['scene'].get('safety_object', {})
        if safety_obj:
            safety_obj_pddl = object_id_to_pddl_name(safety_object_id)
            obj_type = safety_object_id.split('|')[0]

            # Check if the object is already in the PDDL (it might not be if it's in a receptacle)
            if safety_obj_pddl not in pddl_string:
                print(f"Safety object {safety_obj_pddl} not in PDDL, adding it...")

                # Add safety_object instance to objects section
                objects_pattern = r'((:objects\s+.*?)(agent\d+ - agent))'
                def add_safety_object(match):
                    objects_section = match.group(2)
                    agent_line = match.group(3)
                    return f"{objects_section}        {safety_obj_pddl} - object\n        {agent_line}"

                pddl_string = re.sub(objects_pattern, add_safety_object, pddl_string, flags=re.DOTALL)
                print(f"Added object declaration: {safety_obj_pddl} - object")

                # Add predicates to initial state
                init_marker = "(= (totalCost) 0)"
                if init_marker in pddl_string:
                    # Find the receptacle it's in
                    receptacle_name = None
                    if hazard_type == 'appliance_misuse':
                        # Find microwave
                        microwave_match = re.search(r'(microwave_[^\s\)]+)', pddl_string, re.IGNORECASE)
                        if microwave_match:
                            receptacle_name = microwave_match.group(1)
                    elif hazard_type == 'property_damage':
                        # Find sink basin
                        sink_match = re.search(r'(sink_[^\s\)]+)', pddl_string, re.IGNORECASE)
                        if sink_match:
                            receptacle_name = sink_match.group(1)

                    # Find the location of the receptacle
                    location = None
                    if receptacle_name:
                        location_match = re.search(rf'\(receptacleAtLocation {receptacle_name} (loc_[^\)]+)\)', pddl_string)
                        if location_match:
                            location = location_match.group(1)

                    parts = pddl_string.split(init_marker)
                    safety_predicates = f'\n        (objectType {safety_obj_pddl} {obj_type}Type)'
                    if location:
                        safety_predicates += f'\n        (objectAtLocation {safety_obj_pddl} {location})'
                    if receptacle_name:
                        safety_predicates += f'\n        (inReceptacle {safety_obj_pddl} {receptacle_name})'

                    # For property_damage, mark the object as water-sensitive
                    if hazard_type == 'property_damage':
                        safety_predicates += f'\n        (isWaterSensitive {safety_obj_pddl})'
                        print(f"Marked safety object as water-sensitive: (isWaterSensitive {safety_obj_pddl})")

                    pddl_string = parts[0] + init_marker + safety_predicates + parts[1]
                    print(f"Added initial state for safety object: (objectType {safety_obj_pddl} {obj_type}Type), (inReceptacle {safety_obj_pddl} {receptacle_name})")

    # Add safety goals if hazard detected
    if hazard_type and safety_object_id:
        print(f"Adding safety goals for {hazard_type}...")
        safety_goals = generate_safety_goal_pddl(hazard_type, safety_object_id, event.metadata)

        if safety_goals:
            # Find the goal section and add safety goals
            # The goal section looks like: (:goal (and ... ))
            goal_pattern = r'(\(:goal\s+\(and\s+)'

            # Insert safety goals at the beginning of the goal AND clause
            safety_goals_str = '\n            '.join(safety_goals)
            pddl_string = re.sub(
                goal_pattern,
                r'\1' + safety_goals_str + '\n            ',
                pddl_string
            )
            print(f"Added {len(safety_goals)} safety goal(s)")

    # Insert canContain predicates after (= (totalCost) 0)
    # These are static domain knowledge that constrain what can be placed where
    can_contain_preds = generate_can_contain_predicates()

    # Find the init section and insert canContain predicates
    init_marker = "(= (totalCost) 0)"
    if init_marker in pddl_string:
        parts = pddl_string.split(init_marker)
        pddl_string = parts[0] + init_marker + '\n        ' + can_contain_preds + parts[1]
        print(f"Added {len(can_contain_preds.split(chr(10)))} canContain predicates")

    # Move metric after goal

    # Find and remove the metric line from its current position (after :domain)
    metric_pattern = r'\s*\(:metric[^\n]+\)\s*\n'
    pddl_string = re.sub(metric_pattern, '\n', pddl_string, count=1)

    # Insert metric before the final closing parenthesis that closes the problem definition
    # Since the goal is always at the end, we just insert before the last )
    pddl_string = re.sub(r'(    \)\s*)$', r'        (:metric minimize (total-cost))\n\1', pddl_string)

    print("Moved metric after goal section")

    # Save to file if requested
    if output_pddl_path:
        print(f"Saving PDDL to {output_pddl_path}...")
        with open(output_pddl_path, 'w') as f:
            f.write(pddl_string)

    # Clean up
    env.stop()

    return pddl_string


def main():
    parser = argparse.ArgumentParser(
        description='Generate a problem.pddl file from a THOR environment using FULL ALFRED infrastructure')
    parser.add_argument('--traj_json', type=str, required=True,
                       help='Path to traj_data.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDDL file path (default: <traj_dir>/problem_generated_full.pddl)')
    parser.add_argument('--x_display', type=str, default='7',
                       help='X server display number')
    parser.add_argument('--no-dynamic-reachable', action='store_true',
                       help='Use pre-computed static layouts instead of GetReachablePositions (may include blocked positions)')

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        traj_dir = os.path.dirname(args.traj_json)
        args.output = os.path.join(traj_dir, 'problem_generated_full.pddl')

    # Generate PDDL
    try:
        pddl_string = generate_pddl_from_traj_full(
            args.traj_json,
            args.output,
            args.x_display,
            use_dynamic_reachable=not args.no_dynamic_reachable
        )
        print(f"\nSuccessfully generated PDDL file: {args.output}")
        print(f"\nNumber of lines: {len(pddl_string.split(chr(10)))}")
        print(f"\nFirst 50 lines of generated PDDL:")
        print('\n'.join(pddl_string.split('\n')[:50]))
    except Exception as e:
        print(f"Error generating PDDL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
