#!/usr/bin/env python3
"""
Script to generate a problem.pddl file from a THOR environment initialization.

Usage:
    python generate_problem_pddl.py --traj_json <path_to_traj_data.json> --output <output.pddl>

Or use as a library:
    from generate_problem_pddl import generate_pddl_from_env, generate_pddl_from_traj
"""

import os
import sys
import json
import copy
import argparse
import numpy as np

# Add ALFRED paths
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'gen'))

from alfred.env.thor_env import ThorEnv
from alfred.gen import constants
from alfred.gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from alfred.gen.agents.deterministic_planner_agent import DeterministicPlannerAgent
from alfred.gen.utils import game_util
from alfred.gen.utils import py_util


def fix_pddl_str_chars(input_str):
    """Convert special characters for PDDL compatibility"""
    return py_util.multireplace(input_str, {
        '-': '_minus_',
        '#': '-',
        '|': '_bar_',
        '+': '_plus_',
        '.': '_dot_',
        ',': '_comma_'
    })


def generate_pddl_from_traj(traj_json_path, output_pddl_path=None, x_display='0'):
    """
    Generate a problem.pddl file from an existing ALFRED trajectory JSON.

    Args:
        traj_json_path: Path to traj_data.json file
        output_pddl_path: Path to save the generated PDDL (optional)
        x_display: X server display number

    Returns:
        str: The generated PDDL string
    """
    # Load trajectory data
    with open(traj_json_path, 'r') as f:
        traj_data = json.load(f)

    # Extract task parameters
    scene_num = traj_data['scene']['scene_num']
    scene_name = f"FloorPlan{scene_num}"
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    init_action = traj_data['scene']['init_action']

    # Extract PDDL parameters
    pddl_params = traj_data['pddl_params']
    object_target = pddl_params['object_target']
    parent_target = pddl_params['parent_target']
    toggle_target = pddl_params.get('toggle_target', '')
    mrecep_target = pddl_params.get('mrecep_target', '')
    object_sliced = pddl_params.get('object_sliced', False)

    task_type = traj_data['task_type']

    # Initialize THOR environment
    print(f"Initializing THOR environment on display {x_display}...")
    env = ThorEnv(x_display=x_display)

    # Reset to the specific scene
    print(f"Resetting to {scene_name}...")
    env.reset(scene_name, silent=True)

    # Restore scene to match trajectory initial state
    print("Restoring scene state...")
    toggle_object = traj_data['scene'].get('toggle_object', None)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object)

    # Execute initial action
    if isinstance(init_action, list):
        for act in init_action:
            if act:
                env.step(dict(act))
    else:
        env.step(dict(init_action))

    # Generate PDDL from current environment state
    print("Generating PDDL from environment state...")
    pddl_string = generate_pddl_from_env(
        env=env,
        task_type=task_type,
        object_target=object_target,
        parent_target=parent_target,
        toggle_target=toggle_target,
        mrecep_target=mrecep_target,
        object_sliced=object_sliced,
        problem_id=0
    )

    # Save to file if requested
    if output_pddl_path:
        print(f"Saving PDDL to {output_pddl_path}...")
        with open(output_pddl_path, 'w') as f:
            f.write(pddl_string)

    # Clean up
    env.stop()

    return pddl_string


def generate_pddl_from_env(env, task_type, object_target, parent_target,
                           toggle_target='', mrecep_target='', object_sliced=False,
                           problem_id=0, agent_pose=(0, 0, 0, 0)):
    """
    Generate a problem.pddl file from a THOR environment state.

    Args:
        env: ThorEnv instance with initialized scene
        task_type: Task type (e.g., 'pick_and_place_simple')
        object_target: Target object type (e.g., 'PepperShaker')
        parent_target: Target receptacle type (e.g., 'Drawer')
        toggle_target: Toggle object type (optional)
        mrecep_target: Movable receptacle type (optional)
        object_sliced: Whether object needs slicing
        problem_id: Problem ID number
        agent_pose: Agent's (x, y, rotation, horizon) position

    Returns:
        str: Complete PDDL problem string
    """
    # Get current environment metadata
    metadata = env.last_event.metadata
    object_dict = {obj['objectId']: obj for obj in metadata['objects']}

    # Setup PDDL goal type
    constants.pddl_goal_type = task_type

    # Convert object names to indices
    object_target_idx = constants.OBJECTS.index(object_target) if object_target else None
    parent_target_idx = constants.OBJECTS.index(parent_target) if parent_target else None
    toggle_target_idx = constants.OBJECTS.index(toggle_target) if toggle_target else None
    mrecep_target_idx = constants.OBJECTS.index(mrecep_target) if mrecep_target else None

    # Build PDDL header
    receptacle_types = copy.deepcopy(constants.RECEPTACLES) - set(constants.MOVABLE_RECEPTACLES)
    objects = copy.deepcopy(constants.OBJECTS_SET) - receptacle_types

    object_str = '\n        '.join([obj + ' - object' for obj in sorted(objects)])
    otype_str = '\n        '.join([obj + 'Type - otype' for obj in sorted(objects)])
    rtype_str = '\n        '.join([obj + 'Type - rtype' for obj in sorted(receptacle_types)])

    # Get goal PDDL from goal library
    import goal_library as glib
    goal_type = task_type
    if object_sliced:
        goal_type += "_slice"
    goal_str = glib.gdict[goal_type]['pddl']
    goal_str = goal_str.format(
        obj=object_target if object_target else "",
        recep=parent_target if parent_target else "",
        toggle=toggle_target if toggle_target else "",
        mrecep=mrecep_target if mrecep_target else ""
    )

    # Fix special characters
    pddl_start = f'''
(define (problem plan_{problem_id})
    (:domain put_task)
    (:metric minimize (totalCost))
    (:objects
        agent1 - agent
        {object_str}
        {otype_str}
        {rtype_str}
'''

    pddl_init_header = '''
    (:init
        (= (totalCost) 0)
'''

    pddl_start = fix_pddl_str_chars(pddl_start)
    pddl_init_header = fix_pddl_str_chars(pddl_init_header)
    goal_str = fix_pddl_str_chars(goal_str)

    # Collect object instances from environment
    # Filter for relevant objects
    knife_obj = {'ButterKnife', 'Knife'} if object_sliced else set()

    object_instances = set()
    receptacle_instances = set()

    for obj in metadata['objects']:
        obj_type = obj['objectType']
        obj_id = obj['objectId']

        # Add receptacles (non-movable)
        if obj['receptacle'] and obj_type in constants.RECEPTACLES and obj_type not in constants.MOVABLE_RECEPTACLES_SET:
            receptacle_instances.add(obj_id)

        # Add relevant objects
        if (obj_type == object_target or
            obj_type == mrecep_target or
            obj_type == toggle_target or
            obj_type in knife_obj or
            obj_type in constants.MOVABLE_RECEPTACLES_SET):
            object_instances.add(obj_id)

    # Build object and receptacle instance strings
    object_instance_str = '\n        '.join(sorted([f"{obj_id} - object" for obj_id in object_instances]))
    receptacle_instance_str = '\n        '.join(sorted([f"{recep_id} - receptacle" for recep_id in receptacle_instances]))

    # Build location strings (simplified - using agent location only for now)
    agent_location = f'loc|{agent_pose[0]}|{agent_pose[1]}|{agent_pose[2]}|{agent_pose[3]}'
    location_str = f'{agent_location} - location'

    # Build middle section
    pddl_mid_start = f'''
        {object_instance_str}
        {receptacle_instance_str}
        {location_str}
        )
'''

    # Build init facts
    # Agent location
    agent_location_str = f'(atLocation agent1 {agent_location})'

    # Receptacle types
    receptacle_type_strs = [f'(receptacleType {recep_id} {object_dict[recep_id]["objectType"]}Type)'
                            for recep_id in receptacle_instances]
    receptacle_type_str = '\n        '.join(receptacle_type_strs)

    # Object types
    object_type_strs = [f'(objectType {obj_id} {object_dict[obj_id]["objectType"]}Type)'
                       for obj_id in object_instances]
    object_type_str = '\n        '.join(object_type_strs)

    # Openable receptacles
    openable_strs = [f'(openable {recep_id})'
                     for recep_id in receptacle_instances
                     if object_dict[recep_id]['objectType'] in constants.OPENABLE_CLASS_SET]
    openable_str = '\n        '.join(openable_strs)

    # Opened receptacles
    opened_strs = [f'(opened {obj["objectId"]})'
                   for obj in metadata['objects']
                   if obj['objectId'] in receptacle_instances and obj.get('isOpen', False)]
    opened_str = '\n        '.join(opened_strs)

    # Object containment relationships
    in_receptacle_strs = []
    for obj in metadata['objects']:
        if obj['objectId'] in object_instances and obj.get('parentReceptacles'):
            for parent_recep in obj['parentReceptacles']:
                if parent_recep in receptacle_instances:
                    in_receptacle_strs.append(f"(inReceptacle {obj['objectId']} {parent_recep})")
    in_receptacle_str = '\n        '.join(in_receptacle_strs)

    # Object properties
    cleanable_strs = [f'(cleanable {obj_id})'
                      for obj_id in object_instances
                      if object_dict[obj_id]['objectType'] in constants.VAL_ACTION_OBJECTS.get('Cleanable', [])]
    cleanable_str = '\n        '.join(cleanable_strs)

    heatable_strs = [f'(heatable {obj_id})'
                     for obj_id in object_instances
                     if object_dict[obj_id]['objectType'] in constants.VAL_ACTION_OBJECTS.get('Heatable', [])]
    heatable_str = '\n        '.join(heatable_strs)

    coolable_strs = [f'(coolable {obj_id})'
                     for obj_id in object_instances
                     if object_dict[obj_id]['objectType'] in constants.VAL_ACTION_OBJECTS.get('Coolable', [])]
    coolable_str = '\n        '.join(coolable_strs)

    toggleable_strs = [f'(toggleable {obj_id})'
                       for obj_id in object_instances
                       if object_dict[obj_id]['objectType'] in constants.VAL_ACTION_OBJECTS.get('Toggleable', [])]
    toggleable_str = '\n        '.join(toggleable_strs)

    sliceable_strs = [f'(sliceable {obj_id})'
                      for obj_id in object_instances
                      if object_dict[obj_id]['objectType'] in constants.VAL_ACTION_OBJECTS.get('Sliceable', [])]
    sliceable_str = '\n        '.join(sliceable_strs)

    # Receptacle objects (movable)
    receptacle_obj_strs = [f'(isReceptacleObject {obj_id})'
                           for obj_id in object_instances
                           if object_dict[obj_id]['objectType'] in constants.MOVABLE_RECEPTACLES_SET]
    receptacle_obj_str = '\n        '.join(receptacle_obj_strs)

    # Build complete init section
    pddl_mid_init = f'''
        {receptacle_type_str}
        {object_type_str}
        {receptacle_obj_str}
        {openable_str}
        {agent_location_str}
        {opened_str}
        {cleanable_str}
        {heatable_str}
        {coolable_str}
        {toggleable_str}
        {sliceable_str}
        {in_receptacle_str}
        )
'''

    # Fix special characters
    pddl_mid_start = fix_pddl_str_chars(pddl_mid_start)
    pddl_mid_init = fix_pddl_str_chars(pddl_mid_init)

    # Combine all sections
    pddl_string = (pddl_start + '\n' +
                   pddl_mid_start + '\n' +
                   pddl_init_header + '\n' +
                   pddl_mid_init + '\n' +
                   goal_str)

    return pddl_string


def main():
    parser = argparse.ArgumentParser(
        description='Generate a problem.pddl file from a THOR environment or ALFRED trajectory')
    parser.add_argument('--traj_json', type=str, required=True,
                       help='Path to traj_data.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDDL file path (default: <traj_dir>/problem_generated.pddl)')
    parser.add_argument('--x_display', type=str, default='7',
                       help='X server display number')

    args = parser.parse_args()

    # Set default output path if not specified
    if args.output is None:
        traj_dir = os.path.dirname(args.traj_json)
        args.output = os.path.join(traj_dir, 'problem_generated.pddl')

    # Generate PDDL
    try:
        pddl_string = generate_pddl_from_traj(
            args.traj_json,
            args.output,
            args.x_display
        )
        print(f"\nSuccessfully generated PDDL file: {args.output}")
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
