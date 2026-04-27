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

from alfred.env.thor_env import ThorEnv
from alfred.gen import constants
from alfred.gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from alfred.gen.agents.deterministic_planner_agent import DeterministicPlannerAgent
from alfred.gen.utils import game_util
import copy


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
    constants.save_path = '/tmp/alfred_pddl_gen'
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
        reachable_positions = reachable_event.metadata['reachablePositions'/home/josue/Desktop/Research/SLED/MSS/alfred/gen/pipeline_pddl_to_video.py]

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

    print("Setting up problem...")
    agent.setup_problem({'info': info}, scene=scene_info, objs=task_objs)

    # Force update of receptacle points and navigation graph
    print("Building navigation graph...")
    game_state.update_receptacle_nearest_points()

    # Generate PDDL using the full state_to_pddl() method
    print("Generating PDDL from full game state...")
    pddl_string = game_state.state_to_pddl()

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
    import re

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
