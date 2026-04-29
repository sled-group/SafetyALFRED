#!/usr/bin/env python3
"""
Script to render and visualize a plan generated from PDDL.

Takes a trajectory JSON, generates PDDL, runs the planner, and executes
the plan in THOR while saving video frames.

Usage:
    python render_plan.py --traj_json <path> --domain <path> --output_dir <path>
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
from gen.utils import video_util
from generate_problem_pddl_full import generate_pddl_from_traj_full

# Import DANLI planner
import importlib.util
danli_planner_path = '/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/planner.py'
spec = importlib.util.spec_from_file_location("danli_planner", danli_planner_path)
danli_planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(danli_planner)
PDDLPlanner = danli_planner.PDDLPlanner


def pddl_action_to_thor_actions(pddl_action, env, agent_loc_history):
    """
    Convert a PDDL action tuple to THOR API actions.

    Args:
        pddl_action: Tuple like ('gotolocation', 'agent1', 'loc_start', 'loc_end')
        env: THOR environment
        agent_loc_history: List tracking agent location history

    Returns:
        list: List of THOR API action dicts
    """
    action_name = pddl_action[0].lower()
    thor_actions = []

    if action_name == 'gotolocation':
        # Extract target location
        target_loc = pddl_action[3]  # loc_end
        # Parse location: loc_bar__minus_14_bar_5_bar_3_bar_30
        # Format: loc|x|y|z|rotation
        parts = target_loc.replace('loc_bar_', '').split('_bar_')

        # Convert coordinates
        def parse_coord(s):
            s = s.replace('_minus_', '-').replace('_plus_', '+').replace('_dot_', '.')
            return float(s)

        x = parse_coord(parts[0])
        y = parse_coord(parts[1])
        z = parse_coord(parts[2])
        rotation = float(parts[3])

        # ALFRED uses 0.25 grid size
        x_pos = x * constants.AGENT_STEP_SIZE
        z_pos = y * constants.AGENT_STEP_SIZE  # y in PDDL is z in THOR
        rotation_deg = rotation * 90

        # Get current agent state
        agent_pos = env.last_event.metadata['agent']['position']
        agent_rot = env.last_event.metadata['agent']['rotation']['y']
        agent_hor = env.last_event.metadata['agent']['cameraHorizon']

        # Calculate horizon from z coordinate (which is the third index in PDDL)
        target_horizon = int(z)

        # Teleport to target
        thor_actions.append({
            'action': 'TeleportFull',
            'x': x_pos,
            'y': 0.9009992,  # Standard agent height
            'z': z_pos,
            'rotateOnTeleport': True,
            'rotation': rotation_deg,
            'horizon': target_horizon,
            'standing': True
        })

        agent_loc_history.append(target_loc)

    elif action_name == 'pickupobjectinreceptacle1':
        # Extract object ID from PDDL format
        object_id_pddl = pddl_action[3]
        # Convert PDDL name to THOR object ID
        # Format: peppershaker_bar__minus_00_dot_92_bar__plus_00_dot_93_bar__minus_01_dot_39
        # To: PepperShaker|-00.92|+00.93|-01.39

        # Split into type and coordinates
        parts = object_id_pddl.split('_bar_')
        object_type = parts[0].title()  # e.g., 'peppershaker' -> 'Peppershaker'

        # Special case for multi-word types
        if object_type.lower() == 'peppershaker':
            object_type = 'PepperShaker'
        elif object_type.lower() == 'countertop':
            object_type = 'CounterTop'
        elif object_type.lower() == 'sinkbasin':
            object_type = 'SinkBasin'

        # Convert coordinates
        coords = []
        for i in range(1, len(parts)):
            coord = parts[i].replace('_minus_', '-').replace('_plus_', '+').replace('_dot_', '.')
            # Handle leading underscores
            coord = coord.strip('_')
            coords.append(coord)

        object_id_thor = object_type + '|' + '|'.join(coords)

        thor_actions.append({
            'action': 'PickupObject',
            'objectId': object_id_thor,
            'forceAction': True,
            'manualInteract': False
        })

    elif action_name == 'pickupobjectnoreceptacle':
        # Extract object ID
        object_id = pddl_action[3]
        object_id_thor = object_id.replace('_bar_', '|').replace('_plus_', '+').replace('_minus_', '-').replace('_dot_', '.')

        thor_actions.append({
            'action': 'PickupObject',
            'objectId': object_id_thor,
            'forceAction': True,
            'manualInteract': False
        })

    elif action_name == 'putobjectinreceptacle1':
        # Extract receptacle ID from PDDL format
        receptacle_id_pddl = pddl_action[5]
        # Convert to THOR format
        parts = receptacle_id_pddl.split('_bar_')
        receptacle_type = parts[0].title()

        # Special case for multi-word types
        type_mapping = {
            'countertop': 'CounterTop',
            'sinkbasin': 'SinkBasin',
            'stoveburner': 'StoveBurner',
            'garbagecan': 'GarbageCan',
            'coffeetable': 'CoffeeTable',
            'sidetable': 'SideTable',
            'diningtable': 'DiningTable',
        }
        receptacle_type = type_mapping.get(receptacle_type.lower(), receptacle_type)

        # Convert coordinates
        coords = []
        for i in range(1, len(parts)):
            coord = parts[i].replace('_minus_', '-').replace('_plus_', '+').replace('_dot_', '.')
            coord = coord.strip('_')
            coords.append(coord)

        receptacle_id_thor = receptacle_type + '|' + '|'.join(coords)

        # First try to open if it's openable
        objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
        receptacle = objects.get(receptacle_id_thor)

        if receptacle and receptacle.get('openable', False) and not receptacle.get('isOpen', False):
            thor_actions.append({
                'action': 'OpenObject',
                'objectId': receptacle_id_thor,
                'forceAction': True
            })

        # Put object in receptacle
        thor_actions.append({
            'action': 'PutObject',
            'objectId': receptacle_id_thor,
            'forceAction': True,
            'placeStationary': True
        })

    elif action_name == 'openobject':
        receptacle_id = pddl_action[3]
        receptacle_id_thor = receptacle_id.replace('_bar_', '|').replace('_plus_', '+').replace('_minus_', '-').replace('_dot_', '.')

        thor_actions.append({
            'action': 'OpenObject',
            'objectId': receptacle_id_thor,
            'forceAction': True
        })

    elif action_name == 'closeobject':
        receptacle_id = pddl_action[3]
        receptacle_id_thor = receptacle_id.replace('_bar_', '|').replace('_plus_', '+').replace('_minus_', '-').replace('_dot_', '.')

        thor_actions.append({
            'action': 'CloseObject',
            'objectId': receptacle_id_thor,
            'forceAction': True
        })

    else:
        print(colored(f"Warning: Unknown PDDL action: {action_name}", 'yellow'))

    return thor_actions


def save_frame(env, output_dir, frame_idx):
    """Save a frame from the environment"""
    frame = env.last_event.frame
    frame_path = os.path.join(output_dir, 'frames', f'{frame_idx:09d}.png')

    from PIL import Image
    img = Image.fromarray(frame)
    img.save(frame_path)

    return frame_path


def render_plan(traj_json_path, domain_path, output_dir, x_display='7'):
    """
    Generate PDDL, create a plan, and render it in THOR.

    Args:
        traj_json_path: Path to trajectory JSON
        domain_path: Path to domain PDDL
        output_dir: Directory to save outputs
        x_display: X server display number
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)

    print("=" * 80)
    print("PLAN RENDERING")
    print("=" * 80)

    # Step 1: Generate PDDL
    print("\n[1/5] Generating PDDL from trajectory...")
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
    print("\n[2/5] Running Fast Downward planner...")
    plan_file = os.path.join(output_dir, 'sas_plan')

    try:
        planner = PDDLPlanner(
            fd_path='/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/fast-downward-24.06.1/fast-downward.py',
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
    print("\n[3/5] Initializing THOR environment...")

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

    # Step 4: Execute plan
    print("\n[4/5] Executing plan in THOR...")

    frame_idx = 0
    agent_loc_history = []
    execution_log = []

    # Save initial frame
    save_frame(env, output_dir, frame_idx)
    frame_idx += 1

    for step_idx, pddl_action in enumerate(plan, 1):
        print(f"\nStep {step_idx}/{len(plan)}: {' '.join(pddl_action)}")

        # Convert PDDL action to THOR actions
        thor_actions = pddl_action_to_thor_actions(pddl_action, env, agent_loc_history)

        for thor_action in thor_actions:
            print(f"  Executing: {thor_action['action']}")
            event = env.step(thor_action)

            # Save frame
            save_frame(env, output_dir, frame_idx)
            frame_idx += 1

            # Check success
            if event.metadata['lastActionSuccess']:
                print(colored(f"  ✓ Success", 'green'))
                execution_log.append({
                    'step': step_idx,
                    'pddl_action': ' '.join(pddl_action),
                    'thor_action': thor_action,
                    'success': True
                })
            else:
                error_msg = event.metadata.get('errorMessage', 'Unknown error')
                print(colored(f"  ✗ Failed: {error_msg}", 'red'))
                execution_log.append({
                    'step': step_idx,
                    'pddl_action': ' '.join(pddl_action),
                    'thor_action': thor_action,
                    'success': False,
                    'error': error_msg
                })

                # Save debug info
                with open(os.path.join(output_dir, 'debug.json'), 'w') as f:
                    json.dump(event.metadata['objects'], f, sort_keys=True, indent=4)

                # Continue anyway to see what happens
                # return False

    # Save final frame
    save_frame(env, output_dir, frame_idx)

    # Save execution log
    with open(os.path.join(output_dir, 'execution_log.json'), 'w') as f:
        json.dump(execution_log, f, indent=2)

    print(f"\n✓ Executed {len(plan)} actions, saved {frame_idx} frames")

    # Step 5: Create video
    print("\n[5/5] Creating video...")

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
        description='Render a plan generated from PDDL in THOR')
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
