#!/usr/bin/env python3
"""
Complete pipeline: ALFRED trajectory → PDDL → Plan → Render video (Thor 5.0)

This script takes an ALFRED trajectory and:
1. Generates a PDDL problem from it
2. Plans using Fast Downward
3. Executes the plan in THOR 5.0
4. Converts to ALFRED trajectory format
5. Renders with smooth navigation and time delays

IMPORTANT: This pipeline uses AI2-THOR 5.0 actions for safety hazard object placement.
           Use the modern virtual environment:
           source /home/josue/Desktop/Research/SLED/MSS/E.T./et_env_safety_modern/bin/activate

Usage:
    python pipeline_pddl_to_video_thor5.py --traj_json <path> --output_dir <path>
"""

import os
import sys
import json
import argparse
import shutil
import glob
import numpy as np
from termcolor import colored

# Resolve bundle root: this script lives at <BUNDLE>/alfred/gen/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))

# Add ALFRED paths
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'gen'))

# Add E.T. gen directory for imports (bundled copy)
et_gen_dir = os.path.join(_BUNDLE_ROOT, 'E.T.', 'alfred', 'gen')
sys.path.insert(0, et_gen_dir)
sys.path.insert(0, _THIS_DIR)
# Make the bundled E.T./alfred package importable as `alfred.*`
sys.path.insert(0, os.path.join(_BUNDLE_ROOT, 'E.T.'))

from alfred.env.thor_env_thor5 import ThorEnv
from alfred.gen import constants
from alfred.gen.utils import video_util, game_util, augment_util
from alfred.gen.graph.graph_obj import Graph
from generate_problem_pddl_full_thor5 import generate_pddl_from_traj_full
from safety_initialization import initialize_safety_hazard_scene

# Import DANLI planner
import importlib.util
danli_planner_path = os.path.join(
    _BUNDLE_ROOT, 'alfred_git', 'alfred', 'data', 'DANLI', 'pddl', 'planner.py')
spec = importlib.util.spec_from_file_location("danli_planner", danli_planner_path)
danli_planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(danli_planner)
PDDLPlanner = danli_planner.PDDLPlanner

# Import conversion functions from other scripts
from render_plan_with_navigation import (
    pddl_action_to_navigation_sequence,
    convert_pddl_object_to_thor,
    save_frame,
    add_delay_frames
)
from convert_plan_to_traj import convert_plan_to_traj


def find_position_away_from_fridge(current_x, current_z, fridge_position, nav_graph, min_distance=0.5):
    """
    Find a reachable position that is at least min_distance away from fridge.

    Returns the closest valid position to current position that satisfies the distance constraint.
    """
    if fridge_position is None or nav_graph is None:
        return current_x, current_z, False

    fridge_x = fridge_position['x']
    fridge_z = fridge_position['z']

    # Check if current position already satisfies constraint
    current_dist = np.sqrt((current_x - fridge_x)**2 + (current_z - fridge_z)**2)
    if current_dist >= min_distance:
        return current_x, current_z, False

    # Find closest reachable point that is at least min_distance from fridge
    best_dist_to_current = float('inf')
    best_point = None

    for point in nav_graph.points:
        point_x = point[0] * constants.AGENT_STEP_SIZE
        point_z = point[1] * constants.AGENT_STEP_SIZE

        # Check distance to fridge
        dist_to_fridge = np.sqrt((point_x - fridge_x)**2 + (point_z - fridge_z)**2)
        if dist_to_fridge < min_distance:
            continue

        # Find closest valid point to current position
        dist_to_current = np.sqrt((point_x - current_x)**2 + (point_z - current_z)**2)
        if dist_to_current < best_dist_to_current:
            best_dist_to_current = dist_to_current
            best_point = (point_x, point_z)

    if best_point:
        print(colored(f"    → Adjusted position from ({current_x:.2f}, {current_z:.2f}) to ({best_point[0]:.2f}, {best_point[1]:.2f}) - {min_distance}m from fridge", 'yellow'))
        return best_point[0], best_point[1], True

    return current_x, current_z, False


_DEFAULT_DOMAIN = os.path.join(
    _BUNDLE_ROOT, 'alfred_git', 'alfred', 'data', 'DANLI', 'pddl', 'domain.pddl')
_DEFAULT_FD = os.path.join(
    _BUNDLE_ROOT, 'alfred_git', 'alfred', 'data', 'DANLI', 'pddl',
    'fast-downward-24.06.1', 'fast-downward.py')


def run_complete_pipeline(
    traj_json_path,
    output_dir,
    domain_path=_DEFAULT_DOMAIN,
    x_display='7',
    render_final=True,
    smooth_nav=True,
    time_delays=True,
    use_dynamic_reachable=True,
    use_teleport=False,
    add_sink_item=False,
    alternative_cabinet=None,
    alternative_object_location=None,
    clear_sink_objects=False,
    clear_microwave_objects=False
):
    """
    Run the complete pipeline from trajectory to rendered video.

    Args:
        traj_json_path: Path to original ALFRED trajectory
        output_dir: Directory to save all outputs
        domain_path: Path to PDDL domain file
        x_display: X server display number
        render_final: Whether to render final trajectory with ALFRED rendering
        smooth_nav: Use smooth navigation in final render
        time_delays: Use time delays in final render
        use_dynamic_reachable: Use GetReachablePositions for actual navigable points
        use_teleport: Use TeleportFull for navigation (calculates exact rotation to face objects)
        add_sink_item: For property damage with sink, add extra sink-appropriate item during initialization
        alternative_cabinet: For fall_trip_hazard: use alternative cabinet (0-based index)
        alternative_object_location: For appliance_misuse/property_damage: use alternative location for target object (0-based index of objects >= 1m away)
        clear_sink_objects: If True, remove all objects from sink except safety_object and target_object
        clear_microwave_objects: If True, remove all objects from microwaves except target and safety objects

    Returns:
        dict: Results with paths to all outputs
    """

    os.makedirs(output_dir, exist_ok=True)

    # Handle alternative cabinet: create modified trajectory file with replaced cabinet IDs
    if alternative_cabinet is not None:
        print("\n" + "=" * 80)
        print("ALTERNATIVE CABINET MODE")
        print("=" * 80)

        # Load original trajectory
        with open(traj_json_path, 'r') as f:
            traj_data = json.load(f)

        # Check if this is a fall_trip_hazard scenario
        safety_issue_type = None
        if 'safety_receptacle' in traj_data['scene'] and traj_data['scene']['safety_receptacle']:
            safety_issue_type = traj_data['scene']['safety_receptacle'].get('safetyIssue', '')

        is_fall_trip_hazard = safety_issue_type and ('fall' in safety_issue_type.lower() or 'trip' in safety_issue_type.lower())

        if is_fall_trip_hazard:
            original_cabinet_id = traj_data['scene']['safety_receptacle'].get('objectId')
            scene_num = traj_data['scene'].get('scene_num')

            if original_cabinet_id and scene_num:
                # Load openable.json to find alternative cabinets
                floor_plan = f"FloorPlan{scene_num}"
                # Get the absolute path to layouts directory
                # pipeline is at MSS/alfred/gen/, we need MSS/E.T./alfred/gen/layouts/
                script_dir = os.path.dirname(os.path.abspath(__file__))
                mss_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to MSS directory
                layouts_dir = os.path.join(mss_dir, 'E.T.', 'alfred', 'gen', 'layouts')
                openable_json_path = os.path.join(layouts_dir, f'{floor_plan}-openable.json')

                if os.path.exists(openable_json_path):
                    with open(openable_json_path, 'r') as f:
                        openable_data = json.load(f)

                    # Find all cabinets with y < 1.00
                    all_cabinets = []
                    for obj_id in openable_data.keys():
                        if 'Cabinet' in obj_id:
                            parts = obj_id.split('|')
                            if len(parts) >= 3:
                                y_val = float(parts[2])
                                if y_val < 1.00:
                                    all_cabinets.append(obj_id)

                    # Exclude original cabinet
                    available_cabinets = [cab for cab in all_cabinets if cab != original_cabinet_id]

                    if available_cabinets:
                        available_cabinets.sort()
                        cabinet_index = alternative_cabinet % len(available_cabinets)
                        selected_cabinet = available_cabinets[cabinet_index]

                        print(f"Original cabinet: {original_cabinet_id}")
                        print(f"Selected alternative: {selected_cabinet} (index {cabinet_index}/{len(available_cabinets)})")
                        print(f"Replacing cabinet IDs in trajectory...")

                        # Replace in scene.safety_receptacle
                        if traj_data['scene']['safety_receptacle'].get('objectId') == original_cabinet_id:
                            traj_data['scene']['safety_receptacle']['objectId'] = selected_cabinet

                        # Replace in high_pddl actions
                        high_replaced = 0
                        if 'plan' in traj_data and 'high_pddl' in traj_data['plan']:
                            for action in traj_data['plan']['high_pddl']:
                                if 'planner_action' in action and 'objectId' in action['planner_action']:
                                    if action['planner_action']['objectId'] == original_cabinet_id:
                                        action['planner_action']['objectId'] = selected_cabinet
                                        high_replaced += 1

                        # Replace in low_actions
                        low_replaced = 0
                        if 'plan' in traj_data and 'low_actions' in traj_data['plan']:
                            for action in traj_data['plan']['low_actions']:
                                if 'api_action' in action and 'objectId' in action['api_action']:
                                    if action['api_action']['objectId'] == original_cabinet_id:
                                        action['api_action']['objectId'] = selected_cabinet
                                        low_replaced += 1
                                if 'discrete_action' in action and 'args' in action['discrete_action']:
                                    if action['discrete_action']['args'] == original_cabinet_id:
                                        action['discrete_action']['args'] = selected_cabinet

                        print(f"✓ Replaced {high_replaced} high_pddl, {low_replaced} low_actions")

                        # Save modified trajectory in the same directory as the original
                        input_dir = os.path.dirname(traj_json_path)
                        alt_traj_path = os.path.join(input_dir, f'traj_data_alt_cabinet_{alternative_cabinet}.json')
                        with open(alt_traj_path, 'w') as f:
                            json.dump(traj_data, f, indent=2)
                        print(f"✓ Saved alternative trajectory: {alt_traj_path}")
                        print("=" * 80 + "\n")

                        # Use the alternative trajectory for the rest of the pipeline
                        traj_json_path = alt_traj_path
                    else:
                        print(f"⚠ No alternative cabinets found, using original trajectory")
                else:
                    print(f"⚠ Openable file not found: {openable_json_path}")
            else:
                print(f"⚠ Missing cabinet ID or scene number")
        else:
            print(f"⚠ Not a fall_trip_hazard scenario, ignoring alternative_cabinet parameter")

        print()

    # Handle alternative object location: move target object to a different position
    if alternative_object_location is not None:
        print("\n" + "=" * 80)
        print("ALTERNATIVE OBJECT LOCATION MODE")
        print("=" * 80)

        # Load original trajectory
        with open(traj_json_path, 'r') as f:
            traj_data = json.load(f)

        # Check if this is an appliance_misuse or property_damage scenario
        safety_issue_type = None
        if 'safety_receptacle' in traj_data['scene'] and traj_data['scene']['safety_receptacle']:
            safety_issue_type = traj_data['scene']['safety_receptacle'].get('safetyIssue', '')

        is_appliance_misuse = safety_issue_type and 'appliance' in safety_issue_type.lower() and 'misuse' in safety_issue_type.lower()
        is_property_damage = safety_issue_type and 'property' in safety_issue_type.lower() and 'damage' in safety_issue_type.lower()
        is_spoilage = safety_issue_type and 'spoilage' in safety_issue_type.lower()

        # For spoilage scenarios, track fridge position for minimum distance enforcement
        fridge_position = None
        fridge_object_id = None
        if is_spoilage:
            # Get fridge position from safety_receptacle
            safety_receptacle = traj_data['scene'].get('safety_receptacle', {})
            if safety_receptacle and 'Fridge' in safety_receptacle.get('objectId', ''):
                fridge_object_id = safety_receptacle.get('objectId')
                fridge_position = safety_receptacle.get('position')
                print(f"  Spoilage scenario detected - Fridge at ({fridge_position['x']:.2f}, {fridge_position['z']:.2f})")
                print(f"  Will enforce minimum 0.5m distance from fridge for teleports")

        if is_appliance_misuse or is_property_damage:
            # Get target object from pddl_params
            pddl_params = traj_data.get('pddl_params', {})
            target_object_type = pddl_params.get('object_target')

            if target_object_type:
                # Find the target object in object_poses (it's a list of dicts)
                object_poses = traj_data['scene']['object_poses']
                target_object_entry = None
                target_position = None
                target_index = None

                for idx, obj_entry in enumerate(object_poses):
                    obj_id = obj_entry['objectName']
                    if target_object_type in obj_id:
                        target_object_entry = obj_entry
                        target_position = obj_entry['position']
                        target_index = idx
                        break

                if target_object_entry and target_position:
                    print(f"Target object: {target_object_entry['objectName']}")
                    print(f"Current position: x={target_position['x']:.3f}, y={target_position['y']:.3f}, z={target_position['z']:.3f}")

                    # Find all objects that are at least 1 meter away
                    import math
                    alternative_objects = []

                    for idx, obj_entry in enumerate(object_poses):
                        if idx == target_index:
                            continue

                        alt_pos = obj_entry['position']
                        distance = math.sqrt(
                            (alt_pos['x'] - target_position['x']) ** 2 +
                            (alt_pos['z'] - target_position['z']) ** 2
                        )

                        if distance >= 1.0:
                            alternative_objects.append({
                                'index': idx,
                                'name': obj_entry['objectName'],
                                'position': alt_pos,
                                'rotation': obj_entry['rotation'],
                                'distance': distance
                            })

                    if alternative_objects:
                        # Sort by distance for consistent ordering
                        alternative_objects.sort(key=lambda x: x['distance'])

                        print(f"Found {len(alternative_objects)} objects >= 1m away")

                        # Select alternative by index
                        alt_index = alternative_object_location % len(alternative_objects)
                        selected_alt = alternative_objects[alt_index]

                        print(f"Selected alternative: {selected_alt['name']} (index {alt_index}/{len(alternative_objects)})")
                        print(f"  Distance: {selected_alt['distance']:.2f}m")
                        print(f"  Position: x={selected_alt['position']['x']:.3f}, y={selected_alt['position']['y']:.3f}, z={selected_alt['position']['z']:.3f}")

                        # Replace target object position with selected alternative position
                        object_poses[target_index]['position'] = selected_alt['position'].copy()
                        object_poses[target_index]['rotation'] = selected_alt['rotation'].copy()

                        # Remove the alternative object from object_poses (by index)
                        # Must remove by index in reverse order to avoid index shifting
                        del object_poses[selected_alt['index']]

                        print(f"✓ Replaced {target_object_entry['objectName']} position")
                        print(f"✓ Removed {selected_alt['name']} from object_poses")

                        # Save modified trajectory in the same directory as the original
                        input_dir = os.path.dirname(traj_json_path)
                        alt_traj_path = os.path.join(input_dir, f'traj_data_alt_obj_loc_{alternative_object_location}.json')
                        with open(alt_traj_path, 'w') as f:
                            json.dump(traj_data, f, indent=2)
                        print(f"✓ Saved alternative trajectory: {alt_traj_path}")
                        print("=" * 80 + "\n")

                        # Use the alternative trajectory for the rest of the pipeline
                        traj_json_path = alt_traj_path
                    else:
                        print(f"⚠ No objects found >= 1m away from target object")
                else:
                    print(f"⚠ Could not find target object {target_object_type} in object_poses")
            else:
                print(f"⚠ No target object specified in pddl_params")
        else:
            print(f"⚠ Not an appliance_misuse or property_damage scenario, ignoring alternative_object_location parameter")

        print()

    print("=" * 80)
    print("PDDL PLANNING AND RENDERING PIPELINE (THOR 5.0)")
    print("=" * 80)
    print(f"Input: {traj_json_path}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    results = {
        'success': False,
        'pddl_problem': None,
        'plan_file': None,
        'plan_execution_video': None,
        'converted_trajectory': None,
        'final_video': None,
        'error': None,
        'partial_execution': False,
        'execution_stats': {}
    }

    # Track if we've fallen back to skip_placement mode
    skip_placement_mode = False

    try:
        # =====================================================================
        # STEP 1: Generate PDDL Problem
        # =====================================================================
        print("\n[1/6] Generating PDDL problem from trajectory...")
        problem_pddl_path = os.path.join(output_dir, 'problem.pddl')

        try:
            pddl_string = generate_pddl_from_traj_full(
                traj_json_path,
                problem_pddl_path,
                x_display,
                use_dynamic_reachable=use_dynamic_reachable,
                skip_target_object_placement=(alternative_object_location is not None)
            )
            print(colored(f"  ✓ PDDL problem generated: {problem_pddl_path}", 'green'))
            results['pddl_problem'] = problem_pddl_path
        except Exception as e:
            print(colored(f"  ✗ Failed to generate PDDL: {e}", 'red'))
            import traceback
            traceback.print_exc()
            results['error'] = f"PDDL generation failed: {e}"
            return results

        # =====================================================================
        # STEP 2: Generate Plan using Fast Downward
        # =====================================================================
        print("\n[2/6] Running Fast Downward planner...")
        plan_file = os.path.join(output_dir, 'sas_plan')

        try:
            planner = PDDLPlanner(
                fd_path=_DEFAULT_FD,
                plan_file=plan_file,
                alias='ff-astar',
                timeout=60
            )
            plan, runtime = planner.plan(domain_path, problem_pddl_path, debug=False)

            if plan is None:
                print(colored("  ✗ Failed to generate plan", 'red'))
                results['error'] = "Planning failed"
                return results

            print(colored(f"  ✓ Plan generated: {len(plan)} actions in {runtime:.2f}s", 'green'))

            # Save plan in readable format
            plan_txt_path = os.path.join(output_dir, 'plan.txt')
            with open(plan_txt_path, 'w') as f:
                for i, action in enumerate(plan, 1):
                    f.write(f"{i}. {' '.join(action)}\n")
                    print(f"    {i}. {' '.join(action)}")

            results['plan_file'] = plan_txt_path

        except Exception as e:
            print(colored(f"  ✗ Planner error: {e}", 'red'))
            import traceback
            traceback.print_exc()
            results['error'] = f"Planning failed: {e}"
            return results

        # =====================================================================
        # STEP 3: Execute Plan in THOR
        # =====================================================================
        print("\n[3/6] Executing plan in THOR...")

        execution_dir = os.path.join(output_dir, 'plan_execution')
        os.makedirs(execution_dir, exist_ok=True)

        # Clear existing frames to prevent mixing with previous runs
        frames_dir = os.path.join(execution_dir, 'frames')
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)

        # Load trajectory data
        with open(traj_json_path, 'r') as f:
            traj_data = json.load(f)

        # Add trajectory path to traj_data for property_damage object extraction
        traj_data['traj_path'] = traj_json_path

        # Check for spoilage scenario and track fridge position for minimum distance enforcement
        safety_issue_type = None
        if 'safety_receptacle' in traj_data['scene'] and traj_data['scene']['safety_receptacle']:
            safety_issue_type = traj_data['scene']['safety_receptacle'].get('safetyIssue', '')

        is_spoilage = safety_issue_type and 'spoilage' in safety_issue_type.lower()
        fridge_position = None
        fridge_object_id = None
        if is_spoilage:
            # Get fridge position from safety_receptacle
            safety_receptacle = traj_data['scene'].get('safety_receptacle', {})
            if safety_receptacle and 'Fridge' in safety_receptacle.get('objectId', ''):
                fridge_object_id = safety_receptacle.get('objectId')
                fridge_position = safety_receptacle.get('position')
                print(f"  Spoilage scenario detected - Fridge at ({fridge_position['x']:.2f}, {fridge_position['z']:.2f})")
                print(f"  Will enforce minimum 0.5m distance from fridge for teleports")

        scene_num = traj_data['scene']['scene_num']
        scene_name = f'FloorPlan{scene_num}'

        env = ThorEnv(
            x_display=x_display,
            player_screen_width=300,

        )

        # Reset scene with object rendering enabled for instance_detections2D
        env.reset(scene_name, silent=True, render_object_image=True)

        # Initialize scene with safety hazard handling using unified function
        debug_log_path = os.path.join(output_dir, 'debug.txt')
        converted_traj_path = os.path.join(output_dir, 'converted_trajectory', 'traj_data.json')
        # Only use spawn and placement when alternative_cabinet, alternative_object_location, or add_sink_item are enabled
        use_spawn_and_placement = (alternative_cabinet is not None) or (alternative_object_location is not None) or add_sink_item
        _, skip_placement_mode = initialize_safety_hazard_scene(env, traj_data, debug_log_path, add_sink_item=add_sink_item, skip_placement=skip_placement_mode, save_modified_traj=converted_traj_path, skip_target_object_placement=(alternative_object_location is not None), use_spawn_and_placement=use_spawn_and_placement, clear_sink_objects=clear_sink_objects, clear_microwave_objects=clear_microwave_objects)

        print(f"  Environment initialized: {scene_name}")

        # Build navigation graph
        try:
            nav_graph = Graph(use_gt=True, construct_graph=True, scene_id=scene_num)
            print(f"  Navigation graph built: {len(nav_graph.points)} nodes")
        except Exception as e:
            print(colored(f"  ✗ Failed to build navigation graph: {e}", 'red'))
            env.stop()
            results['error'] = f"Navigation graph failed: {e}"
            return results

        # Execute plan
        frame_idx = 0
        agent_loc_history = []
        execution_log = []

        # Save initial frame
        save_frame(env, execution_dir, frame_idx)
        frame_idx += 1

        # Filter out redundant pick-and-place sequences (pick from receptacle A, put back in receptacle A)
        filtered_plan = []
        skip_until = -1  # Track indices to skip

        for i, current_action in enumerate(plan):
            # If we're in a skip range, don't add this action
            if i < skip_until:
                continue

            # Check if this is a pickupobjectinreceptacle1 action
            if current_action[0].lower() == 'pickupobjectinreceptacle1':
                pickup_object = current_action[3]  # Object ID
                pickup_receptacle = current_action[4]  # Receptacle ID

                # Look ahead for the next non-navigation action
                j = i + 1
                while j < len(plan):
                    next_action = plan[j]
                    action_type = next_action[0].lower()

                    # Skip navigation actions
                    if action_type == 'gotolocation':
                        j += 1
                        continue

                    # Check if it's a putobjectinreceptacle1 with same object and receptacle
                    if (action_type == 'putobjectinreceptacle1' and
                        next_action[4] == pickup_object and  # Same object
                        next_action[5] == pickup_receptacle):  # Same receptacle

                        # Found redundant sequence - skip from current index to j (inclusive)
                        obj_name = pickup_object.split('_bar_')[0]
                        recep_name = pickup_receptacle.split('_bar_')[0]
                        print(colored(f"  Filtering redundant sequence: pickup {obj_name} from {recep_name} → put back in {recep_name}", 'yellow'))
                        skip_until = j + 1
                        break
                    else:
                        # Found a different non-navigation action, not redundant
                        break

            # Add action if not in skip range
            if i >= skip_until:
                filtered_plan.append(current_action)

        if len(plan) != len(filtered_plan):
            print(colored(f"\nFiltered {len(plan) - len(filtered_plan)} redundant actions ({len(plan)} → {len(filtered_plan)})", 'green'))
        plan = filtered_plan

        for step_idx, pddl_action in enumerate(plan, 1):
            print(colored(f"\n  [{step_idx}/{len(plan)}] {' '.join(pddl_action)}", 'cyan', attrs=['bold']))

            # Get next action for optimization
            next_pddl_action = plan[step_idx] if step_idx < len(plan) else None

            # Skip navigation if the next object to interact with is already detected in frame with >= 50% visibility
            action_name = pddl_action[0].lower()
            if action_name == 'gotolocation' and next_pddl_action and use_teleport:
                # First, teleport to the target location and check visibility from there
                # Parse target location from the gotolocation action
                target_loc = pddl_action[3]  # loc_end
                parts = target_loc.replace('loc_bar_', '').split('_bar_')

                def parse_coord(s):
                    s = s.replace('_minus_', '-').replace('_plus_', '+').replace('_dot_', '.')
                    return float(s)

                nav_x = int(parse_coord(parts[0]))
                nav_y = int(parse_coord(parts[1]))
                nav_rotation_index = int(parse_coord(parts[2]))
                nav_horizon = int(parse_coord(parts[3]))

                # Convert to THOR coordinates
                nav_x_pos = nav_x * constants.AGENT_STEP_SIZE
                nav_z_pos = nav_y * constants.AGENT_STEP_SIZE
                agent_y = env.last_event.metadata['agent']['position']['y']

                # Determine target object from next action
                next_action_name = next_pddl_action[0].lower()
                target_object_id = None

                if next_action_name == 'pickupobjectinreceptacle1':
                    # Target is the receptacle (index 4)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[4])
                elif next_action_name == 'pickupobjectnoreceptacle':
                    # Target is the object (index 3)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])
                elif next_action_name == 'putobjectinreceptacle1':
                    # Target is the receptacle (index 5)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[5])
                elif next_action_name == 'putobjectinreceptacleobject1':
                    # Target is the movable receptacle (index 5)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[5])
                elif next_action_name in ['openobject', 'closeobject', 'toggleobjecton', 'toggleobjectoff', 'toggleobject']:
                    # Target is the object (index 3)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])
                elif next_action_name == 'sliceobject':
                    # Target is the object to slice (index 3)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])
                elif next_action_name in ['heatobject', 'heatobjectwithin']:
                    # Target is the microwave (index 3)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])
                elif next_action_name == 'coolobject':
                    # Target is the fridge (index 3)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])
                elif next_action_name in ['cleanobject', 'cleanobjectwithin']:
                    # Target is the sink basin (index 3)
                    target_object_id = convert_pddl_object_to_thor(next_pddl_action[3])

                # If we have a target object, teleport to the navigation location and check visibility
                if target_object_id:
                    # For spoilage scenarios, ensure ALL teleports are at least 0.5m from fridge
                    if is_spoilage and fridge_position:
                        nav_x_pos, nav_z_pos, was_adjusted = find_position_away_from_fridge(
                            nav_x_pos, nav_z_pos, fridge_position, nav_graph, min_distance=0.5
                        )

                    # Calculate rotation to face the target object from the navigation position
                    objects = {obj['objectId']: obj for obj in env.last_event.metadata['objects']}
                    target_obj = objects.get(target_object_id)

                    if target_obj:
                        # Calculate angle to face the object
                        target_pos = target_obj['position']
                        dx = target_pos['x'] - nav_x_pos
                        dz = target_pos['z'] - nav_z_pos
                        angle_rad = np.arctan2(dx, dz)
                        rotation_deg = np.degrees(angle_rad)
                        if rotation_deg < 0:
                            rotation_deg += 360

                        # Calculate horizon to look at object
                        horizontal_dist = np.sqrt(dx**2 + dz**2)
                        camera_height = agent_y + 0.675
                        vertical_dist = target_pos['y'] - camera_height
                        if horizontal_dist > 0.01:
                            horizon_rad = np.arctan2(-vertical_dist, horizontal_dist)
                            calculated_horizon = np.degrees(horizon_rad)
                            calculated_horizon = np.clip(calculated_horizon, -30, 60)
                        else:
                            calculated_horizon = nav_horizon

                        # Teleport to the navigation location facing the target object
                        teleport_action = {
                            'action': 'TeleportFull',
                            'x': nav_x_pos,
                            'y': agent_y,
                            'z': nav_z_pos,
                            'rotation': {'x': 0, 'y': rotation_deg, 'z': 0},
                            'horizon': calculated_horizon,
                            'standing': True
                        }

                        teleport_event = env.step(teleport_action)

                        if teleport_event.metadata['lastActionSuccess']:
                            # Check if target object is visible with >= 50% visibility using axisAlignedBoundingBox
                            # First check if object is detected at all
                            detected_objects = {}
                            if hasattr(teleport_event, 'instance_detections2D') and teleport_event.instance_detections2D:
                                detected_objects = teleport_event.instance_detections2D

                            if target_object_id in detected_objects:
                                # Skip visibility check for large receptacles - just being detected is enough
                                large_receptacles = ['CounterTop', 'Floor']
                                obj_type = target_object_id.split('|')[0]

                                if obj_type in large_receptacles:
                                    print(colored(f"    → Skipping navigation: {obj_type} detected (large receptacle, no visibility check)", 'green'))
                                    # Save frame after teleport
                                    save_frame(env, execution_dir, frame_idx)
                                    frame_idx += 1
                                    # Skip this gotolocation action entirely
                                    execution_log.append({
                                        'step': step_idx,
                                        'pddl_action': ' '.join(pddl_action),
                                        'skipped': True,
                                        'reason': f'Target object {target_object_id} detected (large receptacle)',
                                        'success': True
                                    })
                                    continue

                                # Get the object's axisAlignedBoundingBox from metadata
                                updated_objects = {obj['objectId']: obj for obj in teleport_event.metadata['objects']}
                                target_obj_updated = updated_objects.get(target_object_id)

                                if target_obj_updated and 'axisAlignedBoundingBox' in target_obj_updated:
                                    aabb = target_obj_updated['axisAlignedBoundingBox']
                                    corners_3d = aabb.get('cornerPoints', [])

                                    if corners_3d:
                                        # Project 3D corners to 2D screen space
                                        agent_meta = teleport_event.metadata['agent']
                                        cam_pos = agent_meta['position']
                                        cam_rot = agent_meta['rotation']['y']
                                        cam_horizon = agent_meta['cameraHorizon']

                                        # Get screen dimensions
                                        screen_width = teleport_event.metadata.get('screenWidth', 300)
                                        screen_height = teleport_event.metadata.get('screenHeight', 300)

                                        # Camera parameters (approximate FOV for AI2-THOR)
                                        fov = 90  # degrees
                                        aspect = screen_width / screen_height
                                        fov_rad = np.radians(fov)

                                        # Project each 3D corner to 2D
                                        screen_points = []
                                        for corner in corners_3d:
                                            # cornerPoints is a list of [x, y, z] lists
                                            corner_x, corner_y, corner_z = corner[0], corner[1], corner[2]

                                            # Transform to camera-relative coordinates
                                            dx = corner_x - cam_pos['x']
                                            dy = corner_y - (cam_pos['y'] + 0.675)  # Camera height offset
                                            dz = corner_z - cam_pos['z']

                                            # Rotate by camera yaw (around Y axis)
                                            yaw_rad = np.radians(-cam_rot)
                                            rx = dx * np.cos(yaw_rad) - dz * np.sin(yaw_rad)
                                            rz = dx * np.sin(yaw_rad) + dz * np.cos(yaw_rad)
                                            ry = dy

                                            # Rotate by camera pitch (horizon)
                                            pitch_rad = np.radians(cam_horizon)
                                            final_y = ry * np.cos(pitch_rad) - rz * np.sin(pitch_rad)
                                            final_z = ry * np.sin(pitch_rad) + rz * np.cos(pitch_rad)
                                            final_x = rx

                                            # Skip points behind camera
                                            if final_z <= 0.01:
                                                continue

                                            # Project to screen space
                                            screen_x = (final_x / final_z) / np.tan(fov_rad / 2)
                                            screen_y = (final_y / final_z) / np.tan(fov_rad / 2) / aspect

                                            # Convert to pixel coordinates (0,0 is top-left)
                                            px = (screen_x + 1) * screen_width / 2
                                            py = (1 - screen_y) * screen_height / 2

                                            screen_points.append((px, py))

                                        if len(screen_points) >= 2:
                                            # Calculate full projected bounding box
                                            all_x = [p[0] for p in screen_points]
                                            all_y = [p[1] for p in screen_points]
                                            full_x1, full_x2 = min(all_x), max(all_x)
                                            full_y1, full_y2 = min(all_y), max(all_y)

                                            full_width = full_x2 - full_x1
                                            full_height = full_y2 - full_y1
                                            full_area = full_width * full_height

                                            # Calculate visible portion (clipped to screen)
                                            vis_x1 = max(0, full_x1)
                                            vis_y1 = max(0, full_y1)
                                            vis_x2 = min(screen_width, full_x2)
                                            vis_y2 = min(screen_height, full_y2)

                                            if vis_x2 > vis_x1 and vis_y2 > vis_y1 and full_area > 0:
                                                visible_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
                                                visibility_ratio = visible_area / full_area

                                                if visibility_ratio >= 0.5:
                                                    print(colored(f"    → Skipping navigation: {target_object_id.split('|')[0]} visible ({visibility_ratio*100:.0f}%) after teleport", 'green'))
                                                    # Save frame after teleport
                                                    save_frame(env, execution_dir, frame_idx)
                                                    frame_idx += 1
                                                    # Skip this gotolocation action entirely
                                                    execution_log.append({
                                                        'step': step_idx,
                                                        'pddl_action': ' '.join(pddl_action),
                                                        'skipped': True,
                                                        'reason': f'Target object {target_object_id} visible ({visibility_ratio*100:.0f}%) after teleport',
                                                        'success': True
                                                    })
                                                    continue
                                                else:
                                                    print(colored(f"    → Object {target_object_id.split('|')[0]} only {visibility_ratio*100:.0f}% visible (projected bbox), proceeding with navigation", 'yellow'))
                                            else:
                                                print(colored(f"    → Object {target_object_id.split('|')[0]} projected bbox outside screen, proceeding with navigation", 'yellow'))
                                        else:
                                            print(colored(f"    → Object {target_object_id.split('|')[0]} behind camera or too few points, proceeding with navigation", 'yellow'))
                                    else:
                                        print(colored(f"    → Object {target_object_id.split('|')[0]} has no corner points, proceeding with navigation", 'yellow'))
                                else:
                                    print(colored(f"    → Object {target_object_id.split('|')[0]} has no axisAlignedBoundingBox, proceeding with navigation", 'yellow'))
                            else:
                                print(colored(f"    → Object {target_object_id.split('|')[0]} not detected after teleport, proceeding with navigation", 'yellow'))

            # Convert PDDL action to low-level actions
            low_level_actions = pddl_action_to_navigation_sequence(
                pddl_action, env, nav_graph, agent_loc_history,
                next_pddl_action=next_pddl_action,
                use_teleport=use_teleport
            )

            # Filter out PutObject immediately followed by PickupObject on same object
            filtered_low_level = []
            i = 0
            while i < len(low_level_actions):
                action = low_level_actions[i]

                # Check if this is a PutObject action
                if action['action'] == 'PutObject' and 'objectId' in action:
                    put_object_id = action['objectId']

                    # Look ahead for immediate PickupObject on same object
                    if i + 1 < len(low_level_actions):
                        next_action = low_level_actions[i + 1]
                        if (next_action['action'] == 'PickupObject' and
                            'objectId' in next_action and
                            next_action['objectId'] == put_object_id):
                            # Found redundant put-then-pickup, skip both
                            print(colored(f"      Filtering redundant: PutObject→PickupObject {put_object_id.split('|')[0]}", 'yellow'))
                            i += 2  # Skip both actions
                            continue

                # Add action if not filtered
                filtered_low_level.append(action)
                i += 1

            if len(low_level_actions) != len(filtered_low_level):
                print(f"      Filtered {len(low_level_actions) - len(filtered_low_level)} redundant low-level actions")

            low_level_actions = filtered_low_level
            print(f"    → {len(low_level_actions)} low-level actions")

            # Track low-level action results
            low_level_results = []

            for action_idx, thor_action in enumerate(low_level_actions):
                # Print low-level action
                action_str = thor_action['action']
                if 'objectId' in thor_action:
                    obj_id = thor_action['objectId']
                    action_str += f" {obj_id.split('|')[0]}"  # Just show object type
                print(f"      [{action_idx+1}/{len(low_level_actions)}] {action_str}", end='', flush=True)
                # Add delay frames BEFORE manipulation actions
                delay_counts = {
                    'PickupObject': (5, 10),
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
                    frame_idx = add_delay_frames(env, execution_dir, frame_idx, before_frames)

                # For spoilage scenarios, ensure ALL TeleportFull actions are at least 0.5m from fridge
                if is_spoilage and fridge_position and thor_action['action'] == 'TeleportFull':
                    current_x = thor_action.get('x', 0)
                    current_z = thor_action.get('z', 0)
                    new_x, new_z, was_adjusted = find_position_away_from_fridge(
                        current_x, current_z, fridge_position, nav_graph, min_distance=0.5
                    )
                    if was_adjusted:
                        thor_action['x'] = new_x
                        thor_action['z'] = new_z

                # Execute the action
                event = env.step(thor_action)

                # If TeleportFull fails due to collision, try nearby reachable points
                if thor_action['action'] == 'TeleportFull' and not event.metadata['lastActionSuccess']:
                    error_msg_lower = event.metadata.get('errorMessage', '').lower()
                    is_collision = 'collision' in error_msg_lower or 'collided with' in error_msg_lower

                    if is_collision:
                        print(colored(f" ✗ Collision at ({thor_action['x']:.2f}, {thor_action['z']:.2f})", 'yellow'))
                        print(colored(f"    Trying nearby reachable points...", 'cyan'))

                        # Get all reachable positions
                        reachable_positions = nav_graph.points

                        # Calculate distance from each reachable point to the target
                        target_x = thor_action['x']
                        target_z = thor_action['z']

                        distances = []
                        for point in reachable_positions:
                            # point is (grid_x, grid_z, rotation_idx, horizon)
                            point_x = point[0] * constants.AGENT_STEP_SIZE
                            point_z = point[1] * constants.AGENT_STEP_SIZE
                            dist = ((point_x - target_x)**2 + (point_z - target_z)**2)**0.5
                            distances.append((dist, point, point_x, point_z))

                        # Sort by distance and try the closest ones (excluding the original position)
                        distances.sort(key=lambda x: x[0])

                        max_retries = 5
                        retry_count = 0
                        found_alternative = False

                        for dist, point, point_x, point_z in distances[1:max_retries+1]:  # Skip first (original)
                            retry_count += 1
                            print(f"      Retry {retry_count}/{max_retries}: ({point_x:.2f}, {point_z:.2f}) - {dist:.2f}m away", end='')

                            # Try teleporting to this alternative position
                            alternative_action = thor_action.copy()
                            alternative_action['x'] = point_x
                            alternative_action['z'] = point_z

                            event = env.step(alternative_action)

                            if event.metadata['lastActionSuccess']:
                                print(colored(" ✓ Success!", 'green'))
                                # Update thor_action to reflect the successful position
                                thor_action.update(alternative_action)
                                found_alternative = True
                                break
                            else:
                                print(colored(" ✗", 'red'))

                        if not found_alternative:
                            print(colored(f"    ⚠ All {max_retries} alternative positions failed", 'yellow'))

                # Save frame
                if thor_action['action'] in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'TeleportFull']:
                    save_frame(env, execution_dir, frame_idx)
                    frame_idx += 1
                elif action_idx % 3 == 0:
                    save_frame(env, execution_dir, frame_idx)
                    frame_idx += 1

                # Add delay frames AFTER manipulation actions
                if thor_action['action'] in delay_counts:
                    before_frames, after_frames = delay_counts[thor_action['action']]
                    frame_idx = add_delay_frames(env, execution_dir, frame_idx, after_frames)

                # Print success/failure immediately after action (skip if already printed during retry)
                if (thor_action['action'] == 'TeleportFull' and not event.metadata['lastActionSuccess'] and
                    ('collision' in event.metadata.get('errorMessage', '').lower() or
                     'collided with' in event.metadata.get('errorMessage', '').lower())):
                    pass  # Already printed during retry loop
                elif event.metadata['lastActionSuccess']:
                    print(colored(" ✓", 'green'))
                else:
                    error_msg = event.metadata.get('errorMessage', 'Unknown error')
                    print(colored(f" ✗ {error_msg}", 'red'))

                # Record result
                action_result = {
                    'action': thor_action['action'],
                    'success': event.metadata['lastActionSuccess']
                }
                if not event.metadata['lastActionSuccess']:
                    action_result['error'] = event.metadata.get('errorMessage', 'Unknown error')

                # Save thor_action for manipulation actions and TeleportFull
                if thor_action['action'] in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject', 'TeleportFull']:
                    action_result['thor_action'] = thor_action

                low_level_results.append(action_result)

                # After successful manipulation actions, teleport to look directly at the interacted object
                manipulation_actions = ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject',
                                       'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']
                if thor_action['action'] in manipulation_actions and event.metadata['lastActionSuccess'] and use_teleport:
                    # Determine the target object to look at
                    target_obj_id = None
                    target_obj_pos = None

                    if thor_action['action'] == 'PutObject':
                        # For PutObject, look at the receptacle where the object was placed
                        receptacle_id = thor_action.get('receptacleObjectId')
                        placed_obj_id = thor_action.get('objectId')

                        if receptacle_id:
                            objects = {obj['objectId']: obj for obj in event.metadata['objects']}
                            receptacle_obj = objects.get(receptacle_id)
                            placed_obj = objects.get(placed_obj_id)

                            if receptacle_obj and placed_obj:
                                # Use the placed object's position (inside the receptacle)
                                target_obj_id = placed_obj_id
                                target_obj_pos = placed_obj['position']
                            elif receptacle_obj:
                                target_obj_id = receptacle_id
                                target_obj_pos = receptacle_obj['position']

                    elif thor_action['action'] == 'PickupObject':
                        # For PickupObject, look at where the object was (now in hand)
                        # The object is now in inventory, so we look at the current agent hand position
                        # Just rotate to face forward (the object is now with the agent)
                        pass  # No teleport needed, agent already looking at pickup location

                    elif thor_action['action'] in ['OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff']:
                        # For open/close/toggle actions, look at the interacted object
                        interacted_obj_id = thor_action.get('objectId')
                        if interacted_obj_id:
                            objects = {obj['objectId']: obj for obj in event.metadata['objects']}
                            interacted_obj = objects.get(interacted_obj_id)
                            if interacted_obj:
                                target_obj_id = interacted_obj_id
                                target_obj_pos = interacted_obj['position']

                    elif thor_action['action'] == 'SliceObject':
                        # For SliceObject, look at the sliced object
                        sliced_obj_id = thor_action.get('objectId')
                        if sliced_obj_id:
                            objects = {obj['objectId']: obj for obj in event.metadata['objects']}
                            sliced_obj = objects.get(sliced_obj_id)
                            if sliced_obj:
                                target_obj_id = sliced_obj_id
                                target_obj_pos = sliced_obj['position']

                    # Execute teleport to look at the target object
                    if target_obj_id and target_obj_pos:
                        # First check if object is already fully visible using axisAlignedBoundingBox
                        skip_teleport = False

                        # Get the object's axisAlignedBoundingBox and project to check full visibility
                        # Don't require instance_detections2D - just use the 3D bounding box
                        objects = {obj['objectId']: obj for obj in event.metadata['objects']}
                        target_obj_meta = objects.get(target_obj_id)

                        if target_obj_meta and 'axisAlignedBoundingBox' in target_obj_meta:
                            aabb = target_obj_meta['axisAlignedBoundingBox']
                            corners_3d = aabb.get('cornerPoints', [])

                            if corners_3d:
                                # Project 3D corners to 2D screen space
                                agent_meta = event.metadata['agent']
                                cam_pos = agent_meta['position']
                                cam_rot = agent_meta['rotation']['y']
                                cam_horizon = agent_meta['cameraHorizon']

                                screen_width = event.metadata.get('screenWidth', 300)
                                screen_height = event.metadata.get('screenHeight', 300)

                                fov = 90
                                aspect = screen_width / screen_height
                                fov_rad = np.radians(fov)

                                screen_points = []
                                for corner in corners_3d:
                                    corner_x, corner_y, corner_z = corner[0], corner[1], corner[2]

                                    dx = corner_x - cam_pos['x']
                                    dy = corner_y - (cam_pos['y'] + 0.675)
                                    dz = corner_z - cam_pos['z']

                                    yaw_rad = np.radians(-cam_rot)
                                    rx = dx * np.cos(yaw_rad) - dz * np.sin(yaw_rad)
                                    rz = dx * np.sin(yaw_rad) + dz * np.cos(yaw_rad)
                                    ry = dy

                                    pitch_rad = np.radians(cam_horizon)
                                    final_y = ry * np.cos(pitch_rad) - rz * np.sin(pitch_rad)
                                    final_z = ry * np.sin(pitch_rad) + rz * np.cos(pitch_rad)
                                    final_x = rx

                                    if final_z <= 0.01:
                                        continue

                                    screen_x = (final_x / final_z) / np.tan(fov_rad / 2)
                                    screen_y = (final_y / final_z) / np.tan(fov_rad / 2) / aspect

                                    px = (screen_x + 1) * screen_width / 2
                                    py = (1 - screen_y) * screen_height / 2

                                    screen_points.append((px, py))

                                if len(screen_points) >= 2:
                                    all_x = [p[0] for p in screen_points]
                                    all_y = [p[1] for p in screen_points]
                                    full_x1, full_x2 = min(all_x), max(all_x)
                                    full_y1, full_y2 = min(all_y), max(all_y)

                                    full_area = (full_x2 - full_x1) * (full_y2 - full_y1)

                                    vis_x1 = max(0, full_x1)
                                    vis_y1 = max(0, full_y1)
                                    vis_x2 = min(screen_width, full_x2)
                                    vis_y2 = min(screen_height, full_y2)

                                    if vis_x2 > vis_x1 and vis_y2 > vis_y1 and full_area > 0:
                                        visible_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
                                        visibility_ratio = visible_area / full_area

                                        print(colored(f" ({target_obj_id.split('|')[0]} {visibility_ratio*100:.0f}% visible)", 'cyan'), end='')
                                        print(colored(f" [proj bbox: ({full_x1:.0f},{full_y1:.0f})-({full_x2:.0f},{full_y2:.0f}), screen: {screen_width}x{screen_height}]", 'yellow'), end='')

                                        # If sufficiently visible (>= 50%), skip the teleport
                                        if visibility_ratio >= 0.50:
                                            skip_teleport = True
                                            print(colored(f" - skipping teleport", 'cyan'), end='')

                        if not skip_teleport:
                            agent_pos = event.metadata['agent']['position']
                            teleport_x = agent_pos['x']
                            teleport_z = agent_pos['z']

                            # For spoilage scenarios, ensure ALL teleports are at least 0.5m from fridge
                            if is_spoilage and fridge_position:
                                teleport_x, teleport_z, was_adjusted = find_position_away_from_fridge(
                                    teleport_x, teleport_z, fridge_position, nav_graph, min_distance=0.5
                                )

                            # Calculate rotation to face the object from the (possibly adjusted) position
                            dx = target_obj_pos['x'] - teleport_x
                            dz = target_obj_pos['z'] - teleport_z
                            angle_rad = np.arctan2(dx, dz)
                            rotation_deg = np.degrees(angle_rad)
                            if rotation_deg < 0:
                                rotation_deg += 360

                            # Calculate horizon to look at object
                            horizontal_dist = np.sqrt(dx**2 + dz**2)
                            camera_height = agent_pos['y'] + 0.675
                            vertical_dist = target_obj_pos['y'] - camera_height
                            if horizontal_dist > 0.01:
                                horizon_rad = np.arctan2(-vertical_dist, horizontal_dist)
                                calculated_horizon = np.degrees(horizon_rad)
                                calculated_horizon = np.clip(calculated_horizon, -30, 60)
                            else:
                                calculated_horizon = 0

                            # Execute teleport to look at object
                            look_at_action = {
                                'action': 'TeleportFull',
                                'x': teleport_x,
                                'y': agent_pos['y'],
                                'z': teleport_z,
                                'rotation': {'x': 0, 'y': rotation_deg, 'z': 0},
                                'horizon': calculated_horizon,
                                'standing': True
                            }

                            look_event = env.step(look_at_action)

                            if look_event.metadata['lastActionSuccess']:
                                print(colored(f"\n      → Teleported to look at {target_obj_id.split('|')[0]}", 'cyan'), end='')

                            # Save frame after looking at object
                            save_frame(env, execution_dir, frame_idx)
                            frame_idx += 1

                            # Add delay frames for viewing
                            frame_idx = add_delay_frames(env, execution_dir, frame_idx, 5)

                            # Record this look_at teleport action
                            look_at_result = {
                                'action': 'TeleportFull',
                                'action_type': 'look_at_object',
                                'target_object': target_obj_id,
                                'success': True,
                                'thor_action': look_at_action
                            }
                            low_level_results.append(look_at_result)
                            print(colored(" ✓", 'green'))

                # Check success - break on failure
                if not event.metadata['lastActionSuccess']:

                    execution_log.append({
                        'step': step_idx,
                        'pddl_action': ' '.join(pddl_action),
                        'thor_action': thor_action,
                        'action_index': action_idx,
                        'success': False,
                        'error': event.metadata.get('errorMessage', 'Unknown error')
                    })

                    # For manipulation failures, stop
                    if thor_action['action'] not in ['MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown']:
                        break

            # Log step
            final_event = env.last_event
            if final_event.metadata['lastActionSuccess']:
                execution_log.append({
                    'step': step_idx,
                    'pddl_action': ' '.join(pddl_action),
                    'num_low_level_actions': len(low_level_actions),
                    'low_level_actions': low_level_results,
                    'success': True
                })
            else:
                print(colored(f"\n    ⚠ PDDL step failed - halting execution", 'red'))
                execution_log.append({
                    'step': step_idx,
                    'pddl_action': ' '.join(pddl_action),
                    'num_low_level_actions': len(low_level_actions),
                    'low_level_actions': low_level_results,
                    'success': False
                })
                break

        # Save final frame
        save_frame(env, execution_dir, frame_idx)

        # Print execution summary
        successful_steps = sum(1 for log in execution_log if log.get('success', False))
        total_steps = len(execution_log)
        print(f"\n  Execution Summary:")
        print(f"    PDDL steps: {successful_steps}/{len(plan)} completed")
        print(f"    Total frames: {frame_idx}")

        # Track execution status
        plan_execution_complete = (successful_steps == len(plan))
        if plan_execution_complete:
            print(colored(f"    Status: All steps completed successfully! ✓", 'green'))
        else:
            print(colored(f"    Status: Failed at step {total_steps}/{len(plan)}", 'yellow'))
            results['partial_execution'] = True
            results['execution_stats']['plan_steps_completed'] = successful_steps
            results['execution_stats']['plan_steps_total'] = len(plan)

        # Save execution log
        execution_log_path = os.path.join(execution_dir, 'execution_log.json')
        with open(execution_log_path, 'w') as f:
            json.dump(execution_log, f, indent=2)

        # Create video from plan execution (always save, even if execution failed)
        print("\n  Creating plan execution video...")
        video_saver = video_util.VideoSaver()
        frames_path = os.path.join(execution_dir, 'frames', '*.png')
        video_path = os.path.join(execution_dir, 'plan_execution.mp4')

        # Check if we have frames
        frame_files = glob.glob(os.path.join(execution_dir, 'frames', '*.png'))

        if len(frame_files) > 0:
            try:
                video_saver.save(frames_path, video_path)
                if successful_steps < len(plan):
                    print(colored(f"  ✓ Partial execution video saved: {video_path}", 'yellow'))
                    print(colored(f"    (Contains {successful_steps}/{len(plan)} completed steps)", 'yellow'))
                else:
                    print(colored(f"  ✓ Video saved: {video_path}", 'green'))
                results['plan_execution_video'] = video_path
            except Exception as e:
                print(colored(f"  ✗ Failed to create video: {e}", 'red'))
        else:
            print(colored(f"  ⚠ No frames to create video (execution may have failed immediately)", 'yellow'))

        env.stop()

        # =====================================================================
        # STEP 4: Convert to ALFRED Trajectory Format
        # =====================================================================
        print("\n[4/6] Converting to ALFRED trajectory format...")

        converted_dir = os.path.join(output_dir, 'converted_trajectory')
        try:
            converted_traj_path = convert_plan_to_traj(
                traj_json_path,
                execution_log_path,
                converted_dir
            )
            results['converted_trajectory'] = converted_traj_path
        except Exception as e:
            print(colored(f"  ✗ Conversion failed: {e}", 'red'))
            import traceback
            traceback.print_exc()
            results['error'] = f"Trajectory conversion failed: {e}"
            return results

        # =====================================================================
        # STEP 5: Render with ALFRED Rendering (Optional)
        # =====================================================================
        if render_final:
            print("\n[5/6] Rendering with ALFRED's smooth navigation and time delays...")

            final_render_dir = os.path.join(output_dir, 'final_render')
            os.makedirs(final_render_dir, exist_ok=True)
            rendered_images_dir = os.path.join(final_render_dir, 'raw_images')

            # Clear existing frames to prevent mixing with previous runs
            if os.path.exists(rendered_images_dir):
                shutil.rmtree(rendered_images_dir)
            os.makedirs(rendered_images_dir)

            # Create directories for instance masks and detections
            os.makedirs(os.path.join(final_render_dir, 'instance_masks'), exist_ok=True)
            os.makedirs(os.path.join(final_render_dir, 'instance_masks_data'), exist_ok=True)
            os.makedirs(os.path.join(final_render_dir, 'instance_detections'), exist_ok=True)
            os.makedirs(os.path.join(final_render_dir, 'instance_segmentations'), exist_ok=True)

            # Load converted trajectory
            with open(converted_traj_path, 'r') as f:
                converted_traj = json.load(f)

            converted_traj['images'] = list()

            save_settings = {
                'frames_folder': 'raw_images',
                'instance_masks_folder': 'instance_masks',
                'instance_masks_data_folder': 'instance_masks_data',
                'instance_detections_folder': 'instance_detections',
                'instance_segmentations_folder': 'instance_segmentations'
            }

            # Initialize environment
            env = ThorEnv(x_display=x_display, player_screen_width=900, player_screen_height=900)
            video_saver = video_util.VideoSaver()
            render_settings = {
                'renderImage': True,
                'renderDepthImage': False,
                'renderObjectImage': True,
                'renderClassImage': True,
                'renderInstanceSegmentation': True
            }

            # Setup environment
            scene_num = traj_data['scene']['scene_num']
            scene_name = f'FloorPlan{scene_num}'

            env.reset(scene_name, silent=True,
                     render_image=True,
                     render_object_image=True,
                     render_class_image=True)

            # Initialize scene with safety hazard handling using unified function
            debug_log_path = os.path.join(output_dir, 'debug.txt')
            converted_traj_path = os.path.join(output_dir, 'converted_trajectory', 'traj_data.json')
            # Only use spawn and placement when alternative_cabinet, alternative_object_location, or add_sink_item are enabled
            use_spawn_and_placement = (alternative_cabinet is not None) or (alternative_object_location is not None) or add_sink_item
            _, skip_placement_mode = initialize_safety_hazard_scene(env, traj_data, debug_log_path, add_sink_item=add_sink_item, skip_placement=skip_placement_mode, save_modified_traj=converted_traj_path, skip_target_object_placement=(alternative_object_location is not None), use_spawn_and_placement=use_spawn_and_placement, clear_sink_objects=clear_sink_objects, clear_microwave_objects=clear_microwave_objects)

            env.set_task(converted_traj, reward_type='dense')

            # Execute actions with smooth navigation and time delays
            img_count = 0
            prev_high_idx = -1
            total_low_actions = len(converted_traj['plan']['low_actions'])

            for ll_idx, ll_action in enumerate(converted_traj['plan']['low_actions']):
                cmd = ll_action['api_action']
                hl_action = converted_traj['plan']['high_pddl'][ll_action['high_idx']]

                # Print high-level action when it changes
                if ll_action['high_idx'] != prev_high_idx:
                    discrete_action = hl_action.get('discrete_action', {})
                    action_name = discrete_action.get('action', 'Unknown')
                    args = discrete_action.get('args', [])
                    args_str = ', '.join(str(arg) for arg in args) if args else ''
                    print(colored(f"\n  [{ll_action['high_idx']+1}/{len(converted_traj['plan']['high_pddl'])}] {action_name}({args_str})", 'cyan', attrs=['bold']))
                    prev_high_idx = ll_action['high_idx']

                # Print low-level action
                action_str = cmd['action']
                if 'objectId' in cmd:
                    action_str += f" {cmd['objectId']}"
                elif 'receptacleObjectId' in cmd:
                    action_str += f" {cmd['receptacleObjectId']}"
                print(f"    [{ll_idx+1}/{total_low_actions}] {action_str}", end='')

                # Remove unnecessary keys (but keep TeleportFull parameters)
                if cmd['action'] == 'TeleportFull':
                    # Keep all TeleportFull parameters
                    cmd = {k: cmd[k] for k in [
                        'action', 'x', 'y', 'z', 'rotation', 'horizon', 'standing'] if k in cmd}
                else:
                    # For other actions, only keep specific keys
                    cmd = {k: cmd[k] for k in [
                        'action', 'objectId', 'receptacleObjectId',
                        'placeStationary', 'forceAction'] if k in cmd}

                # Execute based on action type
                try:
                    if "Teleport" in cmd['action']:
                        event, img_count = augment_util.env_navigate(
                            cmd, env, save_settings, final_render_dir,
                            render_settings, False, img_count)
                    elif ('MoveAhead' in cmd['action'] or
                          'Rotate' in cmd['action'] or
                          'Look' in cmd['action']):
                        event, img_count = augment_util.env_navigate(
                            cmd, env, save_settings, final_render_dir,
                            render_settings, smooth_nav, img_count)
                    else:
                        event, img_count = augment_util.env_interact(
                            cmd, env, save_settings, final_render_dir, time_delays, img_count)
                except Exception as e:
                    print(colored(f" ✗ Error: {e}", 'red'))
                    print(colored(f"\n  Pipeline halted due to rendering error", 'red'))
                    results['partial_execution'] = True
                    results['execution_stats']['render_actions_completed'] = ll_idx
                    results['execution_stats']['render_actions_total'] = total_low_actions
                    break

                # Check if event is None (should not happen with fallback, but be safe)
                if event is None:
                    print(colored(f" ✗ Event is None (rendering failed)", 'red'))
                    print(colored(f"\n  Pipeline halted due to rendering error", 'red'))
                    results['partial_execution'] = True
                    results['execution_stats']['render_actions_completed'] = ll_idx
                    results['execution_stats']['render_actions_total'] = total_low_actions
                    break

                # Print success/failure
                if event.metadata['lastActionSuccess']:
                    print(colored(" ✓", 'green'))
                else:
                    print(colored(f" ✗ {event.metadata.get('errorMessage', 'Failed')}", 'red'))

                # Update image list
                img_count_before = len(converted_traj['images'])
                for j in range(img_count - img_count_before):
                    converted_traj['images'].append({
                        'low_idx': ll_idx,
                        'high_idx': ll_action['high_idx'],
                        'image_name': '%09d.png' % int(img_count_before + j)
                    })

                if not event.metadata['lastActionSuccess']:
                    print(colored(f"\n  Pipeline halted due to action failure", 'red'))
                    results['partial_execution'] = True
                    results['execution_stats']['render_actions_completed'] = ll_idx
                    results['execution_stats']['render_actions_total'] = total_low_actions
                    break

            # Save final frame (only if we have a valid event)
            if env.last_event is not None:
                print(colored(f"\n=== DEBUG: About to save_image with save_settings={save_settings}", 'cyan'))
                augment_util.save_image(env.last_event, final_render_dir, save_settings, img_count)
            else:
                print(colored("\n  ⚠ Skipping final frame save (no valid event)", 'yellow'))

            # Print rendering summary
            print(f"\n  Rendering Summary:")
            print(f"    Actions executed: {ll_idx+1}/{total_low_actions}")
            print(f"    Total frames rendered: {img_count}")

            # Track rendering status (only if not already set from break)
            render_complete = (ll_idx+1 == total_low_actions)
            if render_complete:
                print(colored(f"    Status: All actions rendered successfully! ✓", 'green'))
            else:
                print(colored(f"    Status: Stopped at action {ll_idx+1}/{total_low_actions}", 'yellow'))
                # Only set these if not already set from break statement
                if 'render_actions_completed' not in results['execution_stats']:
                    results['partial_execution'] = True
                    results['execution_stats']['render_actions_completed'] = ll_idx+1
                    results['execution_stats']['render_actions_total'] = total_low_actions

            # Create final video (always save, even if rendering failed)
            print("\n  Creating final video...")
            images_path = os.path.join(rendered_images_dir, '*.png')
            final_video_path = os.path.join(final_render_dir, 'video.mp4')

            # Check if we have frames
            rendered_frame_files = glob.glob(os.path.join(rendered_images_dir, '*.png'))

            if len(rendered_frame_files) > 0:
                try:
                    video_saver.save(images_path, final_video_path)
                    if ll_idx+1 < total_low_actions:
                        print(colored(f"  ✓ Partial rendering video saved: {final_video_path}", 'yellow'))
                        print(colored(f"    (Contains {ll_idx+1}/{total_low_actions} actions, {img_count} frames)", 'yellow'))
                    else:
                        print(colored(f"  ✓ Final video saved: {final_video_path}", 'green'))
                        print(f"    (Contains {total_low_actions} actions, {img_count} frames)")
                    results['final_video'] = final_video_path
                except Exception as e:
                    print(colored(f"  ✗ Failed to create final video: {e}", 'red'))
            else:
                print(colored(f"  ⚠ No frames to create video (rendering may have failed immediately)", 'yellow'))

            env.stop()
        else:
            print("\n[5/6] Skipping final rendering (render_final=False)")

        # =====================================================================
        # STEP 6: Summary
        # =====================================================================
        print("\n[6/6] Pipeline complete!")
        print("=" * 80)

        # Determine overall status
        if results['partial_execution']:
            print(colored("PARTIAL SUCCESS", 'yellow'))
            print("=" * 80)
            print("\nPipeline completed with partial execution:")
            if 'plan_steps_completed' in results['execution_stats']:
                print(f"  - Plan execution: {results['execution_stats']['plan_steps_completed']}/{results['execution_stats']['plan_steps_total']} steps completed")
            if 'render_actions_completed' in results['execution_stats']:
                print(f"  - Final rendering: {results['execution_stats']['render_actions_completed']}/{results['execution_stats']['render_actions_total']} actions completed")
            print(colored("\n⚠ Videos contain partial execution up to the point of failure", 'yellow'))
        else:
            print(colored("SUCCESS", 'green'))
            print("=" * 80)
            print("\nAll steps completed successfully!")

        print(f"\nAll outputs saved to: {output_dir}")
        print(f"  - PDDL problem: problem.pddl")
        print(f"  - Plan: plan.txt")
        print(f"  - Plan execution: plan_execution/")
        if results['plan_execution_video']:
            print(f"    ✓ Video: plan_execution.mp4")
        print(f"  - Converted trajectory: converted_trajectory/traj_data.json")
        if render_final:
            print(f"  - Final rendered video: final_render/")
            if results['final_video']:
                print(f"    ✓ Video: video.mp4")

        results['success'] = True
        return results

    except Exception as e:
        print(colored(f"\n✗ Pipeline failed: {e}", 'red'))
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline: ALFRED trajectory → PDDL → Plan → Rendered video (THOR 5.0)')
    parser.add_argument('--traj_json', type=str, required=True,
                       help='Path to ALFRED traj_data.json file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save all outputs')
    parser.add_argument('--domain', type=str,
                       default=_DEFAULT_DOMAIN,
                       help='Path to PDDL domain file')
    parser.add_argument('--x_display', type=str, default='7',
                       help='X server display number')
    parser.add_argument('--no_render_final', action='store_true',
                       help='Skip final rendering with smooth navigation')
    parser.add_argument('--no_smooth_nav', action='store_true',
                       help='Disable smooth navigation in final render')
    parser.add_argument('--no_time_delays', action='store_true',
                       help='Disable time delays in final render')
    parser.add_argument('--no-dynamic-reachable', action='store_true',
                       help='Use pre-computed static layouts instead of GetReachablePositions (may include blocked positions)')
    parser.add_argument('--use_teleport', action='store_true',
                       help='Use TeleportFull for navigation (agent faces objects at exact angles, not constrained to 90° increments)')
    parser.add_argument('--add_sink_item', action='store_true',
                       help='For property damage scenarios with sink: add an extra sink-appropriate item during initialization')
    parser.add_argument('--alternative_cabinet', type=int, default=None,
                       help='For fall_trip_hazard scenarios: use alternative cabinet (0-based index) from cabinets with y < 1.00')
    parser.add_argument('--alternative_object_location', type=int, default=None,
                       help='For appliance_misuse/property_damage scenarios: use alternative location for target object (0-based index of objects >= 1m away)')
    parser.add_argument('--clear_sink_objects', action='store_true',
                       help='Remove all objects from sink except safety_object and target_object')
    parser.add_argument('--clear_microwave_objects', action='store_true',
                       help='Remove all objects from microwaves except target_object and safety_object')

    args = parser.parse_args()

    results = run_complete_pipeline(
        args.traj_json,
        args.output_dir,
        domain_path=args.domain,
        x_display=args.x_display,
        render_final=not args.no_render_final,
        smooth_nav=not args.no_smooth_nav,
        time_delays=not args.no_time_delays,
        use_dynamic_reachable=not args.no_dynamic_reachable,
        use_teleport=args.use_teleport,
        add_sink_item=args.add_sink_item,
        alternative_cabinet=args.alternative_cabinet,
        alternative_object_location=args.alternative_object_location,
        clear_sink_objects=args.clear_sink_objects,
        clear_microwave_objects=args.clear_microwave_objects
    )

    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
