#!/usr/bin/env python3
"""
Complete pipeline: ALFRED trajectory → PDDL → Plan → Render video (Thor 5.0)

This script takes an ALFRED trajectory and:
1. Generates a PDDL problem from it
2. Plans using Fast Downward
3. Executes the plan in THOR 5.0
4. Converts to ALFRED trajectory format
5. Renders with smooth navigation and time delays

Usage:
    python pipeline_pddl_to_video_thor5.py --traj_json <path> --output_dir <path>
"""

import os
import sys
import json
import argparse
import shutil
import glob
from termcolor import colored

# Add ALFRED paths
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'gen'))

# Add E.T. gen directory for imports
et_gen_dir = '/home/josue/Desktop/Research/SLED/MSS/E.T./alfred/gen'
sys.path.insert(0, et_gen_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alfred.env.thor_env_thor5 import ThorEnv
from alfred.gen import constants
from alfred.gen.utils import video_util, game_util, augment_util
from alfred.gen.graph.graph_obj import Graph
from generate_problem_pddl_full_thor5 import generate_pddl_from_traj_full

# Import DANLI planner
import importlib.util
danli_planner_path = '/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/planner.py'
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


def run_complete_pipeline(
    traj_json_path,
    output_dir,
    domain_path='/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/domain.pddl',
    x_display='7',
    render_final=True,
    smooth_nav=True,
    time_delays=True,
    use_dynamic_reachable=True
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

    Returns:
        dict: Results with paths to all outputs
    """

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PDDL PLANNING AND RENDERING PIPELINE")
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
                use_dynamic_reachable=use_dynamic_reachable
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
                fd_path='/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/fast-downward-24.06.1/fast-downward.py',
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

        if "toggle_object" in traj_data["scene"] and traj_data["scene"]["toggle_object"] and traj_data["scene"]["toggle_object"]["setup_toggle"]:
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

        for i in range(1/0.05):
            env.noop()

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

        for step_idx, pddl_action in enumerate(plan, 1):
            print(colored(f"\n  [{step_idx}/{len(plan)}] {' '.join(pddl_action)}", 'cyan', attrs=['bold']))

            # Convert PDDL action to low-level actions
            low_level_actions = pddl_action_to_navigation_sequence(
                pddl_action, env, nav_graph, agent_loc_history
            )

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

                # Execute the action
                event = env.step(thor_action)

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

                # Print success/failure immediately after action
                if event.metadata['lastActionSuccess']:
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

                # Save thor_action for manipulation actions
                if thor_action['action'] in ['PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff', 'SliceObject']:
                    action_result['thor_action'] = thor_action

                low_level_results.append(action_result)

                # Check success - break on failure
                if not event.metadata['lastActionSuccess']:

                    execution_log.append({
                        'step': step_idx,
                        'pddl_action': ' '.join(pddl_action),
                        'thor_action': thor_action,
                        'action_index': action_idx,
                        'success': False,
                        'error': error_msg
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

            # Load converted trajectory
            with open(converted_traj_path, 'r') as f:
                converted_traj = json.load(f)

            converted_traj['images'] = list()

            save_settings = {'frames_folder': 'raw_images'}

            # Initialize environment
            env = ThorEnv(x_display=x_display, player_screen_width=300, player_screen_height=300)
            video_saver = video_util.VideoSaver()
            render_settings = {
                'renderImage': True,
                'renderDepthImage': False,
                'renderObjectImage': False,
                'renderClassImage': False
            }

            # Setup environment
            scene_num = converted_traj['scene']['scene_num']
            scene_name = f'FloorPlan{scene_num}'
            env.reset(scene_name, silent=True)

            object_poses = converted_traj['scene']['object_poses']
            object_toggles = converted_traj['scene']['object_toggles']
            dirty_and_empty = converted_traj['scene']['dirty_and_empty']

            if "toggle_object" in converted_traj["scene"] and converted_traj["scene"]["toggle_object"] and traj_data["scene"]["toggle_object"]["setup_toggle"]:
                toggle_object = converted_traj['scene']['toggle_object']
            else:
                toggle_object = None

            env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object)

            init_action = converted_traj['scene']['init_action']
            if isinstance(init_action, list):
                for act in init_action:
                    if act:
                        env.step(dict(act))
            else:
                env.step(dict(init_action))

            for i in range(1/0.05):
                env.noop()

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

                # Remove unnecessary keys
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
        description='Complete pipeline: ALFRED trajectory → PDDL → Plan → Rendered video')
    parser.add_argument('--traj_json', type=str, required=True,
                       help='Path to ALFRED traj_data.json file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save all outputs')
    parser.add_argument('--domain', type=str,
                       default='/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/domain.pddl',
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

    args = parser.parse_args()

    results = run_complete_pipeline(
        args.traj_json,
        args.output_dir,
        domain_path=args.domain,
        x_display=args.x_display,
        render_final=not args.no_render_final,
        smooth_nav=not args.no_smooth_nav,
        time_delays=not args.no_time_delays,
        use_dynamic_reachable=not args.no_dynamic_reachable
    )

    return 0 if results['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
