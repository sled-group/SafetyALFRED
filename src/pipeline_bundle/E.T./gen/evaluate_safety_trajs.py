import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import sys
import json
import numpy as np
import threading
import time
import copy
import random
import glob
import shutil

from termcolor import colored
from sacred import Ingredient, Experiment

from env.thor_env import ThorEnv
from gen import constants
from gen.utils import augment_util, video_util
from utils import helper_util, model_util


def extract_safety_category(filename):
    """
    Extract safety category from filename.
    Expected pattern: traj_data_safety_traj_{CATEGORY}_{object_info}.json
    Example: traj_data_safety_traj_fire_hazard_StoveKnob|+02.02|+01.04|-01.51_0.json -> fire_hazard
    """
    if 'safety_traj_' not in filename:
        return None
    
    # Find the part after 'safety_traj_'
    parts = filename.split('safety_traj_')
    if len(parts) < 2:
        return None
    
    # Get everything after 'safety_traj_' and split by '_'
    after_safety_traj = parts[1]
    
    # Handle different safety category patterns
    if after_safety_traj.startswith('fire_hazard'):
        return 'fire_hazard'
    elif after_safety_traj.startswith('appliance_misuse'):
        return 'appliance_misuse'
    elif after_safety_traj.startswith('property_damage'):
        return 'property_damage'
    elif after_safety_traj.startswith('fall_trip_hazard'):
        return 'fall_trip_hazard'
    elif after_safety_traj.startswith('unsanitary'):
        return 'unsanitary'
    elif after_safety_traj.startswith('spoilage'):
        return 'spoilage'
    else:
        # Fallback: try to extract the first part before the next '_'
        category_parts = after_safety_traj.split('_')
        if len(category_parts) >= 2 and category_parts[0] in ['fire', 'appliance', 'property', 'fall']:
            return '_'.join(category_parts[:2])  # e.g., fire_hazard
        return category_parts[0] if category_parts else None


args_ingredient = Ingredient('args')
ex = Experiment('render_trajs', ingredients=[args_ingredient])


@args_ingredient.config
def cfg_args():
    # dataset folder to dump frames to
    data_output = 'evaluate_safety_2.1.0'
    # dataset folder to load jsons from
    data_input = 'generated_safety_2.1.0'
    # smooth naviagation (like the original data)
    smooth_nav = True
    # time delays (like the original data)
    time_delays = True
    # whether to shuffle the order of augmenting
    shuffle = False
    # number of threads to start in parallel
    num_threads = 72
    # frame size to render
    render_size = 300
    # X server number
    x_display = '7'
    # render and save RGB images
    render_frames = False  # Will be overridden by debug_render
    # render and save depth images
    render_depth = False
    # render and save class segmentation masks
    render_class_masks = False
    # render and save instance segmentation masks
    render_instance_masks = False
    # save object bounding boxes
    save_detections = False
    # partitions to render data for
    partitions = ('valid_unseen', 'valid_seen', 'train')
    # whether to overwrite data folder if it already exists
    overwrite = False
    # whether to perform safety verification during trajectory execution
    verify_during_execution = False
    # whether to render and save images/videos for debugging purposes
    debug_render = True

#ACTIONS
def place_object_in(env, objectID, receptacleID, traj_data):
    object_id = objectID
    # receptacle_id = "Microwave|-00.37|+01.11|+00.43"
    # receptacle_id = "Sink|-01.39|+00.98|+00.44|SinkBasin"
    receptacle_id = receptacleID
    objects = {obj["objectId"]: obj for obj in env.last_event.metadata["objects"]}
    obj = objects.get(object_id)
    receptacle = objects.get(receptacle_id)
    # print("OBJ:", obj)
    receptacle_position = receptacle["position"]
    receptacle_rotation = receptacle["rotation"]
    print(receptacle)
    #Must do this because using PlaceObjectAtPoint only works if object is not pickupable
    objects = env.last_event.metadata["objects"]
    # env.step(dict(action="GetSpawnCoordinatesAboveReceptacle", objectId=receptacle_id, anywhere=True))
    # reachable_positions = env.last_event.metadata["actionReturn"]
    # print("REACHABLE POS:", len(reachable_positions))
    # env.step(dict(
    #     action="PickupObject",
    #     objectId=object_id,
    #     forceAction=True,
    #     manualInteract=False
    # ))
    # event1 = env.last_event
    # env.step(dict(
    #     action="OpenObject",
    #     objectId=receptacle_id,
    #     openness=0.1,
    #     forceAction=True
    # ))
    # env.step(dict(
    #     action="PutObject",
    #     objectId=receptacle_id,
    #     forceAction=True,
    #     placeStationary=True
    # ))
    # print("SUCCESS:", env.last_event.metadata["lastAction"], env.last_event.metadata["lastActionSuccess"])
    # event3 = env.last_event
    # env.step(dict(
    #     action="CloseObject",
    #     objectId=receptacle_id,
    #     forceAction=True
    # ))
    #This only seems to work if picking up the object and plaing it manually does not work
    # if not (event1.metadata["lastActionSuccess"] and event3.metadata["lastActionSuccess"]):
    #     #calculate closest position 
    #     receptacle_pos = np.array([receptacle_position["x"], receptacle_position["z"]])  # Ignore y-axis for navigation
    #     reachable_positions_np = np.array([[p["x"], p["z"]] for p in reachable_positions])
    #     distances = np.linalg.norm(reachable_positions_np - receptacle_pos, axis=1)
    #     closest_index = np.argmin(distances)
    #     closest_position = reachable_positions[closest_index]
    #     #place object
    #     env.step(dict(action='PlaceObjectAtPoint', 
    #             objectId=object_id, 
    #             position=closest_position))
    #     print("MOVE SUCCESS:", env.last_event.metadata["lastActionSuccess"])
    # if not (event1.metadata["lastActionSuccess"] and event3.metadata["lastActionSuccess"]):/
    if True:
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        # for objs in object_poses:
        #     if objs['objectName'] == obj["name"]:
        #         objs["position"]["x"] = obj['position']['x']
        #         objs["position"]["y"] = obj["position"]["y"]
        #         objs["position"]["z"] = obj["position"]["z"]
        #         print(objs)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        # objects = {obj["objectId"]: obj for obj in env.last_event.metadata["objects"]}
        # obj = objects.get(object_id)
        # obj_pos = env.last_event.metadata["objects"]
        # print(env.last_event[""])
        # # env.reset("FloorPlan16")
        # js_obj = {"objectName": obj["name"], "rotation":{"y":0,"x":0,"z":0},"position":{"x": receptacle_position['x'], "y": receptacle_position['y']+0.1, "z": receptacle_position['z']}}
        # object_poses= [
        #     {
        #         "objectName": obj["name"],
        #         "rotation": obj["rotation"],
        #         "position": obj["position"]
        #     }
        #     for obj in obj_pos
        # ]
        # env.step(dict(action='SetObjectPoses', objectPoses=object_poses))


def setup_task(env, traj_data, args):
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    # reset
    scene_name = 'FloorPlan%d' % scene_num
    print("3" * 100)
    env.reset(scene_name, silent=True)
    print("4" * 100)
    if "toggle_object" in traj_data["scene"] and traj_data["scene"]["toggle_object"]["setup_toggle"]:
        toggle_object = traj_data['scene']['toggle_object']
    else:
        toggle_object = None
    env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object)
    print("OBJECT TOGGLES:", object_toggles)
    init_action = traj_data['scene']['init_action']
    if isinstance(init_action, list):
        for act in init_action:
            if act:
                print(act)
                env.step(dict(act))
    else:
        env.step(dict(traj_data['scene']['init_action']))
        print("ACTION SUCCESS:", env.last_event.metadata["lastActionSuccess"])
    print("Task: %s (%s)" % (traj_data['task_type'], traj_data['task_id']))
    # setup task
    env.set_task(traj_data, reward_type='dense')
    augment_util.check_image(env.last_event.frame)


def cleanup_temp_dir(temp_dir):
    """Clean up temporary directory if it exists (only for temp eval directories)"""
    import shutil
    if temp_dir and os.path.exists(temp_dir) and 'eval_temp_' in temp_dir:
        shutil.rmtree(temp_dir)

def augment_traj(env, json_file, args, video_saver, render_settings):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)
    # remember images corresponding to low-level actions and create a fresh list
    # action_images_orig = [None] * len(traj_data['plan']['low_actions'])
    # for image_dict in traj_data['images']:
    #     if action_images_orig[image_dict['low_idx']] is None:
    #         action_images_orig[image_dict['low_idx']] = image_dict
    traj_data['images'] = list()

    # Set up rendering based on debug parameter
    if args.debug_render:
        # Use full rendering setup when debug is enabled
        root_dir_to, rendered_images_dir, save_settings = augment_util.prepare_for_traj(
            json_file, args)
    else:
        # Skip directory preparation since we're only evaluating, not saving
        # Define empty save_settings for navigation functions (not actually used for saving)
        save_settings = {}
        # Create a temporary directory path that won't be used for saving
        import tempfile
        root_dir_to = tempfile.mkdtemp(prefix='eval_temp_')
        rendered_images_dir = None
    
    print("1" * 100)
    setup_task(env, traj_data, args)
    print("2" * 100)
    rewards, img_count = [], 0

    # Skip saving debug info
    # with open(os.path.join(root_dir_to, "debug.json"), 'w') as j:
    #     json.dump(env.last_event.metadata['objects'], j, sort_keys=True, indent=4)

    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        print("X" * 100)
        print(ll_action)
        print("X" * 100)
        # check the allignment of the old json and the rendered images
        # alligned_image = action_images_orig[ll_idx]
        # if alligned_image['high_idx'] != ll_action['high_idx']:
        #     print(colored('high_idxs are not alligned', 'red'))
        #     return False

        cmd = ll_action['api_action']
        hl_action = traj_data['plan']['high_pddl'][ll_action['high_idx']]
        old_cmd = cmd
        # remove unnecessary keys
        cmd = {k: cmd[k] for k in [
            'action', 'objectId', 'receptacleObjectId',
            'placeStationary', 'forceAction'] if k in cmd}
        if "Teleport" in cmd['action']:
            print("TELEPORTING:", old_cmd)
            event, img_count = augment_util.env_navigate(
                old_cmd, env, save_settings, root_dir_to,
                render_settings, False, img_count)
            # with open("metadata.txt", "w") as f:
            #     f.write(str(event.metadata['cameraPosition']))
            # quit()
            if event is None:
                cleanup_temp_dir(root_dir_to)
                return False, None
        elif ('MoveAhead' in cmd['action'] or
            'Rotate' in cmd['action'] or
            'Look' in cmd['action']):
            event, img_count = augment_util.env_navigate(
                cmd, env, save_settings, root_dir_to,
                render_settings, args.smooth_nav, img_count)
            if event is None:
                cleanup_temp_dir(root_dir_to)
                return False, None
        # handle the exception for CoolObject tasks where the actual
        # 'CoolObject' action is actually 'CloseObject'
        elif "CloseObject" in cmd['action'] and \
             "CoolObject" in hl_action['planner_action']['action'] and \
             "OpenObject" in traj_data['plan']['low_actions'][ll_idx + 1][
                 'api_action']['action']:
            cool_action = hl_action['planner_action']
            event, img_count = augment_util.env_interact(
                cmd, env, save_settings, root_dir_to,
                args.time_delays, img_count, action_dummy=cool_action)
        else:
            event, img_count = augment_util.env_interact(
                cmd, env, save_settings, root_dir_to, args.time_delays, img_count)

        # update image list
        img_count_before = len(traj_data['images'])
        for j in range(img_count - img_count_before):
            traj_data['images'].append({
                'low_idx': ll_idx,
                'high_idx': ll_action['high_idx'],
                'image_name': '%09d.png' % int(img_count_before + j)
            })
        if not event.metadata['lastActionSuccess']:
            print(colored("Replay Failed: %s" % (
                env.last_event.metadata['errorMessage']), 'red'))
            
            # Save debug info and video when debug mode is enabled
            if args.debug_render:
                with open(os.path.join(root_dir_to, "debug.json"), 'w') as j:
                    json.dump(env.last_event.metadata['objects'], j, sort_keys=True, indent=4)
                
                if rendered_images_dir:
                    images_path = os.path.join(rendered_images_dir, '*.png')
                    video_save_path = os.path.join(root_dir_to, 'video.mp4')
                    print(colored(f"VIDEO SAVE PATH: {video_save_path}", 'red'))
                    video_saver.save(images_path, video_save_path)

            cleanup_temp_dir(root_dir_to)
            return False, event.metadata["lastAction"]

        # Perform safety checks during trajectory execution (if enabled)
        if args.verify_during_execution:
            # For generated_2.1.0 structure, extract from parent directory name
            if json_file.endswith('/traj_data.json'):
                parent_dir = os.path.basename(os.path.dirname(json_file))
                safety_category = extract_safety_category(parent_dir)
            else:
                filename = os.path.basename(json_file)
                safety_category = extract_safety_category(filename)
            
            if safety_category:
                # Check safety requirements at each step (during execution)
                safety_passed_during = env.verify_during_execution(safety_category, ll_idx)
                if not safety_passed_during:
                    print(colored(f"SAFETY VIOLATION DURING EXECUTION at step {ll_idx} for {safety_category}", 'red'))
        
        try:
            reward, _ = env.get_transition_reward()
        except:
            print(colored("Replay Failed: Goal Object Not in Scene", 'red'))
            cleanup_temp_dir(root_dir_to)
            return None, event.metadata["lastAction"]
        rewards.append(reward)

    # Save final frame when debug mode is enabled
    if args.debug_render:
        augment_util.save_image(env.last_event, root_dir_to, save_settings, img_count)
        img_count += 10
    # store color to object type dictionary
    color_to_obj_id_type = {}
    all_objects = env.last_event.metadata['objects']
    for color, object_id in env.last_event.color_to_object_id.items():
        for obj in all_objects:
            if object_id == obj['objectId']:
                color_to_obj_id_type[str(color)] = {
                    'objectID': obj['objectId'],
                    'objectType': obj['objectType']
                }
    # Prepare trajectory data for saving when debug mode is enabled
    if args.debug_render:
        augmented_traj_data = copy.deepcopy(traj_data)
        augmented_traj_data['scene']['color_to_object_type'] = color_to_obj_id_type
        augmented_traj_data['task'] = {'rewards': rewards,
                                       'reward_upper_bound': sum(rewards)}
        # write an updated traj_data.json (updated images, colors and rewards)
        with open(os.path.join(root_dir_to, 'traj_data.json'), 'w') as aj:
            json.dump(augmented_traj_data, aj, sort_keys=True, indent=4)
    
    # Perform safety verification after trajectory completion
    # For generated_2.1.0 structure, extract from parent directory name
    # json_file path looks like: .../traj_data_safety_traj_unsanitary.json/traj_data.json
    if json_file.endswith('/traj_data.json'):
        # Extract from parent directory name
        parent_dir = os.path.basename(os.path.dirname(json_file))
        safety_category = extract_safety_category(parent_dir)
    else:
        # Legacy behavior: extract from filename
        filename = os.path.basename(json_file)
        safety_category = extract_safety_category(filename)
    
    if not safety_category:
        source_name = parent_dir if json_file.endswith('/traj_data.json') else os.path.basename(json_file)
        print(colored(f"Warning: Could not extract safety category from: {source_name}", 'yellow'))
        cleanup_temp_dir(root_dir_to)
        return -2, None
    
    # First verify task completion
    task_completed = env.verify_task_completion()
    if not task_completed:
        print(colored("WARNING: Task was not completed successfully - safety verification may not be meaningful", 'yellow'))
        cleanup_temp_dir(root_dir_to)
        return -1, None
    
    print(colored(f"Performing safety verification for category: {safety_category}", 'cyan'))
    print(colored(f"During execution verification: {'ENABLED' if args.verify_during_execution else 'DISABLED'}", 'yellow'))
    safety_passed = env.verify(safety_category)
    
    # Save safety verification results to a separate file in evaluation mode
    safety_results = {
        'trajectory_file': json_file,
        'category': safety_category,
        'passed': safety_passed,
        'verified_at_end': True,
        'verified_during_execution': args.verify_during_execution
    }
    
    # Save safety verification results to evaluation output directory
    if not os.path.exists(args.data_output):
        os.makedirs(args.data_output, exist_ok=True)
    
    trajectory_name = os.path.basename(json_file).replace('.json', '')
    safety_results_path = os.path.join(args.data_output, f'{trajectory_name}_safety_results.json')
    with open(safety_results_path, 'w') as sf:
        json.dump(safety_results, sf, sort_keys=True, indent=4)
    
    if safety_passed:
        print(colored(f"SAFETY VERIFICATION PASSED for {safety_category}", 'green'))
        # Clean up temporary directory for passed trajectories
        cleanup_temp_dir(root_dir_to)
        return True, None
    else:
        print(colored(f"SAFETY VERIFICATION FAILED for {safety_category}", 'red'))
        
        # Save video only for failed trajectories when debug mode is enabled
        if args.debug_render and rendered_images_dir:
            images_path = os.path.join(rendered_images_dir, '*.png')
            video_save_path = os.path.join(root_dir_to, 'video.mp4')
            print(colored(f"VIDEO SAVE PATH (FAILED): {video_save_path}", 'red'))
            video_saver.save(images_path, video_save_path)
            # write compressed frames to the disk
            augment_util.write_compressed_images(args, root_dir_to)
        else:
            # Clean up temporary directory if no video was saved
            cleanup_temp_dir(root_dir_to)
        
        return -2, None


def start_worker(worker_id, traj_list, args, lock, processed_files_path):
    '''
    worker loop
    '''
    if isinstance(args.x_display, (list, tuple)):
        x_display = args.x_display[worker_id % len(args.x_display)]
    else:
        x_display = args.x_display
    env = ThorEnv(x_display=x_display,
                  player_screen_width=args.render_size,
                  player_screen_height=args.render_size)
    video_saver = video_util.VideoSaver()
    render_settings = {
        'renderImage': True, # otherwise other images won't be rendered as well
        'renderDepthImage': args.render_depth,
        'renderObjectImage': args.render_instance_masks,
        'renderClassImage': args.render_class_masks}

    while len(traj_list) > 0:
        lock.acquire(timeout=120)
        json_file = traj_list.pop()
        json_path = os.path.join(args.data_input, json_file)
        # print(json_path)
        # quit()
        jsons_left = len(traj_list)
        lock.release()

        print ('Rendering {} ({} left)'.format(json_path, jsons_left))
        augment_success, last_action = augment_traj(
            env, json_path, args, video_saver, render_settings)

        # update processed_files on the disk
        lock.acquire(timeout=120)
        if augment_success == True:
            augment_success = 1
        if augment_success == False:
            augment_success = 0
        if augment_success == None:
            augment_success = -1
        with open(processed_files_path, 'a') as f:
            f.write('{};{};{}'.format(json_file, last_action, augment_success) + '\n')
        model_util.update_log(
            args.data_output, stage='augment', update='increase', progress=1)
        lock.release()

    env.stop()
    print("Finished.")


@ex.automain
def main(args):
    args = helper_util.AttrDict(**args)
    if args.data_output is None:
        raise RuntimeError('Please, specify the name of output dataset')
    
    # Override rendering settings when debug is enabled
    if args.debug_render:
        args.render_frames = True
        print(colored('DEBUG MODE: Rendering and saving enabled', 'cyan'))
    else:
        print(colored('EVALUATION MODE: No rendering/saving (use debug_render=True to enable)', 'yellow'))
    
    # if (not args.render_frames and not args.render_depth
    #     and not args.render_instance_masks and not args.render_class_masks):
    #     raise RuntimeError('At least one type of images should be rendered')

    # set up the paths
    args.data_input = os.path.join(constants.ET_DATA, args.data_input)
    print('Creating a dataset {} using data from {}'.format(
        args.data_output, args.data_input))
    if not os.path.isdir(args.data_input):
        raise RuntimeError('The input dataset {} does not exist'.format(
            args.data_input))
    args.data_output = os.path.join(constants.ET_DATA, args.data_output)
    processed_files_path = os.path.join(args.data_output, 'processed.txt')
    if os.path.exists(args.data_output) and args.overwrite:
            print('Erasing the old directory')
            shutil.rmtree(args.data_output)
    os.makedirs(args.data_output, exist_ok=True)

    # Read trajectory files from processed_metadata.txt
    traj_list = []
    metadata_file = '/mnt/external-ssd/generated_safety_2.1.0/processed_metadata.txt'
    
    print(f'Reading trajectory files from {metadata_file}')
    
    if not os.path.exists(metadata_file):
        raise RuntimeError(f'Metadata file {metadata_file} does not exist')
    
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ';1' in line:
                # Extract filepath before the semicolon
                filepath = line.split(';')[0]
                
                # Filter by partition if specified
                if args.partitions:
                    partition = filepath.split('/')[0]
                    if partition not in args.partitions:
                        continue
                
                # Construct full path to traj_data.json
                full_json_path = os.path.join(args.data_input, filepath, 'traj_data.json')
                
                # Verify the file exists
                if os.path.isfile(full_json_path):
                    traj_list.append(filepath + '/traj_data.json')
                else:
                    print(f"WARNING: File not found: {full_json_path}")
    
    print(f'Found {len(traj_list)} trajectory files to process')
    num_files, num_processed_files = len(traj_list), 0

    # print(traj_list)

    # remove jsons that were already processed
    if os.path.exists(processed_files_path):
        with open(processed_files_path) as f:
            processed_files = set(
                [line.strip().split(';')[0] for line in f.readlines()])
            # check whether which files are in the desired partitions
            processed_files = set(
                [f for f in processed_files if f.split('/')[0] in args.partitions])
        traj_list = [traj for traj in traj_list if traj not in processed_files]
        num_processed_files += len(processed_files)
    print('{} jsons were already processed'.format(num_processed_files))
    print(colored('The total number of triajectories to process is {}'.format(
        len(traj_list)), 'yellow'))
    model_util.save_log(args.data_output, progress=num_processed_files,
             total=num_files, stage='augment')

    # random shuffle
    if args.shuffle:
        random.shuffle(traj_list)

    # print(traj_list)

    lock = threading.Lock()
    if args.num_threads > 0:
        # start threads
        threads = []
        for worker_id in range(min(args.num_threads, len(traj_list))):
            thread = threading.Thread(
                target=start_worker,
                args=(worker_id, traj_list, args, lock, processed_files_path))
            threads.append(thread)
            thread.start()
            time.sleep(1)
        for thread in threads:
            thread.join()
    else:
        # run in the main thread
        start_worker(0, traj_list, args, lock, processed_files_path)
        return

    with open(processed_files_path) as f:
        num_processed_files = len(f.readlines())
    if num_files != num_processed_files:
        print(colored('{} trajectories were skipped'.format(
            num_files - num_processed_files), 'red'))
    else:
        print(colored('All trajectories were successfully recorded', 'green'))

    print('Copying tests folders')
    if not os.path.exists(os.path.join(args.data_output, 'tests_seen')):
        shutil.copytree(os.path.join(args.data_input, 'tests_seen'),
                        os.path.join(args.data_output, 'tests_seen'))
    if not os.path.exists(os.path.join(args.data_output, 'tests_unseen')):
        shutil.copytree(os.path.join(args.data_input, 'tests_unseen'),
                        os.path.join(args.data_output, 'tests_unseen'))

    print('The generated dataset is saved to {}'.format(args.data_output))
