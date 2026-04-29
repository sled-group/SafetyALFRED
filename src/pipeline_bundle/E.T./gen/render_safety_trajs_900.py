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


args_ingredient = Ingredient('args')
ex = Experiment('render_trajs_900', ingredients=[args_ingredient])


@args_ingredient.config
def cfg_args():
    # dataset folder to dump frames to
    data_output = 'generated_safety_2.1.0_900x900'
    # dataset folder to load jsons from
    data_input = 'json_feat_2.1.0'
    # metadata file containing list of trajectories to rerender
    metadata_file = '/mnt/external-ssd/generated_safety_2.1.0/processed_metadata.txt'
    # smooth naviagation (like the original data)
    smooth_nav = False
    # time delays (like the original data)
    time_delays = False
    # whether to shuffle the order of augmenting
    shuffle = False
    # number of threads to start in parallel
    num_threads = 16
    # frame size to render (900x900 for rerendering)
    render_size = 900
    # X server number
    x_display = '7'
    # render and save RGB images
    render_frames = True
    # render and save depth images
    render_depth = False
    # render and save class segmentation masks
    render_class_masks = False
    # render and save instance segmentation masks
    render_instance_masks = False
    # save object bounding boxes
    save_detections = True
    # whether to overwrite data folder if it already exists
    overwrite = False
    # test mode - render a single trajectory
    test_mode = False
    # test trajectory path (for test mode)
    test_traj = "/mnt/external-ssd/accepted_videos/property_damage_middle/pick_clean_then_place_in_recep/trial_T20190909_033139_582791/converted_trajectory/traj_data.json"
    # test_traj = "/mnt/external-ssd/generated_safety_2.1.0/train/pick_heat_then_place_in_recep-Egg-None-Fridge-20/trial_T20190907_224507_776787/traj_data_safety_traj_spoilage.json/traj_data.json"
    # use teleport for navigation actions (instead of MoveAhead/Rotate)
    use_teleport = False

#ACTIONS
def decode_location(location_str):
    """
    Decode location string like 'loc|1|-4|1|30' to position and rotation.
    Format: loc|x_steps|z_steps|rotation_idx|horizon
    Step size = 0.25, Rotations = [0, 90, 180, 270]
    """
    if not location_str or location_str == 'loc|0|0|0|0':
        return None

    parts = location_str.split('|')
    if len(parts) != 5 or parts[0] != 'loc':
        return None

    try:
        x_steps = int(parts[1])
        z_steps = int(parts[2])
        rotation_idx = int(parts[3])
        horizon = int(parts[4])

        # Convert to actual position
        x = x_steps * 0.25
        z = z_steps * 0.25
        rotation = [0, 90, 180, 270][rotation_idx]

        return {
            'x': x,
            'z': z,
            'rotation': rotation,
            'horizon': horizon
        }
    except (ValueError, IndexError):
        return None


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
    # print(receptacle)
    #Must do this because using PlaceObjectAtPoint only works if object is not pickupable
    objects = env.last_event.metadata["objects"]
    if True:
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)


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

    root_dir_to, rendered_images_dir, save_settings = augment_util.prepare_for_traj(
        json_file, args)
    print("1" * 100)
    setup_task(env, traj_data, args)
    print("2" * 100)
    rewards, img_count = [], 0

    with open(os.path.join(root_dir_to, "debug.json"), 'w') as j:
        json.dump(env.last_event.metadata['objects'], j, sort_keys=True, indent=4)

    # Track which high_idx has been teleported
    teleported_goto_indices = set()

    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        # print("X" * 100)
        # print(ll_action)
        # print("X" * 100)

        cmd = ll_action['api_action']
        hl_action = traj_data['plan']['high_pddl'][ll_action['high_idx']]
        old_cmd = cmd
        current_high_idx = ll_action['high_idx']

        # remove unnecessary keys
        cmd = {k: cmd[k] for k in [
            'action', 'objectId', 'receptacleObjectId',
            'placeStationary', 'forceAction'] if k in cmd}

        # Check if this is a navigation action
        is_nav_action = ('MoveAhead' in cmd['action'] or
                        'Rotate' in cmd['action'] or
                        'Look' in cmd['action'])

        # If this is a navigation action for a GotoLocation that we've already teleported to, skip it
        if args.use_teleport and is_nav_action and current_high_idx in teleported_goto_indices:
            print(f"Skipping nav action {ll_idx} (already teleported for high_idx={current_high_idx})")
            continue

        # If use_teleport is enabled and this is the first nav action of a GotoLocation
        if args.use_teleport and is_nav_action and current_high_idx not in teleported_goto_indices:
            current_hl_action = traj_data['plan']['high_pddl'][current_high_idx]

            # Check if the high-level action is GotoLocation
            # Some trajectories have action in discrete_action, others only have planner_action with location
            discrete_action = current_hl_action.get('discrete_action', {})
            action_name = discrete_action.get('action', '')
            planner_action = current_hl_action.get('planner_action', {})

            # Check if it's a GotoLocation by action name OR by having a location field
            is_goto = (action_name == 'GotoLocation' or
                      (not action_name and 'location' in planner_action))

            if is_goto:
                # Get location from high-level action
                location_str = current_hl_action['planner_action'].get('location')
                decoded_loc = decode_location(location_str)

                if decoded_loc:
                    print(f"TELEPORTING to GotoLocation target: {location_str}")
                    # Get current y position (we only modify x, z, rotation, horizon)
                    current_y = env.last_event.metadata['agent']['position']['y']

                    # Create teleport command
                    teleport_cmd = {
                        'action': 'Teleport',
                        'x': decoded_loc['x'],
                        'y': current_y,
                        'z': decoded_loc['z'],
                        'rotateOnTeleport': True,
                        'rotation': decoded_loc['rotation'],
                        'horizon': decoded_loc['horizon'],
                        'standing': True
                    }

                    event, img_count = augment_util.env_navigate(
                        teleport_cmd, env, save_settings, root_dir_to,
                        render_settings, False, img_count)

                    if event is None:
                        return False, None

                    # Mark this GotoLocation as teleported
                    teleported_goto_indices.add(current_high_idx)
                    continue

        if "Teleport" in cmd['action']:
            print("TELEPORTING:", old_cmd)
            event, img_count = augment_util.env_navigate(
                old_cmd, env, save_settings, root_dir_to,
                render_settings, False, img_count)
            if event is None:
                return False, None
        elif is_nav_action:
            event, img_count = augment_util.env_navigate(
                cmd, env, save_settings, root_dir_to,
                render_settings, args.smooth_nav, img_count)
            if event is None:
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
            with open(os.path.join(root_dir_to, "debug.json"), 'w') as j:
                json.dump(env.last_event.metadata['objects'], j, sort_keys=True, indent=4)

            images_path = os.path.join(rendered_images_dir, '*.png')
            video_save_path = os.path.join(root_dir_to, 'video.mp4')
            print(colored(f"VIDEO SAVE PATH: {video_save_path}", 'red'))
            video_saver.save(images_path, video_save_path)

            return False, event.metadata["lastAction"]

        try:
            reward, _ = env.get_transition_reward()
        except:
            print(colored("Replay Failed: Goal Object Not in Scene", 'red'))
            return None, event.metadata["lastAction"]
        rewards.append(reward)

    # save 1 frame in the end and increase the counter by 10
    # (to be alligned with the train data)
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
    augmented_traj_data = copy.deepcopy(traj_data)
    augmented_traj_data['scene']['color_to_object_type'] = color_to_obj_id_type
    augmented_traj_data['task'] = {'rewards': rewards,
                                   'reward_upper_bound': sum(rewards)}
    # write an updated traj_data.json (updated images, colors and rewards)
    with open(os.path.join(root_dir_to, 'traj_data.json'), 'w') as aj:
        json.dump(augmented_traj_data, aj, sort_keys=True, indent=4)

    # save video
    images_path = os.path.join(rendered_images_dir, '*.png')
    video_save_path = os.path.join(root_dir_to, 'video.mp4')
    print(colored(f"VIDEO SAVE PATH: {video_save_path}", 'green'))
    video_saver.save(images_path, video_save_path)
    # write compressed frames to the disk
    augment_util.write_compressed_images(args, root_dir_to)
    return True, None


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
        # The path from metadata is a directory - append traj_data.json to get the actual file
        json_path = os.path.join(args.data_input, json_file, 'traj_data.json')
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

    # Test mode - render a single trajectory
    if args.test_mode:
        if args.test_traj is None:
            raise RuntimeError('Please specify test_traj path for test mode')

        print(f'Test mode: Rendering single trajectory {args.test_traj}')

        # Set up environment
        env = ThorEnv(x_display=args.x_display,
                      player_screen_width=args.render_size,
                      player_screen_height=args.render_size)
        video_saver = video_util.VideoSaver()
        render_settings = {
            'renderImage': True,
            'renderDepthImage': args.render_depth,
            'renderObjectImage': args.render_instance_masks,
            'renderClassImage': args.render_class_masks}

        # Render the trajectory
        augment_success, last_action = augment_traj(
            env, args.test_traj, args, video_saver, render_settings)

        env.stop()

        if augment_success:
            print(colored('Test trajectory rendered successfully!', 'green'))
        else:
            print(colored(f'Test trajectory failed: {last_action}', 'red'))

        return

    # Normal batch mode
    if args.data_output is None:
        raise RuntimeError('Please, specify the name of output dataset')
    if (not args.render_frames and not args.render_depth
        and not args.render_instance_masks and not args.render_class_masks):
        raise RuntimeError('At least one type of images should be rendered')

    # set up the paths
    args.data_input = os.path.join(constants.ET_DATA, args.data_input)
    print('Creating a dataset {} using data from {}'.format(
        args.data_output, args.data_input))
    if not os.path.isdir(args.data_input):
        raise RuntimeError('The input dataset {} does not exist'.format(
            args.data_input))
    args.data_output = os.path.join(constants.ET_DATA, args.data_output)
    processed_files_path = os.path.join(args.data_output, 'processed_900x900.txt')
    if os.path.exists(args.data_output) and args.overwrite:
            print('Erasing the old directory')
            shutil.rmtree(args.data_output)
    os.makedirs(args.data_output, exist_ok=True)

    # read the metadata file to get list of trajectories to rerender
    traj_list = []
    print('Reading trajectories from {}'.format(args.metadata_file))
    if not os.path.exists(args.metadata_file):
        raise RuntimeError('The metadata file {} does not exist'.format(
            args.metadata_file))

    with open(args.metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip error lines
            if line.startswith('Error processing'):
                continue
            # Format: path/to/traj_data.json;success_flag
            parts = line.split(';')
            if len(parts) >= 1:
                traj_path = parts[0]
                # Only add valid paths (should start with train/, valid_seen/, or valid_unseen/)
                if traj_path.startswith(('train/', 'valid_seen/', 'valid_unseen/')):
                    traj_list.append(traj_path)

    num_files = len(traj_list)
    print(colored('Found {} trajectories to rerender at 900x900'.format(
        num_files), 'yellow'))

    # remove jsons that were already processed
    num_processed_files = 0
    if os.path.exists(processed_files_path):
        with open(processed_files_path) as f:
            processed_files = set(
                [line.strip().split(';')[0] for line in f.readlines()])
        traj_list = [traj for traj in traj_list if traj not in processed_files]
        num_processed_files += len(processed_files)
    print('{} jsons were already processed'.format(num_processed_files))
    print(colored('The total number of trajectories to process is {}'.format(
        len(traj_list)), 'yellow'))
    model_util.save_log(args.data_output, progress=num_processed_files,
             total=num_files, stage='augment')

    # random shuffle
    if args.shuffle:
        random.shuffle(traj_list)

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
        tests_seen_src = os.path.join(args.data_input, 'tests_seen')
        if os.path.exists(tests_seen_src):
            shutil.copytree(tests_seen_src,
                            os.path.join(args.data_output, 'tests_seen'))
    if not os.path.exists(os.path.join(args.data_output, 'tests_unseen')):
        tests_unseen_src = os.path.join(args.data_input, 'tests_unseen')
        if os.path.exists(tests_unseen_src):
            shutil.copytree(tests_unseen_src,
                            os.path.join(args.data_output, 'tests_unseen'))

    print('The generated dataset is saved to {}'.format(args.data_output))
