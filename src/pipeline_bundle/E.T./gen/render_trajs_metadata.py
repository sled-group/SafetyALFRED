import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import sys
import json
import numpy as np
from multiprocessing import Process, Manager, Lock, Queue
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
ex = Experiment('render_trajs_metadata', ingredients=[args_ingredient])


@args_ingredient.config
def cfg_args():
    # dataset folder to dump frames to
    data_output = 'test_2.1.0'
    # dataset folder to load jsons from
    data_input = 'json_2.1.0'
    # smooth naviagation (like the original data)
    smooth_nav = True
    # time delays (like the original data)
    time_delays = True
    # whether to shuffle the order of augmenting
    shuffle = False
    # number of threads to start in parallel
    num_threads = 1
    # frame size to render
    render_size = 300
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
    # partitions to render data for
    partitions = ('train', 'valid_seen', 'valid_unseen')
    # whether to overwrite data folder if it already exists
    overwrite = False


def setup_task(env, traj_data, args):
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    
    # reset
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name, silent=True)
    
    # handle toggle_object if present (new in json_2.1.0)
    toggle_object = None
    if "toggle_object" in traj_data["scene"]:
        if traj_data["scene"]["toggle_object"].get("setup_toggle", False):
            toggle_object = traj_data['scene']['toggle_object']
    
    env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object)
    
    # handle init_action - can be list or single action in json_2.1.0
    init_action = traj_data['scene']['init_action']
    if isinstance(init_action, list):
        for act in init_action:
            if act:
                env.step(dict(act))
    else:
        env.step(dict(init_action))
    
    print("Task: %s (%s)" % (traj_data['task_type'], traj_data['task_id']))
    # setup task
    env.set_task(traj_data, reward_type='dense')
    augment_util.check_image(env.last_event.frame)


def augment_traj(env, json_file, args, video_saver, render_settings):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)
    
    # remember images corresponding to low-level actions and create a fresh list
    action_images_orig = [None] * len(traj_data['plan']['low_actions'])
    for image_dict in traj_data['images']:
        if action_images_orig[image_dict['low_idx']] is None:
            action_images_orig[image_dict['low_idx']] = image_dict
    traj_data['images'] = list()

    root_dir_to, rendered_images_dir, save_settings = augment_util.prepare_for_traj(
        json_file, args)
    setup_task(env, traj_data, args)
    rewards, img_count = [], 0

    # save initial metadata for debugging
    with open(os.path.join(root_dir_to, "debug.json"), 'w') as j:
        json.dump(env.last_event.metadata['objects'], j, sort_keys=True, indent=4)

    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        # check the allignment of the old json and the rendered images
        # alligned_image = action_images_orig[ll_idx]
        # if alligned_image['high_idx'] != ll_action['high_idx']:
        #     print(colored('high_idxs are not alligned', 'red'))
        #     return False

        # next cmd under the current hl_action
        cmd = ll_action['api_action']
        hl_action = traj_data['plan']['high_pddl'][ll_action['high_idx']]
        old_cmd = copy.deepcopy(cmd)
        
        # remove unnecessary keys
        cmd = {k: cmd[k] for k in [
            'action', 'objectId', 'receptacleObjectId',
            'placeStationary', 'forceAction'] if k in cmd}
        
        # handle teleport action (specific to json_2.1.0)
        if "Teleport" in cmd['action']:
            event, img_count = augment_util.env_navigate(
                old_cmd, env, save_settings, root_dir_to,
                render_settings, False, img_count)
            if event is None:
                return False
        elif ('MoveAhead' in cmd['action'] or
            'Rotate' in cmd['action'] or
            'Look' in cmd['action']):
            event, img_count = augment_util.env_navigate(
                cmd, env, save_settings, root_dir_to,
                render_settings, args.smooth_nav, img_count)
            if event is None:
                return False
        # handle the exception for CoolObject tasks where the actual
        # 'CoolObject' action is actually 'CloseObject'
        elif "CloseObject" in cmd['action'] and \
             "CoolObject" in hl_action['planner_action']['action'] and \
             ll_idx + 1 < len(traj_data['plan']['low_actions']) and \
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
            
            # save debug metadata on failure
            with open(os.path.join(root_dir_to, "debug_failure.json"), 'w') as j:
                json.dump(env.last_event.metadata['objects'], j, sort_keys=True, indent=4)
            
            # save video even on failure for debugging
            images_path = os.path.join(rendered_images_dir, '*.png')
            video_save_path = os.path.join(root_dir_to, 'video.mp4')
            video_saver.save(images_path, video_save_path)
            
            return False

        try:
            # reward, _ = env.get_transition_reward()
            reward = 0
        except Exception as e:
            print(colored(f"Reward calculation failed: {e}", 'red'))
            reward = 0
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
    video_saver.save(images_path, video_save_path)
    
    # write compressed frames to the disk
    augment_util.write_compressed_images(args, root_dir_to)
    return True


def start_worker(worker_id, work_queue, args, processed_files_path):
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

    while True:
        try:
            json_file = work_queue.get(timeout=30)  # 30 second timeout
            if json_file is None:  # Poison pill to signal shutdown
                break
            json_path = os.path.join(args.data_input, json_file)
        except:
            # Queue is empty, worker can exit
            break

        print ('Worker {}: Rendering {}'.format(worker_id, json_path))
        augment_success = augment_traj(
            env, json_path, args, video_saver, render_settings)

        # update processed_files on the disk
        processed_alfred_path = os.path.join(args.data_output, 'processed_metadata_alfred.txt')
        with open(processed_alfred_path, 'a') as f:
            f.write('{};{}'.format(json_file, int(augment_success)) + '\n')
        
        model_util.update_log(
            args.data_output, stage='augment', update='increase', progress=1)
        
        print('Worker {}: Completed {}'.format(worker_id, json_file))

    env.stop()
    print("Finished.")


@ex.automain
def main(args):
    args = helper_util.AttrDict(**args)
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
    processed_files_path = os.path.join(args.data_output, 'processed_metadata_alfred.txt')
    if os.path.exists(args.data_output) and args.overwrite:
            print('Erasing the old directory')
            shutil.rmtree(args.data_output)
    os.makedirs(args.data_output, exist_ok=True)

    # make a list of all the traj_data json files from json_2.1.0 (floor plans 1-30)
    traj_list = []
    print('Indexing trajectories with floor plans 1-30 in {}'.format(args.partitions))
    for partition in args.partitions:
        for dir_name in sorted(
                glob.glob(os.path.join(args.data_input, partition, '*/*'))):
            if 'trial_' in os.path.basename(dir_name):
                json_path = os.path.join(dir_name, 'traj_data.json')
                if not os.path.isfile(json_path):
                    continue
                
                # Check floor plan number in traj_data.json
                try:
                    with open(json_path, 'r') as f:
                        traj_data = json.load(f)
                    scene_num = traj_data['scene']['scene_num']
                    if 1 <= scene_num <= 30:
                        traj_list.append('/'.join(json_path.split('/')[-4:]))
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading {json_path}: {e}")
                    continue
    num_files, num_processed_files = len(traj_list), 0

    # remove jsons that were already processed (check both processed files)
    processed_alfred_path = os.path.join(args.data_output, 'processed_metadata_alfred.txt')
    processed_files = set()
    
    if os.path.exists(processed_files_path):
        with open(processed_files_path) as f:
            processed_files.update(
                [line.strip().split(';')[0] for line in f.readlines()])
    
    if os.path.exists(processed_alfred_path):
        with open(processed_alfred_path) as f:
            processed_files.update(
                [line.strip().split(';')[0] for line in f.readlines()])
    
    # check which files are in the desired partitions
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

    # Use a proper work queue
    work_queue = Queue()
    
    # Add all trajectories to the queue
    for traj in traj_list:
        work_queue.put(traj)
    
    if args.num_threads > 0:
        # start processes
        processes = []
        num_workers = min(args.num_threads, len(traj_list))
        
        for worker_id in range(num_workers):
            process = Process(
                target=start_worker,
                args=(worker_id, work_queue, args, processed_files_path))
            processes.append(process)
            process.start()
        
        # Add poison pills to signal workers to stop
        for _ in range(num_workers):
            work_queue.put(None)
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
            
        print("All workers finished")
    else:
        # run in the main process
        start_worker(0, work_queue, args, processed_files_path)

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