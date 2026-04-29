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
ex = Experiment('render_trajs', ingredients=[args_ingredient])


@args_ingredient.config
def cfg_args():
    # dataset folder to dump frames to
    data_output = 'evaluate_safety_2.1.0'
    # dataset folder to load jsons from
    data_input = 'json_feat_2.1.0'
    # smooth naviagation (like the original data)
    smooth_nav = False
    # time delays (like the original data)
    time_delays = False
    # whether to shuffle the order of augmenting
    shuffle = False
    # number of threads to start in parallel
    num_threads = 0
    # frame size to render
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
    save_detections = False
    # partitions to render data for
    partitions = ('valid_unseen', 'valid_seen', 'train')
    # whether to overwrite data folder if it already exists
    overwrite = False

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

    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        print("X" * 100)
        print(ll_action)
        print("X" * 100)
        # check the allignment of the old json and the rendered images
        # alligned_image = action_images_orig[ll_idx]
        # if alligned_image['high_idx'] != ll_action['high_idx']:
        #     print(colored('high_idxs are not alligned', 'red'))
        #     return False

        # next cmd under the current hl_action
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

            return False

        filename = os.path.basename(json_file)

        filename_without_ext = os.path.splitext(filename)[0]
        safety_category = filename_without_ext.split('_')[-1]
        env.verify(safety_category)

        try:
            # reward, _ = env.get_transition_reward()
            reward = 0
            pass
        except:
            print(colored("Replay Failed: Goal Object Not in Scene", 'red'))
            return None
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
    return True


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
        # Build the full path first
        full_path = os.path.join(args.data_input, json_file)
        # Check if it's actually a directory (generated_safety_2.1.0 has dirs ending in .json)
        if os.path.isdir(full_path):
            # For directory paths, append traj_data.json
            json_path = os.path.join(full_path, 'traj_data.json')
        else:
            # Direct file path (json_feat_2.1.0 style)
            json_path = full_path
        jsons_left = len(traj_list)
        lock.release()

        print ('Rendering {} ({} left)'.format(json_path, jsons_left))
        augment_success = augment_traj(
            env, json_path, args, video_saver, render_settings)

        # # update processed_files on the disk
        # lock.acquire(timeout=120)
        # if augment_success == True:
        #     augment_success = 1
        # if augment_success == False:
        #     augment_success = 0
        # if augment_success == None:
        #     augment_success = -1
        # with open(processed_files_path, 'a') as f:
        #     f.write('{};{}'.format(json_file, augment_success) + '\n')
        # model_util.update_log(
        #     args.data_output, stage='augment', update='increase', progress=1)
        # lock.release()

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
    # If data_input is an absolute path, use it directly; otherwise join with ET_DATA
    if os.path.isabs(args.data_input):
        args.data_input = args.data_input
    else:
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

    # make a list of all the traj_data json files
    traj_list = []
    print('Indexing images in {}'.format(args.partitions))
    for partition in args.partitions:
        for dir_name in sorted(glob.glob(os.path.join(args.data_input, partition, '*/*'))):
                # trial_T20190909_085448_256298 bread fire hazard
                # trial_T20190907_204345_813064 tomato fire hazard and unsanitary
                # trial_T20190907_143348_068782 bread property damage
            for file_name in os.listdir(dir_name):
                if 'trial_T20190907_155134_457504' in os.path.basename(dir_name):
                # if  "pick_two_obj_and_place" not in dir_name and "Sliced" not in dir_name:
                    # print("FILE_NAME:", file_name)
                    if "traj_data_safety_traj_fire_hazard_StoveKnob|+00.67|+00.90|-01.24_3.json" in file_name:
                        json_path = os.path.join(dir_name, file_name)
                        print(json_path)
                        # quit()
                        # For generated_safety_2.1.0, these are directories, not files
                        if not os.path.isdir(json_path):
                            continue
                        traj_list.append('/'.join(json_path.split('/')[-4:]))
    num_files, num_processed_files = len(traj_list), 0

    # print(traj_list)

    # # remove jsons that were already processed
    # if os.path.exists(processed_files_path):
    #     with open(processed_files_path) as f:
    #         processed_files = set(
    #             [line.strip().split(';')[0] for line in f.readlines()])
    #         # check whether which files are in the desired partitions
    #         processed_files = set(
    #             [f for f in processed_files if f.split('/')[0] in args.partitions])
    #     traj_list = [traj for traj in traj_list if traj not in processed_files]
    #     num_processed_files += len(processed_files)
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
