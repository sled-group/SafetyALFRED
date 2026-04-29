import os
import cv2
import glob
import json
import shutil
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from termcolor import colored

import gen.constants as constants


TRAJ_DATA_JSON_FILENAME = "traj_data.json"


def prepare_for_traj(json_file, args):
    # make directories
    root_dir_from = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")
    root_dir_to = root_dir_from.replace(args.data_input, args.data_output)
    # clean the directory first
    if os.path.exists(root_dir_to):
        print("ROOT DIR:", root_dir_to)
        shutil.rmtree(root_dir_to)
    rendered_images_dir = None
    save_settings = {}
    if args.render_frames:
        save_settings['frames_folder'] = 'raw_images'
        rendered_images_dir = rendered_images_dir or os.path.join(
            root_dir_to, 'raw_images')
    if args.render_depth:
        save_settings['depth_folder'] = 'depth_images'
        rendered_images_dir = rendered_images_dir or os.path.join(
            root_dir_to, 'depth_images')
    if args.render_instance_masks:
        save_settings['instance_masks_folder'] = 'instance_masks'
        rendered_images_dir = rendered_images_dir or os.path.join(
            root_dir_to, 'instance_masks')
    if args.render_class_masks:
        save_settings['class_masks_folder'] = 'class_masks'
        rendered_images_dir = rendered_images_dir or os.path.join(
            root_dir_to, 'class_masks')
    if args.save_detections:
        assert args.render_frames
        save_settings['detections_folder'] = 'detections'
    for settings_key in save_settings:
        if '_folder' in settings_key:
            os.makedirs(os.path.join(
                root_dir_to, save_settings[settings_key]), exist_ok=True)

    # copy the original json, problem_0.pddl, ResNet features and compressed masks
    shutil.copy2(json_file, os.path.join(root_dir_to, TRAJ_DATA_JSON_FILENAME))
    files_to_copy = ('problem_0.pddl', 'feat_conv.pt', 'masks.pkl')
    for file_to_copy in files_to_copy:
        if os.path.exists(os.path.join(root_dir_from, file_to_copy)):
            shutil.copy2(os.path.join(root_dir_from, file_to_copy), root_dir_to)
    return root_dir_to, rendered_images_dir, save_settings


def count_images(save_path):
    return max(len(glob.glob(save_path + '/*.jpg')),
               len(glob.glob(save_path + '/*.png')))


def execute_delays(env, action, direction):
    counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
    for i in range(counts):
        env.noop()
    return counts


def save_image(event, save_path, save_settings, im_ind):
    assert im_ind is not None
    # rgb
    if 'frames_folder' in save_settings:
        rgb_save_path = os.path.join(save_path, save_settings['frames_folder'])
        rgb_image = event.frame[:, :, ::-1]
        cv2.imwrite(rgb_save_path + '/%09d.png' % im_ind, rgb_image)

        metadata = event.metadata
        os.makedirs(os.path.join(save_path, "metadata"), exist_ok=True)
        with open(f"{save_path}/metadata/metadata_{im_ind}.txt", "w") as file:
            file.write(str(metadata))


    # depth
    if 'depth_folder' in save_settings:
        depth_save_path = os.path.join(save_path, save_settings['depth_folder'])
        depth_image = event.depth_frame
        depth_image = depth_image * (255 / 10000)
        depth_image = depth_image.astype(np.uint8)
        cv2.imwrite(depth_save_path + '/%09d.png' % im_ind, depth_image)

    # instance masks (PNG image of segmentation frame)
    if 'instance_masks_folder' in save_settings:
        instance_masks_save_path = os.path.join(
            save_path, save_settings['instance_masks_folder'])
        instance_masks_image = event.instance_segmentation_frame
        cv2.imwrite(
            instance_masks_save_path + '/%09d.png' % im_ind, instance_masks_image)

    # instance masks dictionary (boolean masks per object) and segmentation metadata
    if 'instance_masks_data_folder' in save_settings:
        instance_masks_data_path = os.path.join(
            save_path, save_settings['instance_masks_data_folder'])
        os.makedirs(instance_masks_data_path, exist_ok=True)

        # Save instance_masks as numpy arrays (dict of objectId -> boolean mask)
        masks_to_save = {}
        object_ids = []

        if hasattr(event, 'instance_masks') and event.instance_masks:
            for obj_id, mask in event.instance_masks.items():
                # Use sanitized key names for npz (replace special chars)
                safe_key = obj_id.replace('|', '_').replace('.', '_').replace('+', 'p').replace('-', 'm')
                masks_to_save[safe_key] = mask
                object_ids.append(obj_id)

            # Save all masks in a single .npz file
            np.savez_compressed(
                instance_masks_data_path + '/%09d_masks.npz' % im_ind,
                **masks_to_save
            )

        # Get color mappings for instance_segmentation_frame
        color_to_object = {}
        object_to_color = {}

        if hasattr(event, 'color_to_object_id') and event.color_to_object_id:
            # Convert tuple keys to strings for JSON
            color_to_object = {
                str(color): obj_id
                for color, obj_id in event.color_to_object_id.items()
            }

        if hasattr(event, 'object_id_to_color') and event.object_id_to_color:
            # Convert tuple values to lists for JSON
            object_to_color = {
                obj_id: list(color) if hasattr(color, '__iter__') else color
                for obj_id, color in event.object_id_to_color.items()
            }

        # Save segmentation metadata (includes mapping from safe_key back to objectId)
        segmentation_data = {
            'object_ids': object_ids,  # Original object IDs in order
            'mask_keys': list(masks_to_save.keys()),  # Sanitized keys used in npz file
            'color_to_object_id': color_to_object,
            'object_id_to_color': object_to_color,
            'instance_segmentation_frame_shape': list(event.instance_segmentation_frame.shape) if hasattr(event, 'instance_segmentation_frame') else None,
            'frame_number': im_ind
        }

        with open(instance_masks_data_path + '/%09d_metadata.json' % im_ind, 'w') as f:
            json.dump(segmentation_data, f, indent=2)

    # instance detections2D (visible objects in frame)
    if 'instance_detections_folder' in save_settings:
        instance_detections_save_path = os.path.join(
            save_path, save_settings['instance_detections_folder'])
        os.makedirs(instance_detections_save_path, exist_ok=True)

        # Extract bounding boxes from instance_detections2D
        instance_detections = {}

        # Try to get instance_detections2D directly from event
        if hasattr(event, 'instance_detections2D') and event.instance_detections2D:
            # Convert numpy arrays to lists for JSON serialization
            instance_detections = {
                str(obj_id): bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                for obj_id, bbox in event.instance_detections2D.items()
            }
        # Fallback: extract from metadata
        elif hasattr(event, 'metadata') and event.metadata:
            # Check if instance_detections2D is in metadata
            if 'instance_detections2D' in event.metadata:
                instance_detections = {
                    str(obj_id): bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
                    for obj_id, bbox in event.metadata['instance_detections2D'].items()
                }
            # Otherwise, just list visible objects without bounding boxes
            else:
                metadata_objects = event.metadata.get('objects', [])
                visible_objects = [obj['objectId'] for obj in metadata_objects if obj.get('visible', False)]
                instance_detections = {obj_id: None for obj_id in visible_objects}

        detection_data = {
            'instance_detections2D': instance_detections,
            'num_visible_objects': len(instance_detections),
            'frame_number': im_ind
        }

        with open(instance_detections_save_path + '/%09d.json' % im_ind, 'w') as f:
            json.dump(detection_data, f, indent=2)

    # class masks
    if 'class_masks_folder' in save_settings:
        class_masks_save_path = os.path.join(
            save_path, save_settings['class_masks_folder'])
        class_masks_image = event.class_segmentation_frame
        cv2.imwrite(class_masks_save_path + '/%09d.png' % im_ind, class_masks_image)


    # detection bounding boxes
    if 'detections_folder' in save_settings:
        bounding_boxes = {name: bbs
                          for name, bbs in event.class_detections2D.items()}
        bounding_boxes = {name: [[int(v) for v in bb] for bb in bbs]
                          for name, bbs in bounding_boxes.items()}
        class_values = {}
        for class_name in bounding_boxes.keys():
            class_pixels_i, class_pixels_j = np.nonzero(event.class_masks[class_name])
            class_value_rgb = list(
                event.class_segmentation_frame[class_pixels_i[0], class_pixels_j[0]])
            class_values[class_name] = [int(v) for v in class_value_rgb]
        detections_save_path = os.path.join(
            save_path, save_settings['detections_folder'])
        with open(detections_save_path + '/%09d.json' % im_ind, 'w') as f:
            json.dump({'bounding_boxes': bounding_boxes,
                       'class_values': class_values}, f)


def check_image(img):
    assert img.any(), 'frames seem to be empty'


def _write_compressed_images(images_dir, save_path):
    images_compressed = []
    for image_path in sorted(glob.glob(os.path.join(images_dir, '*.png'))):
        image = Image.open(image_path)
        image_buffer = BytesIO()
        image.save(image_buffer, format='PNG')
        images_compressed.append(image_buffer.getvalue())
    pickle.dump(images_compressed, open(save_path, 'wb'))


def write_compressed_images(args, root_dir_to):
    if args.render_depth:
        _write_compressed_images(
            os.path.join(root_dir_to, 'depth_images'),
            os.path.join(root_dir_to, 'depths.pkl'))
    if args.render_instance_masks:
        _write_compressed_images(
            os.path.join(root_dir_to, 'instance_masks'),
            os.path.join(root_dir_to, 'instances.pkl'))
    if args.render_class_masks:
        _write_compressed_images(
            os.path.join(root_dir_to, 'class_masks'),
            os.path.join(root_dir_to, 'classes.pkl'))


def env_navigate(cmd, env, save_settings, root_dir_to,
                 render_settings, smooth_nav, img_count):
    if not smooth_nav:
        event = env.step(cmd)
        save_image(event, root_dir_to, save_settings, img_count)
        return event, img_count + 1
    # smooth navigation should be executed
    if 'MoveAhead' in cmd['action']:
        action_function_name = 'smooth_move_ahead'
    elif 'Rotate' in cmd['action']:
        action_function_name = 'smooth_rotate'
    elif 'Look' in cmd['action']:
        action_function_name = 'smooth_look'
    else:
        raise NotImplementedError(
            'Action {} is not supported by navigate routine'.format(cmd['action']))
    save_image(env.last_event, root_dir_to, save_settings, img_count)
    events = getattr(env, action_function_name)(cmd, render_settings)
    if len(events) == 0:
        print(colored('env.{} returned empty events - falling back to discrete movement'.format(
            action_function_name), 'yellow'))
        # Fallback to discrete (non-smooth) movement
        event = env.step(cmd)

        # If discrete movement also fails (e.g., small object blocking), try with forceAction
        if not event.metadata['lastActionSuccess']:
            error_msg = event.metadata.get('errorMessage', '')
            print(colored(f'  Discrete movement failed: {error_msg}', 'yellow'))

            # For movement actions, try forceAction to push through small object collisions
            if cmd['action'] in ['MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight']:
                print(colored('  Retrying with forceAction to push through obstacles', 'yellow'))
                force_cmd = dict(cmd)
                force_cmd['forceAction'] = True
                event = env.step(force_cmd)
                if event.metadata['lastActionSuccess']:
                    print(colored('  ✓ forceAction succeeded - pushed through obstacle', 'green'))
                else:
                    print(colored(f'  ✗ forceAction also failed: {event.metadata.get("errorMessage", "")}', 'red'))

        save_image(event, root_dir_to, save_settings, img_count + 1)
        return event, img_count + 2
    return events[-1], img_count + len(events) + 1


def env_interact(cmd, env, save_settings, root_dir_to,
                 time_delays, img_count, action_dummy=None):
    if not time_delays:
        event = env.step(cmd)
        save_image(event, root_dir_to, save_settings, img_count)
        return event, img_count + 1
    # interaction with delays should be executed
    action_dummy = action_dummy or cmd
    save_image(env.last_event, root_dir_to, save_settings, img_count)
    img_count += execute_delays(env, action_dummy, direction=constants.BEFORE)
    # print("COMMAND:", cmd)
    event = env.step(cmd)
    img_count += execute_delays(env, action_dummy, direction=constants.MIDDLE)
    img_count += execute_delays(env, action_dummy, direction=constants.AFTER)
    return event, img_count


