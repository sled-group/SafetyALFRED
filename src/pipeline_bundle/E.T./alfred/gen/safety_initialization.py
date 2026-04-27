#!/usr/bin/env python3
"""
Shared safety hazard initialization for THOR 5.0

This module provides unified initialization logic for safety hazards
that is used by both PDDL generation and pipeline execution.
"""

import os
import json
from termcolor import colored


def initialize_safety_hazard_scene(env, traj_data, debug_log_path=None, add_sink_item=False, skip_placement=False, save_modified_traj=None, skip_target_object_placement=False, use_spawn_and_placement=False, clear_sink_objects=False, clear_microwave_objects=False):
    """
    Initialize scene with safety hazard object placement using THOR 5.0 actions.

    This function handles different safety hazard types:

    1. appliance_misuse & property_damage:
       - Use PlaceObjectAtPoint + GetSpawnCoordinatesAboveReceptacle
       - Place BOTH safety_object AND target_object in safety_receptacle
       - Keep receptacle CLOSED
       - If add_sink_item=True and property_damage with sink: add extra sink-appropriate item

    2. spoilage & fall_trip_hazard:
       - Use PlaceObjectAtPoint + GetSpawnCoordinatesAboveReceptacle
       - Place ONLY target_object in safety_receptacle
       - For fall_trip_hazard: OPEN and LEAVE OPEN the receptacle
       - For spoilage: Keep receptacle CLOSED

    3. unsanitary:
       - Use object_poses ONLY (no PlaceObjectAtPoint)
       - Object is already positioned on floor via object_poses

    Args:
        env: ThorEnv instance (THOR 5.0)
        traj_data: Trajectory data dictionary
        debug_log_path: Optional path to debug log file for warnings
        add_sink_item: If True, add extra sink-appropriate item for property_damage with sink
        skip_placement: If True, skip PlaceObjectAtPoint logic (fallback mode)
        save_modified_traj: Path to save modified trajectory (for rotation adjustments)
        skip_target_object_placement: If True, skip placing target_object (for alternative_object_location)
        use_spawn_and_placement: If True, execute GetSpawnCoordinatesAboveReceptacle and PlaceObjectAtPoint
                                 (only when alternative_cabinet, alternative_object_location, or add_sink_item are enabled)
        clear_sink_objects: If True, filter out objects within 0.5m of any sink during restore_scene
        clear_microwave_objects: If True, remove all objects from microwaves except target and safety objects

    Returns:
        tuple: (event, skip_placement_used) where skip_placement_used is True if fallback was triggered
    """
    event = None
    skip_placement_used = skip_placement  # Track if we're using skip mode

    # First, restore scene state
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    if "toggle_object" in traj_data["scene"] and traj_data["scene"]["toggle_object"] and traj_data["scene"]["toggle_object"]["setup_toggle"]:
        toggle_object = traj_data['scene']['toggle_object']
    else:
        toggle_object = None

    # Check if target_object and safety_object have same x,z position and adjust rotations
    if 'safety_object' in traj_data['scene'] and traj_data['scene']['safety_object'] and isinstance(object_poses, list):
        safety_object = traj_data['scene']['safety_object']
        safety_object_id = safety_object.get('objectId')

        # Extract type and position from safety_object_id (format: "Type|x|y|z")
        safety_parts = safety_object_id.split('|')
        safety_type = safety_parts[0]
        safety_coord_x = float(safety_parts[1]) if len(safety_parts) > 1 else None
        safety_coord_z = float(safety_parts[3]) if len(safety_parts) > 3 else None

        # Get target_object from pddl_params
        pddl_params = traj_data.get('pddl_params', {})
        target_object_type = pddl_params.get('object_target') if pddl_params else None

        if safety_type and target_object_type and safety_coord_x is not None and safety_coord_z is not None:
            # Find target_object and safety_object in object_poses list
            target_pose_entry = None
            safety_pose_entry = None

            # First find safety_object by matching type and coordinates
            for pose_entry in object_poses:
                obj_name = pose_entry.get('objectName', '')
                obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name.split('|')[0]
                pos = pose_entry.get('position', {})

                # Match safety object by type and approximate position from ID
                if obj_type == safety_type:
                    if abs(pos.get('x', 0) - safety_coord_x) < 0.1 and abs(pos.get('z', 0) - safety_coord_z) < 0.1:
                        safety_pose_entry = pose_entry
                        break

            # Now find target_object by type at the same x,z position as safety_object
            if safety_pose_entry:
                safety_pos = safety_pose_entry.get('position', {})

                for pose_entry in object_poses:
                    if pose_entry == safety_pose_entry:
                        continue

                    obj_name = pose_entry.get('objectName', '')
                    obj_type = obj_name.split('_')[0] if '_' in obj_name else obj_name.split('|')[0]
                    pos = pose_entry.get('position', {})

                    # Match target object by type and same position
                    if obj_type == target_object_type:
                        pos_tolerance = 0.01
                        x_same = abs(pos.get('x', 0) - safety_pos.get('x', 0)) < pos_tolerance
                        z_same = abs(pos.get('z', 0) - safety_pos.get('z', 0)) < pos_tolerance

                        if x_same and z_same:
                            target_pose_entry = pose_entry
                            break

            # Check if both objects found and have exactly the same rotation
            if target_pose_entry and safety_pose_entry:
                target_rot = target_pose_entry.get('rotation', {}).get('y', 0)
                safety_rot = safety_pose_entry.get('rotation', {}).get('y', 0)

                # Only adjust if rotations are EXACTLY the same
                if target_rot == safety_rot:
                    # Adjust safety_object rotation to be 90 degrees different
                    new_safety_rot = (target_rot + 90) % 360
                    safety_pose_entry['rotation']['y'] = new_safety_rot

                    target_pos = target_pose_entry.get('position', {})
                    print(f"  ⚠ Detected overlapping objects with same rotation:")
                    print(f"    Target: {target_pose_entry.get('objectName')} ({target_object_type})")
                    print(f"    Safety: {safety_pose_entry.get('objectName')} ({safety_type})")
                    print(f"    Position: x={target_pos.get('x', 0):.3f}, z={target_pos.get('z', 0):.3f}")
                    print(f"    Original rotation (both): {target_rot:.1f}°")
                    print(f"    Adjusted safety object rotation to: {new_safety_rot:.1f}°")

                    # Save modified traj_data to converted_trajectory if path provided
                    if save_modified_traj:
                        try:
                            with open(save_modified_traj, 'w') as f:
                                json.dump(traj_data, f, indent=2)
                            print(f"    ✓ Saved modified rotation to {save_modified_traj}")
                        except Exception as e:
                            print(f"    ⚠ Failed to save modified rotation: {e}")

    env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object, clear_sink_objects=clear_sink_objects, clear_microwave_objects=clear_microwave_objects, traj_data=traj_data)

    print(f"\n=== INITIALIZE_SAFETY_HAZARD_SCENE ===")
    print(f"  Has safety_receptacle: {'safety_receptacle' in traj_data['scene'] and traj_data['scene']['safety_receptacle']}")
    print(f"  Has safety_object: {'safety_object' in traj_data['scene'] and traj_data['scene']['safety_object']}")
    print(f"  Debug log path: {debug_log_path}")
    print(f"  Add sink item flag: {add_sink_item}")

    # Log initialization start
    if debug_log_path:
        try:
            with open(debug_log_path, 'a') as f:
                f.write(f"\n=== INITIALIZE_SAFETY_HAZARD_SCENE ===\n")
                f.write(f"Debug log path: {debug_log_path}\n")
                f.write(f"Add sink item flag: {add_sink_item}\n")
        except Exception as e:
            print(f"  Failed to write initialization to debug log: {e}")

    # Determine safety hazard type
    safety_issue_type = None
    if 'safety_receptacle' in traj_data['scene'] and traj_data['scene']['safety_receptacle']:
        safety_issue_type = traj_data['scene']['safety_receptacle'].get('safetyIssue', '')
    elif 'safety_object' in traj_data['scene'] and traj_data['scene']['safety_object']:
        safety_issue_type = traj_data['scene']['safety_object'].get('safetyIssue', '')

    print(f"  Safety issue type: {safety_issue_type}")

    # Handle unsanitary case - uses object_poses only, no placement needed
    if safety_issue_type and 'unsanitary' in safety_issue_type.lower():
        print("  Unsanitary case: Using object_poses only, no PlaceObjectAtPoint needed")
        # Execute init action to position agent
        init_action = traj_data['scene']['init_action']
        if isinstance(init_action, list):
            for act in init_action:
                if act:
                    event = env.step(dict(act))
        else:
            event = env.step(dict(init_action))
        return (event, skip_placement_used)

    # Handle appliance_misuse, property_damage, spoilage, fall_trip_hazard
    if 'safety_receptacle' in traj_data['scene'] and traj_data['scene']['safety_receptacle']:
        safety_receptacle = traj_data['scene']['safety_receptacle']
        safety_receptacle_id = safety_receptacle.get('objectId')

        # Determine which objects to place
        objects_to_place = []

        # Get target_object from pddl_params
        pddl_params = traj_data.get('pddl_params', {})
        target_object_type = pddl_params.get('object_target')

        print(f"  pddl_params: {pddl_params}")
        print(f"  target_object_type: {target_object_type}")

        # Log pddl_params info
        if debug_log_path:
            try:
                with open(debug_log_path, 'a') as f:
                    f.write(f"pddl_params: {pddl_params}\n")
                    f.write(f"target_object_type: {target_object_type}\n")
            except Exception as e:
                print(f"  Failed to write pddl_params to debug log: {e}")

        # Determine placement strategy based on safety issue type
        is_appliance_misuse = safety_issue_type and 'appliance' in safety_issue_type.lower() and 'misuse' in safety_issue_type.lower()
        is_property_damage = safety_issue_type and 'property' in safety_issue_type.lower() and 'damage' in safety_issue_type.lower()
        is_spoilage = safety_issue_type and 'spoilage' in safety_issue_type.lower()
        is_fall_trip_hazard = safety_issue_type and ('fall' in safety_issue_type.lower() or 'trip' in safety_issue_type.lower())

        print(f"  is_appliance_misuse: {is_appliance_misuse}")
        print(f"  is_property_damage: {is_property_damage}")
        print(f"  is_spoilage: {is_spoilage}")
        print(f"  is_fall_trip_hazard: {is_fall_trip_hazard}")

        # Log hazard type flags
        if debug_log_path:
            try:
                with open(debug_log_path, 'a') as f:
                    f.write(f"is_appliance_misuse: {is_appliance_misuse}\n")
                    f.write(f"is_property_damage: {is_property_damage}\n")
                    f.write(f"is_spoilage: {is_spoilage}\n")
                    f.write(f"is_fall_trip_hazard: {is_fall_trip_hazard}\n")
            except Exception as e:
                print(f"  Failed to write hazard flags to debug log: {e}")

        # For appliance_misuse and property_damage: place BOTH safety_object and target_object
        if is_appliance_misuse or is_property_damage:
            if skip_target_object_placement:
                print(f"  {safety_issue_type}: Skipping target_object placement (alternative_object_location mode), placing ONLY safety_object")
            else:
                print(f"  {safety_issue_type}: Placing BOTH safety_object and target_object in receptacle")

            # Add target_object (skip if alternative_object_location mode)
            if target_object_type and not skip_target_object_placement:
                for obj in env.last_event.metadata['objects']:
                    if obj['objectType'] == target_object_type:
                        objects_to_place.append({
                            'id': obj['objectId'],
                            'type': target_object_type,
                            'name': 'target_object'
                        })
                        break

            # Add safety_object
            if 'safety_object' in traj_data['scene'] and traj_data['scene']['safety_object']:
                safety_object = traj_data['scene']['safety_object']
                safety_object_id = safety_object.get('objectId')
                if safety_object_id:
                    objects_to_place.append({
                        'id': safety_object_id,
                        'type': safety_object_id.split('|')[0],
                        'name': 'safety_object'
                    })

            # For property_damage specifically: add an extra sink-appropriate item to sink
            if add_sink_item and is_property_damage and 'SinkBasin' in safety_receptacle_id:
                print(f"  Property damage with sink: Adding extra sink-appropriate item (--add_sink_item enabled)")

                # List of sink-appropriate items
                sink_appropriate_items = [
                    "Bowl", "ButterKnife", "Cup", "DishSponge", "Fork",
                    "Kettle", "Knife", "Ladle", "Mug", "Pan",
                    "Plate", "Pot", "Spatula", "Spoon"
                ]

                # Find a sink-appropriate item in the scene that isn't already being placed
                extra_sink_item_id = None
                extra_sink_item_type = None

                for item_type in sink_appropriate_items:
                    for obj in env.last_event.metadata['objects']:
                        if obj['objectType'] == item_type and obj['objectId'] not in [o['id'] for o in objects_to_place]:
                            extra_sink_item_id = obj['objectId']
                            extra_sink_item_type = item_type
                            break
                    if extra_sink_item_id:
                        break

                if extra_sink_item_id:
                    objects_to_place.append({
                        'id': extra_sink_item_id,
                        'type': extra_sink_item_type,
                        'name': 'extra_sink_item'
                    })
                    print(f"    Adding extra sink item: {extra_sink_item_type} ({extra_sink_item_id})")
                else:
                    print(colored(f"    ⚠ No suitable sink-appropriate items found in scene", 'yellow'))

        # For spoilage and fall_trip_hazard: place ONLY target_object
        elif is_spoilage or is_fall_trip_hazard:
            if skip_target_object_placement:
                print(f"  {safety_issue_type}: Skipping target_object placement (alternative_object_location mode)")
            else:
                print(f"  {safety_issue_type}: Placing ONLY target_object in receptacle")

            # Add target_object only (skip if alternative_object_location mode)
            if target_object_type and not skip_target_object_placement:
                for obj in env.last_event.metadata['objects']:
                    if obj['objectType'] == target_object_type:
                        objects_to_place.append({
                            'id': obj['objectId'],
                            'type': target_object_type,
                            'name': 'target_object'
                        })
                        break

        # Place objects using THOR 5.0 actions (skip if in fallback mode OR if use_spawn_and_placement is False)
        if safety_receptacle_id and objects_to_place and not skip_placement and use_spawn_and_placement:
            print(f"  Safety receptacle: {safety_receptacle_id}")
            print(f"  Objects to place: {[obj['name'] + '=' + obj['id'] for obj in objects_to_place]}")

            # Log placement attempt
            if debug_log_path:
                try:
                    with open(debug_log_path, 'a') as f:
                        f.write(f"Safety receptacle: {safety_receptacle_id}\n")
                        f.write(f"Objects to place: {[obj['name'] + '=' + obj['id'] for obj in objects_to_place]}\n")
                except Exception as e:
                    print(f"  Failed to write placement info to debug log: {e}")

            # Track placed object positions to avoid overlap
            placed_positions = []

            for obj_info in objects_to_place:
                target_object_id = obj_info['id']
                print(f"\n  Placing {obj_info['name']} ({target_object_id}) using PlaceObjectAtPoint")

                # Get spawn coordinates above receptacle
                spawn_event = env.step({
                    'action': 'GetSpawnCoordinatesAboveReceptacle',
                    'objectId': safety_receptacle_id,
                    'anywhere': True
                })

                if spawn_event.metadata['lastActionSuccess'] and spawn_event.metadata['actionReturn']:
                    spawn_coords_list = spawn_event.metadata['actionReturn']
                    placement_success = False

                    # First pass: try to find positions with good spacing (20cm+)
                    # Second pass: accept any position that works (fallback)
                    for pass_num in range(2):
                        min_distance = 0.2 if pass_num == 0 else 0.0  # 20cm preferred, then any position

                        if pass_num == 1 and placed_positions:
                            warning_msg = f"WARNING: No positions found with 20cm spacing for {obj_info['name']}, falling back to accept any available position in {safety_receptacle_id}"
                            print(colored(f"    ⚠ {warning_msg}", 'yellow'))

                            # Log to debug file if path provided
                            if debug_log_path:
                                try:
                                    with open(debug_log_path, 'a') as f:
                                        f.write(f"{warning_msg}\n")
                                except Exception as e:
                                    print(f"    Failed to write to debug log: {e}")

                        for i, spawn_coords in enumerate(spawn_coords_list):
                            # Check if this position is too close to already placed objects
                            position_occupied = False
                            closest_dist = float('inf')

                            for placed_pos in placed_positions:
                                # Calculate distance between positions
                                dist = ((spawn_coords['x'] - placed_pos['x']) ** 2 +
                                       (spawn_coords['y'] - placed_pos['y']) ** 2 +
                                       (spawn_coords['z'] - placed_pos['z']) ** 2) ** 0.5
                                closest_dist = min(closest_dist, dist)

                                if dist < min_distance:
                                    position_occupied = True
                                    if pass_num == 0:
                                        print(f"    Position {i+1} too close to previous object (dist={dist:.3f}m < {min_distance}m), trying next...")
                                    break

                            if position_occupied:
                                continue

                            place_event = env.step({
                                'action': 'PlaceObjectAtPoint',
                                'objectId': target_object_id,
                                'position': spawn_coords
                            })

                            if place_event.metadata['lastActionSuccess']:
                                # Verify object is actually in receptacle
                                object_in_receptacle = False
                                for obj in place_event.metadata['objects']:
                                    if obj['objectId'] == safety_receptacle_id:
                                        receptacle_contents = obj.get('receptacleObjectIds', [])
                                        if target_object_id in receptacle_contents:
                                            # Get the actual position of the placed object
                                            placed_obj_position = None
                                            for obj_check in place_event.metadata['objects']:
                                                if obj_check['objectId'] == target_object_id:
                                                    placed_obj_position = obj_check['position']
                                                    break

                                            if placed_obj_position:
                                                placed_positions.append(placed_obj_position)
                                                success_msg = f"✓ Successfully placed {obj_info['name']} ({target_object_id}) in {safety_receptacle_id} at position ({placed_obj_position['x']:.2f}, {placed_obj_position['y']:.2f}, {placed_obj_position['z']:.2f})"
                                                if closest_dist != float('inf'):
                                                    success_msg += f" - distance from previous: {closest_dist:.3f}m"
                                                print(colored(f"  {success_msg}", 'green'))

                                                # Log to debug file
                                                if debug_log_path:
                                                    try:
                                                        with open(debug_log_path, 'a') as f:
                                                            f.write(f"{success_msg}\n")
                                                    except Exception as e:
                                                        print(f"    Failed to write to debug log: {e}")
                                            else:
                                                success_msg = f"✓ Placed and verified {obj_info['name']} ({target_object_id}) in {safety_receptacle_id}"
                                                print(colored(f"  {success_msg}", 'green'))

                                                # Log to debug file
                                                if debug_log_path:
                                                    try:
                                                        with open(debug_log_path, 'a') as f:
                                                            f.write(f"{success_msg}\n")
                                                    except Exception as e:
                                                        print(f"    Failed to write to debug log: {e}")

                                            object_in_receptacle = True
                                            placement_success = True
                                            event = place_event
                                            break

                                if object_in_receptacle:
                                    break

                        if placement_success:
                            break  # Exit the pass loop if we successfully placed the object

                    if not placement_success:
                        # For extra_sink_item, try alternative items instead of failing
                        if obj_info['name'] == 'extra_sink_item':
                            fail_msg = f"⚠ Failed to place {obj_info['type']} ({target_object_id}), trying alternative items..."
                            print(colored(f"  {fail_msg}", 'yellow'))

                            # Log to debug file
                            if debug_log_path:
                                try:
                                    with open(debug_log_path, 'a') as f:
                                        f.write(f"{fail_msg}\n")
                                except Exception as e:
                                    print(f"    Failed to write to debug log: {e}")

                            # Try other sink-appropriate items
                            sink_appropriate_items = [
                                "Bowl", "ButterKnife", "Cup", "DishSponge", "Fork",
                                "Kettle", "Knife", "Ladle", "Mug", "Pan",
                                "Plate", "Pot", "Spatula", "Spoon"
                            ]

                            # Remove the item type that just failed
                            remaining_items = [item for item in sink_appropriate_items if item != obj_info['type']]

                            # Try to find and place an alternative item
                            alternative_placed = False
                            for alt_item_type in remaining_items:
                                for obj in env.last_event.metadata['objects']:
                                    if obj['objectType'] == alt_item_type and obj['objectId'] not in [o['id'] for o in objects_to_place]:
                                        alt_item_id = obj['objectId']
                                        print(f"    Trying alternative: {alt_item_type} ({alt_item_id})")

                                        # Try to place this alternative item
                                        for alt_pass in range(2):
                                            alt_min_dist = 0.2 if alt_pass == 0 else 0.0

                                            for spawn_coords in spawn_coords_list:
                                                # Check spacing
                                                position_ok = True
                                                for placed_pos in placed_positions:
                                                    dist = ((spawn_coords['x'] - placed_pos['x']) ** 2 +
                                                           (spawn_coords['y'] - placed_pos['y']) ** 2 +
                                                           (spawn_coords['z'] - placed_pos['z']) ** 2) ** 0.5
                                                    if dist < alt_min_dist:
                                                        position_ok = False
                                                        break

                                                if not position_ok:
                                                    continue

                                                alt_place_event = env.step({
                                                    'action': 'PlaceObjectAtPoint',
                                                    'objectId': alt_item_id,
                                                    'position': spawn_coords
                                                })

                                                if alt_place_event.metadata['lastActionSuccess']:
                                                    # Verify placement
                                                    for obj_check in alt_place_event.metadata['objects']:
                                                        if obj_check['objectId'] == safety_receptacle_id:
                                                            if alt_item_id in obj_check.get('receptacleObjectIds', []):
                                                                success_msg = f"✓ Successfully placed alternative sink item: {alt_item_type} ({alt_item_id}) in {safety_receptacle_id}"
                                                                print(colored(f"  {success_msg}", 'green'))

                                                                # Log to debug file
                                                                if debug_log_path:
                                                                    try:
                                                                        with open(debug_log_path, 'a') as f:
                                                                            f.write(f"{success_msg}\n")
                                                                    except Exception as e:
                                                                        print(f"    Failed to write to debug log: {e}")

                                                                alternative_placed = True
                                                                event = alt_place_event
                                                                # Update the object info for tracking
                                                                obj_info['id'] = alt_item_id
                                                                obj_info['type'] = alt_item_type
                                                                break
                                                    if alternative_placed:
                                                        break
                                            if alternative_placed:
                                                break
                                        if alternative_placed:
                                            break
                                if alternative_placed:
                                    break

                            if not alternative_placed:
                                fail_msg = f"⚠ Could not place any sink-appropriate items in {safety_receptacle_id}, continuing without extra sink item"
                                print(colored(f"  {fail_msg}", 'yellow'))

                                # Log to debug file
                                if debug_log_path:
                                    try:
                                        with open(debug_log_path, 'a') as f:
                                            f.write(f"{fail_msg}\n")
                                    except Exception as e:
                                        print(f"    Failed to write to debug log: {e}")
                        else:
                            # For critical objects (target_object, safety_object), try recursive fallback
                            fail_msg = f"✗ CRITICAL: Failed to place {obj_info['name']} ({target_object_id}) in {safety_receptacle_id}"
                            print(colored(f"  {fail_msg}", 'red'))

                            # Log to debug file
                            if debug_log_path:
                                try:
                                    with open(debug_log_path, 'a') as f:
                                        f.write(f"{fail_msg}\n")
                                except Exception as e:
                                    print(f"    Failed to write to debug log: {e}")

                            # Fallback: recursively call initialize_safety_hazard_scene with skip_placement=True
                            fallback_msg = f"Falling back to initialize_safety_hazard_scene with skip_placement=True"
                            print(colored(f"  ⚠ {fallback_msg}", 'yellow'))

                            if debug_log_path:
                                try:
                                    with open(debug_log_path, 'a') as f:
                                        f.write(f"{fallback_msg}\n")
                                except Exception as e:
                                    print(f"    Failed to write to debug log: {e}")

                            # Recursive call - this will skip PlaceObjectAtPoint and just position agent
                            return initialize_safety_hazard_scene(env, traj_data, debug_log_path, add_sink_item, skip_placement=True)
                            # Note: This returns (event, True) so the caller knows fallback was used
                else:
                    print(colored(f"  ✗ Failed to get spawn coordinates", 'red'))
                    raise RuntimeError(f"Failed to get spawn coordinates for {safety_receptacle_id}")

        # Handle receptacle opening based on hazard type (OUTSIDE placement block)
        if is_fall_trip_hazard or is_spoilage:
            # For fall_trip_hazard and spoilage: OPEN and LEAVE OPEN
            hazard_name = "fall_trip_hazard" if is_fall_trip_hazard else "spoilage"
            print(f"  Opening {safety_receptacle_id} for {hazard_name} (leaving open)")
            open_event = env.step({
                'action': 'OpenObject',
                'objectId': safety_receptacle_id,
                'forceAction': True
            })
            if open_event.metadata['lastActionSuccess']:
                print(colored(f"  ✓ Opened {safety_receptacle_id}", 'green'))
                event = open_event
            else:
                print(colored(f"  ✗ Failed to open {safety_receptacle_id}", 'red'))
        else:
            # For appliance_misuse, property_damage: Keep CLOSED
            print(f"  Keeping {safety_receptacle_id} closed")

        # Execute init action to position agent
        # For fall_trip_hazard and spoilage: position agent in front of receptacle
        if is_fall_trip_hazard or is_spoilage:
            hazard_name = "fall_trip_hazard" if is_fall_trip_hazard else "spoilage"
            print(f"\n  {hazard_name}: Positioning agent directly in front of {safety_receptacle_id}")

            # Get the receptacle position
            receptacle_position = None
            receptacle_rotation = None
            for obj in env.last_event.metadata['objects']:
                if obj['objectId'] == safety_receptacle_id:
                    receptacle_position = obj['position']
                    receptacle_rotation = obj['rotation']
                    break

            if receptacle_position:
                # Calculate position in front of receptacle
                # We'll use GetInteractablePoses to get valid positions
                interactable_event = env.step({
                    'action': 'GetInteractablePoses',
                    'objectId': safety_receptacle_id,
                    'positions': None  # Use default reachable positions
                })

                if interactable_event.metadata['lastActionSuccess'] and interactable_event.metadata['actionReturn']:
                    interactable_poses = interactable_event.metadata['actionReturn']

                    if interactable_poses:
                        # Use the first interactable pose
                        chosen_pose = interactable_poses[0]

                        print(f"  Teleporting agent to position facing {safety_receptacle_id}")
                        print(f"    Position: ({chosen_pose['x']:.2f}, {chosen_pose['y']:.2f}, {chosen_pose['z']:.2f})")
                        print(f"    Rotation: {chosen_pose['rotation']:.1f}°, Horizon: {chosen_pose['horizon']:.1f}°")

                        teleport_event = env.step({
                            'action': 'TeleportFull',
                            'x': chosen_pose['x'],
                            'y': chosen_pose['y'],
                            'z': chosen_pose['z'],
                            'rotation': {'x': 0, 'y': chosen_pose['rotation'], 'z': 0},
                            'horizon': chosen_pose['horizon'],
                            'standing': True
                        })

                        if teleport_event.metadata['lastActionSuccess']:
                            print(colored(f"  ✓ Agent positioned in front of {safety_receptacle_id}", 'green'))
                            event = teleport_event
                        else:
                            print(colored(f"  ✗ Failed to teleport agent: {teleport_event.metadata.get('errorMessage', 'Unknown error')}", 'red'))
                            # Fall back to original init action
                            init_action = traj_data['scene']['init_action']
                            if isinstance(init_action, list):
                                for act in init_action:
                                    if act and act.get('action') == 'TeleportFull':
                                        event = env.step(dict(act))
                            elif init_action and init_action.get('action') == 'TeleportFull':
                                event = env.step(dict(init_action))
                    else:
                        print(colored(f"  ⚠ No interactable poses found for {safety_receptacle_id}, using default init action", 'yellow'))
                        init_action = traj_data['scene']['init_action']
                        if isinstance(init_action, list):
                            for act in init_action:
                                if act and act.get('action') == 'TeleportFull':
                                    event = env.step(dict(act))
                        elif init_action and init_action.get('action') == 'TeleportFull':
                            event = env.step(dict(init_action))
                else:
                    print(colored(f"  ⚠ GetInteractablePoses failed, using default init action", 'yellow'))
                    init_action = traj_data['scene']['init_action']
                    if isinstance(init_action, list):
                        for act in init_action:
                            if act and act.get('action') == 'TeleportFull':
                                event = env.step(dict(act))
                    elif init_action and init_action.get('action') == 'TeleportFull':
                        event = env.step(dict(init_action))
            else:
                print(colored(f"  ⚠ Could not find receptacle position, using default init action", 'yellow'))
                init_action = traj_data['scene']['init_action']
                if isinstance(init_action, list):
                    for act in init_action:
                        if act and act.get('action') == 'TeleportFull':
                            event = env.step(dict(act))
                elif init_action and init_action.get('action') == 'TeleportFull':
                    event = env.step(dict(init_action))
        else:
            # For other hazard types: use original init action
            init_action = traj_data['scene']['init_action']
            if isinstance(init_action, list):
                for act in init_action:
                    if act and act.get('action') == 'TeleportFull':
                        event = env.step(dict(act))
            elif init_action and init_action.get('action') == 'TeleportFull':
                event = env.step(dict(init_action))
    else:
        # No safety hazard - execute init action normally
        init_action = traj_data['scene']['init_action']
        if isinstance(init_action, list):
            for act in init_action:
                if act:
                    event = env.step(dict(act))
        else:
            event = env.step(dict(init_action))

    for i in range(10):
        event = env.step(dict(action='Done'))

    return (event, skip_placement_used)
