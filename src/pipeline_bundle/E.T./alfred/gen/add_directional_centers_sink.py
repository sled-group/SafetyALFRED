#!/usr/bin/env python3
"""
Add front_center, right_center, left_center, and back_center coordinates to sink_spawn_coordinates.json

Front/back are determined relative to the closest reachable position to the sink.
The spawn coordinates are divided into thirds both horizontally (left/center/right) and
depth-wise (front/center/back) to determine directional centers.
"""
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import sys
import json
import glob
import math

# Clean up any existing AI2-THOR pipes
for pipe in glob.glob('/tmp/ai2thor_*'):
    try:
        os.remove(pipe)
    except:
        pass

sys.path.insert(0, '/home/josue/Desktop/Research/SLED/MSS/E.T.')

from alfred.env.thor_env_thor5 import ThorEnv


def distance_3d(p1, p2):
    """Calculate 3D Euclidean distance between two points."""
    return math.sqrt((p1['x'] - p2['x'])**2 +
                     (p1['y'] - p2['y'])**2 +
                     (p1['z'] - p2['z'])**2)


def distance_2d(p1, p2):
    """Calculate 2D Euclidean distance (x, z only)."""
    return math.sqrt((p1['x'] - p2['x'])**2 +
                     (p1['z'] - p2['z'])**2)


def find_center_coordinate(coordinates):
    """Find the most central coordinate in a list."""
    if len(coordinates) == 1:
        return coordinates[0]

    # Calculate centroid
    centroid = {
        'x': sum(c['x'] for c in coordinates) / len(coordinates),
        'y': sum(c['y'] for c in coordinates) / len(coordinates),
        'z': sum(c['z'] for c in coordinates) / len(coordinates)
    }

    # Find coordinate closest to centroid
    min_dist = float('inf')
    center_coord = coordinates[0]

    for coord in coordinates:
        dist = distance_3d(coord, centroid)
        if dist < min_dist:
            min_dist = dist
            center_coord = coord

    return center_coord


def get_directional_centers(spawn_coordinates, sink_position, reachable_positions):
    """
    Get front_center, right_center, left_center, and back_center from spawn coordinates.

    Front/back are relative to the closest reachable position to the sink.
    Spawn coordinates are divided into thirds horizontally and depth-wise.

    Args:
        spawn_coordinates: List of spawn coordinate dicts
        sink_position: Position dict of the sink
        reachable_positions: List of reachable agent positions

    Returns:
        dict with keys: front_center, right_center, left_center, back_center,
                       front_center_idx, right_center_idx, left_center_idx, back_center_idx
    """
    if not spawn_coordinates or not reachable_positions:
        return {}

    # Find closest reachable position to sink
    min_dist = float('inf')
    closest_reachable = None
    for reach_pos in reachable_positions:
        dist = distance_2d(sink_position, reach_pos)
        if dist < min_dist:
            min_dist = dist
            closest_reachable = reach_pos

    if not closest_reachable:
        return {}

    # Calculate distance from each spawn coord to closest reachable position
    spawn_with_distance = []
    for i, coord in enumerate(spawn_coordinates):
        dist = distance_2d(coord, closest_reachable)
        spawn_with_distance.append((i, coord, dist))

    # Sort by distance to find front (closest) and back (furthest)
    spawn_with_distance.sort(key=lambda x: x[2])

    # Divide into thirds by distance (front/center/back)
    num_coords = len(spawn_with_distance)
    third = num_coords // 3

    front_third = spawn_with_distance[:third] if third > 0 else spawn_with_distance[:1]
    back_third = spawn_with_distance[-third:] if third > 0 else spawn_with_distance[-1:]

    result = {}

    # Get all spawn coordinates for left/right calculation
    all_x_values = [coord['x'] for coord in spawn_coordinates]
    min_x = min(all_x_values)
    max_x = max(all_x_values)
    x_range = max_x - min_x

    # Divide horizontally into thirds (left/center/right)
    if x_range > 0:
        left_threshold = min_x + x_range / 3
        right_threshold = min_x + 2 * x_range / 3

        left_coords = [(i, coord) for i, coord in enumerate(spawn_coordinates) if coord['x'] <= left_threshold]
        right_coords = [(i, coord) for i, coord in enumerate(spawn_coordinates) if coord['x'] >= right_threshold]
    else:
        # All x values are the same
        left_coords = [(i, coord) for i, coord in enumerate(spawn_coordinates)]
        right_coords = [(i, coord) for i, coord in enumerate(spawn_coordinates)]

    # Front center: center of the front third
    front_coords = [coord for _, coord, _ in front_third]
    if front_coords:
        front_center = find_center_coordinate(front_coords)
        # Find index
        for i, coord in enumerate(spawn_coordinates):
            if distance_3d(coord, front_center) < 0.01:
                result['front_center'] = front_center
                result['front_center_idx'] = i
                break

    # Back center: center of the back third
    back_coords = [coord for _, coord, _ in back_third]
    if back_coords:
        back_center = find_center_coordinate(back_coords)
        # Find index
        for i, coord in enumerate(spawn_coordinates):
            if distance_3d(coord, back_center) < 0.01:
                result['back_center'] = back_center
                result['back_center_idx'] = i
                break

    # Left center: center of the left third
    if left_coords:
        left_coords_list = [coord for _, coord in left_coords]
        left_center = find_center_coordinate(left_coords_list)
        # Find index
        for i, coord in enumerate(spawn_coordinates):
            if distance_3d(coord, left_center) < 0.01:
                result['left_center'] = left_center
                result['left_center_idx'] = i
                break

    # Right center: center of the right third
    if right_coords:
        right_coords_list = [coord for _, coord in right_coords]
        right_center = find_center_coordinate(right_coords_list)
        # Find index
        for i, coord in enumerate(spawn_coordinates):
            if distance_3d(coord, right_center) < 0.01:
                result['right_center'] = right_center
                result['right_center_idx'] = i
                break

    return result


def add_directional_centers_to_sinks():
    """
    Add directional centers to sink_spawn_coordinates.json
    """
    # Load existing sink data
    sink_file = '/home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/sink_spawn_coordinates.json'

    with open(sink_file, 'r') as f:
        sink_data = json.load(f)

    env = ThorEnv(x_display='7')

    total_sinks = 0
    updated_sinks = 0

    # Process each floor plan
    for scene_num in range(1, 31):
        scene_name = f'FloorPlan{scene_num}'

        if scene_name not in sink_data:
            continue

        print(f"Processing {scene_name}...")

        try:
            # Reset to the scene
            env.reset(scene_name, silent=True)

            # Get reachable positions
            reachable_event = env.step({'action': 'GetReachablePositions'})
            reachable_positions = []
            if reachable_event.metadata['lastActionSuccess']:
                reachable_positions = reachable_event.metadata['actionReturn']

            # Process each sink in this floorplan
            for sink_id, sink_info in sink_data[scene_name].items():
                total_sinks += 1

                # Check if already has directional centers
                has_all = ('front_center' in sink_info and
                          'right_center' in sink_info and
                          'left_center' in sink_info and
                          'back_center' in sink_info)

                if has_all:
                    # Remove old values to recalculate
                    keys_to_remove = ['front_center', 'front_center_idx',
                                     'back_center', 'back_center_idx',
                                     'left_center', 'left_center_idx',
                                     'right_center', 'right_center_idx']
                    for key in keys_to_remove:
                        sink_info.pop(key, None)

                # Get spawn coordinates and position
                spawn_coords = sink_info.get('spawn_coordinates', [])
                sink_position = sink_info.get('position', {})

                if not spawn_coords or not sink_position:
                    print(f"  WARNING: {sink_id} missing spawn_coordinates or position")
                    continue

                print(f"  Processing {sink_id}...")

                # Calculate directional centers
                directional_centers = get_directional_centers(spawn_coords, sink_position, reachable_positions)

                if directional_centers:
                    # Add all directional centers to sink_info
                    for key, value in directional_centers.items():
                        sink_info[key] = value

                    updated_sinks += 1
                    print(f"    Added directional centers")
                    print(f"      front_center_idx: {directional_centers.get('front_center_idx')}")
                    print(f"      back_center_idx: {directional_centers.get('back_center_idx')}")
                    print(f"      left_center_idx: {directional_centers.get('left_center_idx')}")
                    print(f"      right_center_idx: {directional_centers.get('right_center_idx')}")
                else:
                    print(f"    WARNING: Could not calculate directional centers")

        except Exception as e:
            print(f"  Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    env.stop()

    # Save updated data
    output_path = '/home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/sink_spawn_coordinates.json'
    with open(output_path, 'w') as f:
        json.dump(sink_data, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total sinks: {total_sinks}")
    print(f"Updated sinks: {updated_sinks}")
    print(f"Output saved to: {output_path}")
    print()


def main():
    print("=" * 80)
    print("ADD DIRECTIONAL CENTERS TO SINK SPAWN COORDINATES")
    print("=" * 80)
    print()

    add_directional_centers_to_sinks()
    print("Done!")


if __name__ == '__main__':
    main()
