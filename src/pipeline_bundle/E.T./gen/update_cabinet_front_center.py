#!/usr/bin/env python3
"""
Update front_center in cabinet_spawn_coordinates.json to use the center of
the third of coordinates closest to the nearest reachable point.
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

from env.thor_env_thor5 import ThorEnv


def distance_2d(p1, p2):
    """Calculate 2D Euclidean distance (x, z only)."""
    return math.sqrt((p1['x'] - p2['x'])**2 +
                     (p1['z'] - p2['z'])**2)


def distance_3d(p1, p2):
    """Calculate 3D Euclidean distance."""
    return math.sqrt((p1['x'] - p2['x'])**2 +
                     (p1['y'] - p2['y'])**2 +
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


def get_front_center_from_closest_third(spawn_coordinates, cabinet_position, reachable_positions):
    """
    Get front_center from the third of spawn coordinates closest to nearest reachable point.

    Args:
        spawn_coordinates: List of spawn coordinate dicts
        cabinet_position: Position dict of the cabinet
        reachable_positions: List of reachable agent positions

    Returns:
        dict with keys: front_center, front_center_idx
    """
    if not spawn_coordinates or not reachable_positions:
        return {}

    # Find closest reachable position to cabinet
    min_dist = float('inf')
    closest_reachable = None
    for reach_pos in reachable_positions:
        dist = distance_2d(cabinet_position, reach_pos)
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

    # Sort by distance to find closest third
    spawn_with_distance.sort(key=lambda x: x[2])

    # Divide into thirds by distance
    num_coords = len(spawn_with_distance)
    third = num_coords // 3

    # Get the closest third (front third)
    front_third = spawn_with_distance[:third] if third > 0 else spawn_with_distance[:1]

    result = {}

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

    return result


def update_cabinet_front_centers():
    """
    Update front_center in cabinet_spawn_coordinates.json
    """
    # Load existing cabinet data
    cabinet_file = '/home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/cabinet_spawn_coordinates.json'

    with open(cabinet_file, 'r') as f:
        cabinet_data = json.load(f)

    env = ThorEnv(x_display='7')

    total_cabinets = 0
    updated_cabinets = 0

    # Process each floor plan
    for scene_num in range(1, 31):
        scene_name = f'FloorPlan{scene_num}'

        if scene_name not in cabinet_data:
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

            # Process each cabinet in this floorplan
            for cabinet_id, cabinet_info in cabinet_data[scene_name].items():
                total_cabinets += 1

                # Remove old front_center if it exists
                if 'front_center' in cabinet_info:
                    del cabinet_info['front_center']
                if 'front_center_idx' in cabinet_info:
                    del cabinet_info['front_center_idx']

                # Get spawn coordinates and position
                spawn_coords = cabinet_info.get('spawn_coordinates', [])
                cabinet_position = cabinet_info.get('position', {})

                if not spawn_coords or not cabinet_position:
                    print(f"  WARNING: {cabinet_id} missing spawn_coordinates or position")
                    continue

                print(f"  Processing {cabinet_id}...")

                # Calculate front_center from closest third
                front_center_data = get_front_center_from_closest_third(
                    spawn_coords, cabinet_position, reachable_positions
                )

                if front_center_data:
                    # Add front_center to cabinet_info
                    for key, value in front_center_data.items():
                        cabinet_info[key] = value

                    updated_cabinets += 1
                    print(f"    Added front_center")
                    print(f"      front_center_idx: {front_center_data.get('front_center_idx')}")
                else:
                    print(f"    WARNING: Could not calculate front_center")

        except Exception as e:
            print(f"  Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    env.stop()

    # Save updated data
    output_path = '/home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/cabinet_spawn_coordinates.json'
    with open(output_path, 'w') as f:
        json.dump(cabinet_data, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total cabinets: {total_cabinets}")
    print(f"Updated cabinets: {updated_cabinets}")
    print(f"Output saved to: {output_path}")
    print()


def main():
    print("=" * 80)
    print("UPDATE CABINET FRONT_CENTER COORDINATES")
    print("=" * 80)
    print()

    update_cabinet_front_centers()
    print("Done!")


if __name__ == '__main__':
    main()
