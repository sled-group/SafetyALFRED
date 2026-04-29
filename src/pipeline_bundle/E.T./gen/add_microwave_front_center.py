#!/usr/bin/env python3
"""
Add front_center and front_center_idx fields to microwave_spawn_coordinates.json
using the same logic as get_countertop_spawn_coordinates.py
"""
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
import sys
import json
import glob
import math
from collections import OrderedDict

# Clean up any existing AI2-THOR pipes
for pipe in glob.glob('/tmp/ai2thor_*'):
    try:
        os.remove(pipe)
    except:
        pass

sys.path.insert(0, '/home/josue/Desktop/Research/SLED/MSS/E.T.')

from env.thor_env_thor5 import ThorEnv


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


def add_front_center_to_microwaves():
    """
    Add front_center and front_center_idx to existing microwave_spawn_coordinates.json
    """
    # Load existing microwave data
    microwave_file = '/home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/microwave_spawn_coordinates.json'

    with open(microwave_file, 'r') as f:
        microwave_data = json.load(f)

    env = ThorEnv(x_display='7')

    total_microwaves = 0
    updated_microwaves = 0

    # Iterate through kitchen floorplans (1-30)
    for scene_num in range(1, 31):
        scene_name = f'FloorPlan{scene_num}'

        if scene_name not in microwave_data:
            continue

        print(f"Processing {scene_name}...")

        try:
            # Reset to the scene
            env.reset(scene_name, silent=True)

            # Get reachable positions for front_center calculation
            reachable_event = env.step({'action': 'GetReachablePositions'})
            reachable_positions = []
            if reachable_event.metadata['lastActionSuccess']:
                reachable_positions = reachable_event.metadata['actionReturn']

            # Process each microwave in this floorplan
            for microwave_id, microwave_info in microwave_data[scene_name].items():
                total_microwaves += 1

                # Skip if already has front_center
                if 'front_center' in microwave_info:
                    print(f"  {microwave_id} already has front_center, skipping")
                    continue

                spawn_coordinates = microwave_info.get('spawn_coordinates', [])
                microwave_position = microwave_info.get('position', {})

                if not spawn_coordinates:
                    print(f"  {microwave_id}: No spawn coordinates")
                    continue

                print(f"  Processing {microwave_id}...")

                # Calculate front_center if we have reachable positions
                if reachable_positions:
                    # Find closest reachable position to microwave
                    min_reach_dist = float('inf')
                    closest_reachable = None

                    for reach_pos in reachable_positions:
                        dist = distance_2d(microwave_position, reach_pos)
                        if dist < min_reach_dist:
                            min_reach_dist = dist
                            closest_reachable = reach_pos

                    if closest_reachable:
                        # Find spawn coordinates closest to the closest reachable position
                        spawn_distances = []
                        for i, spawn_coord in enumerate(spawn_coordinates):
                            dist = distance_2d(spawn_coord, closest_reachable)
                            spawn_distances.append((dist, i, spawn_coord))

                        # Sort by distance
                        spawn_distances.sort(key=lambda x: x[0])

                        # Take the closest 1/4 of spawn coordinates as "front" coordinates
                        num_front = max(1, min(10, len(spawn_coordinates) // 4))
                        front_coords = [coord for dist, idx, coord in spawn_distances[:num_front]]

                        # Find the centermost among the front coordinates
                        front_center = find_center_coordinate(front_coords)

                        # Find which original spawn coordinate is the front_center
                        min_dist_to_fc = float('inf')
                        front_center_idx = 0
                        for dist, idx, coord in spawn_distances[:num_front]:
                            fc_dist = distance_3d(coord, front_center)
                            if fc_dist < min_dist_to_fc:
                                min_dist_to_fc = fc_dist
                                front_center_idx = idx

                        microwave_info['front_center'] = front_center
                        microwave_info['front_center_idx'] = front_center_idx
                        updated_microwaves += 1
                        print(f"    Added front_center at index {front_center_idx}")
                    else:
                        print(f"    No closest reachable position found")
                else:
                    print(f"    No reachable positions available")

        except Exception as e:
            print(f"  Error processing {scene_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    env.stop()

    # Save updated data
    output_path = '/home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/microwave_spawn_coordinates.json'
    with open(output_path, 'w') as f:
        json.dump(microwave_data, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total microwaves: {total_microwaves}")
    print(f"Updated microwaves: {updated_microwaves}")
    print(f"Output saved to: {output_path}")
    print()


def main():
    print("=" * 80)
    print("ADD FRONT_CENTER TO MICROWAVE SPAWN COORDINATES")
    print("=" * 80)
    print()

    add_front_center_to_microwaves()
    print("Done!")


if __name__ == '__main__':
    main()
