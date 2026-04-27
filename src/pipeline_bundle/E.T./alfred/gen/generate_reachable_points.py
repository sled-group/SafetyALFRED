#!/usr/bin/env python3
"""
Generate reachable points for all FloorPlans in AI2-THOR 5.0
Saves to /home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/reachable_points.json
"""

import json
import os
import sys
from ai2thor.controller import Controller

def get_reachable_points_for_scene(scene_name, x_display='7'):
    """Get all reachable points for a given FloorPlan."""
    print(f"Processing {scene_name}...")

    controller = Controller(
        scene=scene_name,
        gridSize=0.25,
        width=300,
        height=300,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        x_display=x_display
    )

    # Get reachable positions
    event = controller.step(action='GetReachablePositions')

    if event.metadata['lastActionSuccess']:
        reachable_positions = event.metadata['actionReturn']
        print(f"  Found {len(reachable_positions)} reachable points")
    else:
        print(f"  ERROR: Failed to get reachable positions for {scene_name}")
        reachable_positions = []

    controller.stop()

    return reachable_positions


def main():
    # All kitchen FloorPlans in AI2-THOR (1-30)
    floorplans = [f"FloorPlan{i}" for i in range(1, 31)]

    reachable_data = {}

    for floorplan in floorplans:
        try:
            reachable_points = get_reachable_points_for_scene(floorplan)
            reachable_data[floorplan] = {
                'scene_name': floorplan,
                'num_points': len(reachable_points),
                'reachable_positions': reachable_points
            }
        except Exception as e:
            print(f"ERROR processing {floorplan}: {e}")
            reachable_data[floorplan] = {
                'scene_name': floorplan,
                'num_points': 0,
                'reachable_positions': [],
                'error': str(e)
            }

    # Save to JSON file
    output_path = '/home/josue/Desktop/Research/SLED/MSS/spawn_coordinates/reachable_points.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(reachable_data, f, indent=2)

    print(f"\n✓ Successfully saved reachable points to {output_path}")
    print(f"  Total FloorPlans processed: {len(reachable_data)}")
    total_points = sum(data['num_points'] for data in reachable_data.values())
    print(f"  Total reachable points: {total_points}")


if __name__ == '__main__':
    main()
