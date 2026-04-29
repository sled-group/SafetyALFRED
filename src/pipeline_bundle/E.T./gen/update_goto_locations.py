#!/usr/bin/env python3
"""
Script to update GotoLocation high_pddl actions with accurate location encoding
based on the actual agent position after completing the GotoLocation action.

Usage:
    python update_goto_locations.py <trajectory_directory>

Example:
    python update_goto_locations.py /mnt/external-ssd/generated_safety_2.1.0/train/.../traj_data_safety_traj_spoilage.json
"""

import os
import sys
import json
import ast
import glob
from pathlib import Path


# Location encoding parameters
STEP_SIZE = 0.25
ROTATIONS = [0, 90, 180, 270]


def encode_location(x, y, z, rotation_y, horizon):
    """
    Encode agent position into location string format: loc|x_steps|z_steps|rotation_idx|horizon

    Args:
        x: Agent x position
        y: Agent y position
        z: Agent z position
        rotation_y: Agent rotation around y-axis (0, 90, 180, or 270)
        horizon: Camera horizon angle

    Returns:
        Location string in format: loc|x_steps|z_steps|rotation_idx|horizon
    """
    # Convert x, z to step counts
    x_steps = round(x / STEP_SIZE)
    z_steps = round(z / STEP_SIZE)

    # Find rotation index
    rotation_normalized = round(rotation_y) % 360
    if rotation_normalized not in ROTATIONS:
        # Find closest rotation
        rotation_normalized = min(ROTATIONS, key=lambda r: abs(r - rotation_normalized))
    rotation_idx = ROTATIONS.index(rotation_normalized)

    # Round horizon to nearest integer
    horizon_int = round(horizon)

    return f"loc|{x_steps}|{z_steps}|{rotation_idx}|{horizon_int}"


def get_agent_position_from_metadata(metadata_path):
    """
    Extract agent position from metadata file.

    Args:
        metadata_path: Path to metadata_{idx}.txt file

    Returns:
        Dict with keys: x, y, z, rotation_y, horizon
    """
    with open(metadata_path, 'r') as f:
        data = ast.literal_eval(f.read())

    if 'agent' not in data:
        raise ValueError(f"No agent data found in {metadata_path}")

    agent = data['agent']
    return {
        'x': agent['position']['x'],
        'y': agent['position']['y'],
        'z': agent['position']['z'],
        'rotation_y': agent['rotation']['y'],
        'horizon': agent.get('cameraHorizon', 0)
    }


def update_goto_locations(traj_dir, force=False):
    """
    Update GotoLocation actions in traj_data.json with accurate locations.

    Args:
        traj_dir: Path to trajectory directory (ends with .json)
        force: If True, process even if already processed
    """
    traj_dir = Path(traj_dir)

    # Load traj_data.json
    traj_data_path = traj_dir / 'traj_data.json'
    if not traj_data_path.exists():
        raise FileNotFoundError(f"traj_data.json not found in {traj_dir}")

    with open(traj_data_path, 'r') as f:
        traj_data = json.load(f)

    # Check if metadata directory exists
    metadata_dir = traj_dir / 'metadata'
    if not metadata_dir.exists():
        print(f"Warning: No metadata directory found in {traj_dir}")
        return False

    # Track updates
    updates_made = 0

    # Check if already processed - if no GotoLocation has loc|0|0|0|0, skip
    if not force:
        already_processed = True
        for high_action in traj_data['plan']['high_pddl']:
            discrete_action = high_action.get('discrete_action', {})
            if discrete_action.get('action') == 'GotoLocation':
                location = high_action.get('planner_action', {}).get('location', '')
                if location == 'loc|0|0|0|0' or not location:
                    already_processed = False
                    break

        if already_processed:
            print(f"Skipping {traj_data_path} - already processed (no loc|0|0|0|0 found)")
            return False

    # Process each high-level action
    for high_idx, high_action in enumerate(traj_data['plan']['high_pddl']):
        # Safely check if this is a GotoLocation action
        # Some trajectories have action in discrete_action, others only have planner_action with location
        discrete_action = high_action.get('discrete_action', {})
        action_name = discrete_action.get('action', '')
        planner_action = high_action.get('planner_action', {})

        # Check if it's a GotoLocation by action name OR by having a location field
        is_goto = (action_name == 'GotoLocation' or
                  (not action_name and 'location' in planner_action))

        if not is_goto:
            continue

        # Find the last low-level action with this high_idx
        last_low_idx = None
        for low_idx, low_action in enumerate(traj_data['plan']['low_actions']):
            if low_action['high_idx'] == high_idx:
                last_low_idx = low_idx

        if last_low_idx is None:
            print(f"Warning: No low-level actions found for GotoLocation at high_idx={high_idx}")
            continue

        # Metadata files are ordered and correspond to low_actions by order
        # Get list of metadata files sorted by their index
        available_metadata = sorted(metadata_dir.glob('metadata_*.txt'),
                                    key=lambda x: int(x.stem.split('_')[1]))

        if not available_metadata:
            print(f"Warning: No metadata files found in {metadata_dir}")
            continue

        # The metadata files correspond to positions AFTER each action
        # metadata_0.txt is the initial position (before any actions)
        # metadata_1.txt is the position after low_actions[0]
        # So we need metadata at index (last_low_idx + 1) to get position after last_low_idx action
        metadata_idx = last_low_idx + 1

        if metadata_idx >= len(available_metadata):
            print(f"Warning: Not enough metadata files ({len(available_metadata)}) for low_idx={last_low_idx} (need {metadata_idx+1})")
            continue

        metadata_path = available_metadata[metadata_idx]

        try:
            # Get agent position from metadata
            agent_pos = get_agent_position_from_metadata(metadata_path)

            # Encode location
            new_location = encode_location(
                agent_pos['x'],
                agent_pos['y'],
                agent_pos['z'],
                agent_pos['rotation_y'],
                agent_pos['horizon']
            )

            # Update the location
            old_location = high_action['planner_action']['location']
            high_action['planner_action']['location'] = new_location

            print(f"Updated high_idx={high_idx}: {old_location} -> {new_location}")
            print(f"  Agent pos: x={agent_pos['x']:.2f}, z={agent_pos['z']:.2f}, "
                  f"rot={agent_pos['rotation_y']:.0f}°, horizon={agent_pos['horizon']:.0f}°")
            updates_made += 1

        except Exception as e:
            print(f"Error processing high_idx={high_idx}: {e}")
            continue

    if updates_made > 0:
        # Create backup
        backup_path = traj_dir / 'traj_data.json.backup'
        if not backup_path.exists():
            with open(backup_path, 'w') as f:
                with open(traj_data_path, 'r') as orig:
                    f.write(orig.read())
            print(f"Created backup: {backup_path}")

        # Save updated traj_data.json
        with open(traj_data_path, 'w') as f:
            json.dump(traj_data, f, sort_keys=True, indent=4)

        print(f"\nSuccessfully updated {updates_made} GotoLocation actions in {traj_data_path}")
        return True
    else:
        print(f"No GotoLocation actions updated in {traj_dir}")
        return False


def process_directory_recursive(root_dir, pattern='**/traj_data_safety_traj_*.json'):
    """
    Process all trajectory directories matching the pattern recursively.

    Args:
        root_dir: Root directory to search
        pattern: Glob pattern for trajectory directories
    """
    root_path = Path(root_dir)
    traj_dirs = list(root_path.glob(pattern))

    print(f"Found {len(traj_dirs)} trajectory directories to process")

    success_count = 0
    for i, traj_dir in enumerate(traj_dirs, 1):
        print(f"\n[{i}/{len(traj_dirs)}] Processing: {traj_dir}")
        try:
            if update_goto_locations(traj_dir):
                success_count += 1
        except Exception as e:
            print(f"Error: {e}")

    print(f"\n\nSummary: Updated {success_count}/{len(traj_dirs)} trajectories")


def process_from_metadata_file(metadata_file, base_dir='/mnt/external-ssd/generated_safety_2.1.0'):
    """
    Process trajectories listed in a metadata file.

    Args:
        metadata_file: Path to processed_metadata.txt
        base_dir: Base directory containing the trajectories
    """
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    traj_dirs = []
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Error processing'):
                continue

            # Format: path/to/traj_dir.json;success_flag
            parts = line.split(';')
            if len(parts) >= 1:
                traj_path = parts[0]
                # Only add valid paths
                if traj_path.startswith(('train/', 'valid_seen/', 'valid_unseen/')):
                    full_path = Path(base_dir) / traj_path
                    if full_path.exists():
                        traj_dirs.append(full_path)

    print(f"Found {len(traj_dirs)} trajectory directories from metadata file")

    success_count = 0
    for i, traj_dir in enumerate(traj_dirs, 1):
        print(f"\n[{i}/{len(traj_dirs)}] Processing: {traj_dir}")
        try:
            if update_goto_locations(traj_dir):
                success_count += 1
        except Exception as e:
            print(f"Error: {e}")

    print(f"\n\nSummary: Updated {success_count}/{len(traj_dirs)} trajectories")


def main():
    if len(sys.argv) < 2:
        print("Usage: python update_goto_locations.py <trajectory_directory_or_metadata_file> [--force]")
        print("\nExample (single trajectory):")
        print("  python update_goto_locations.py /mnt/external-ssd/generated_safety_2.1.0/train/.../traj_data_safety_traj_spoilage.json")
        print("\nExample (recursive from root):")
        print("  python update_goto_locations.py /mnt/external-ssd/generated_safety_2.1.0/")
        print("\nExample (from metadata file):")
        print("  python update_goto_locations.py /mnt/external-ssd/generated_safety_2.1.0/processed_metadata.txt")
        print("\nUse --force to reprocess already-processed trajectories")
        sys.exit(1)

    path = sys.argv[1]
    force = '--force' in sys.argv

    # Check if it's a metadata file
    if os.path.isfile(path) and path.endswith('processed_metadata.txt'):
        base_dir = os.path.dirname(path)
        process_from_metadata_file(path, base_dir)
    elif os.path.isfile(os.path.join(path, 'traj_data.json')):
        # Single trajectory directory
        update_goto_locations(path, force=force)
    elif os.path.isdir(path):
        # Process recursively
        process_directory_recursive(path)
    else:
        print(f"Error: {path} is not a valid directory or metadata file")
        sys.exit(1)


if __name__ == '__main__':
    main()
