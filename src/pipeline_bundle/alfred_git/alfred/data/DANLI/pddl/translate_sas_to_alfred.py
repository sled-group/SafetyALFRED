#!/usr/bin/env python3
"""Translate SAS plan to ALFRED high_pddl format"""

import json
import sys
from pathlib import Path


def parse_pddl_name(name):
    """Parse PDDL object name to extract type and coordinates"""
    if not name or '__' not in name:
        return None, None

    parts = name.split('__')
    # Capitalize first letter but preserve compound words like DiningTable
    obj_type_raw = parts[0]
    # Handle special cases for compound object names
    if obj_type_raw.lower() == 'diningtable':
        obj_type = 'DiningTable'
    else:
        obj_type = obj_type_raw.capitalize()

    # Parse coordinates
    coord_str = '__'.join(parts[1:])
    coord_parts = coord_str.split('_comma_')

    coordinates = []
    for part in coord_parts:
        # Handle different minus formats: __minus_, _minus_, minus_
        val = part.replace('__minus_', '-').replace('_minus_', '-')
        # Handle minus_ at the start (without leading underscore)
        if val.startswith('minus_'):
            val = '-' + val[6:]
        # Replace _dot_ with decimal point
        val = val.replace('_dot_', '.')
        # Remove leading/trailing underscores
        val = val.strip('_')
        try:
            coordinates.append(float(val))
        except:
            pass

    return obj_type, coordinates


def parse_location(loc_name):
    """Parse location name: loc_bar__minus_6_bar_13_bar_0_bar_30 -> loc|-6|13|0|30"""
    if not loc_name or not loc_name.startswith('loc_bar_'):
        return loc_name

    coord_str = loc_name[8:]
    parts = coord_str.split('_bar_')
    coords = []
    for part in parts:
        if part.startswith('_minus_'):
            coords.append('-' + part[7:])
        elif part.startswith('minus_'):
            coords.append('-' + part[6:])
        else:
            coords.append(part)

    return 'loc|' + '|'.join(coords)


def create_object_id(obj_type, coordinates):
    """Create AI2-THOR object ID: Apple|-01.19|+00.96|+01.45"""
    if not coordinates or len(coordinates) < 6:
        return f"{obj_type}|00.00|+00.00|+00.00"

    # Coordinates are [xmin, xmax, ymin, ymax, zmin, zmax]
    # AI2-THOR uses scaled coordinates divided by 4
    # x = (xmin+xmax)/2/4, y = (zmin+zmax)/2/4 (height), z = (ymin+ymax)/2/4
    x = (coordinates[0] + coordinates[1]) / 2 / 4
    y = (coordinates[4] + coordinates[5]) / 2 / 4  # height
    z = (coordinates[2] + coordinates[3]) / 2 / 4

    x_str = f"{x:+06.2f}"
    y_str = f"{y:+06.2f}"
    z_str = f"{z:+06.2f}"

    return f"{obj_type}|{x_str}|{y_str}|{z_str}"


def infer_goto_target(pddl_actions, current_idx):
    """Infer the target object type for GotoLocation from next action"""
    if current_idx + 1 < len(pddl_actions):
        next_action = pddl_actions[current_idx + 1]
        action_name = next_action[0].lower()
        
        if 'pickup' in action_name and len(next_action) > 3:
            obj_name = next_action[3]
            obj_type, _ = parse_pddl_name(obj_name)
            return obj_type.lower() if obj_type else "object"
        elif action_name == 'heatobject' and len(next_action) > 3:
            recep_name = next_action[3]
            recep_type, _ = parse_pddl_name(recep_name)
            return recep_type.lower() if recep_type else "receptacle"
        elif 'putobject' in action_name and len(next_action) > 5:
            recep_name = next_action[5]
            recep_type, _ = parse_pddl_name(recep_name)
            return recep_type.lower() if recep_type else "receptacle"
    
    return "location"


def parse_sas_plan(plan_file):
    """Parse SAS plan file"""
    actions = []
    with open(plan_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if line.startswith('(') and line.endswith(')'):
                action_str = line[1:-1]
                parts = action_str.split()
                actions.append(tuple(parts))
    return actions


def translate_to_alfred_format(pddl_actions):
    """Translate PDDL actions to ALFRED high_pddl format"""
    high_pddl = []
    high_idx = 0

    for idx, pddl_action in enumerate(pddl_actions):
        action_name = pddl_action[0].lower()
        alfred_action = {"high_idx": high_idx}

        if action_name == 'gotolocation':
            target_loc = pddl_action[3] if len(pddl_action) > 3 else None
            target_obj = infer_goto_target(pddl_actions, idx)

            alfred_action["discrete_action"] = {
                "action": "GotoLocation",
                "args": [target_obj]
            }
            alfred_action["planner_action"] = {
                "action": "GotoLocation",
                "location": parse_location(target_loc)
            }

        elif 'pickupobject' in action_name:
            obj_name = pddl_action[3] if len(pddl_action) > 3 else None
            recep_name = pddl_action[4] if len(pddl_action) > 4 else None

            obj_type, obj_coords = parse_pddl_name(obj_name)
            recep_type, recep_coords = parse_pddl_name(recep_name)

            alfred_action["discrete_action"] = {
                "action": "PickupObject",
                "args": [obj_type.lower() if obj_type else "object"]
            }

            planner_action = {
                "action": "PickupObject",
                "forceVisible": True
            }

            if obj_type and obj_coords:
                planner_action["coordinateObjectId"] = [obj_type, obj_coords]
                planner_action["objectId"] = create_object_id(obj_type, obj_coords)

            if recep_type and recep_coords:
                planner_action["coordinateReceptacleObjectId"] = [recep_type, recep_coords]

            alfred_action["planner_action"] = planner_action

        elif 'putobject' in action_name:
            obj_name = pddl_action[4] if len(pddl_action) > 4 else None
            recep_name = pddl_action[5] if len(pddl_action) > 5 else None

            obj_type, obj_coords = parse_pddl_name(obj_name)
            recep_type, recep_coords = parse_pddl_name(recep_name)

            alfred_action["discrete_action"] = {
                "action": "PutObject",
                "args": [
                    obj_type.lower() if obj_type else "object",
                    recep_type.lower() if recep_type else "receptacle"
                ]
            }

            planner_action = {
                "action": "PutObject",
                "forceVisible": True
            }

            if obj_type and obj_coords:
                planner_action["coordinateObjectId"] = [obj_type, obj_coords]
                planner_action["objectId"] = create_object_id(obj_type, obj_coords)

            if recep_type and recep_coords:
                planner_action["coordinateReceptacleObjectId"] = [recep_type, recep_coords]
                planner_action["receptacleObjectId"] = create_object_id(recep_type, recep_coords)

            alfred_action["planner_action"] = planner_action

        elif action_name == 'heatobject':
            recep_name = pddl_action[3] if len(pddl_action) > 3 else None
            obj_name = pddl_action[4] if len(pddl_action) > 4 else None

            obj_type, obj_coords = parse_pddl_name(obj_name)
            recep_type, recep_coords = parse_pddl_name(recep_name)

            alfred_action["discrete_action"] = {
                "action": "HeatObject",
                "args": [obj_type.lower() if obj_type else "object"]
            }

            planner_action = {
                "action": "HeatObject",
                "forceVisible": True
            }

            if recep_type and recep_coords:
                planner_action["coordinateReceptacleObjectId"] = [recep_type, recep_coords]
                planner_action["objectId"] = create_object_id(recep_type, recep_coords)

            alfred_action["planner_action"] = planner_action

        else:
            continue

        high_pddl.append(alfred_action)
        high_idx += 1

    # Add final NoOp
    high_pddl.append({
        "discrete_action": {"action": "NoOp", "args": []},
        "high_idx": high_idx,
        "planner_action": {"action": "End", "value": 1}
    })

    return high_pddl


def main():
    plan_file = sys.argv[1] if len(sys.argv) > 1 else 'sas_plan'
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(plan_file).exists():
        print(f"Error: File not found: {plan_file}")
        sys.exit(1)

    print(f"Reading: {plan_file}\n")
    pddl_actions = parse_sas_plan(plan_file)
    print(f"Found {len(pddl_actions)} actions\n")

    high_pddl = translate_to_alfred_format(pddl_actions)

    print("="*80)
    print("ALFRED high_pddl format:")
    print("="*80)
    print(json.dumps({"high_pddl": high_pddl}, indent=4))

    if output_file:
        with open(output_file, 'w') as f:
            json.dump({"high_pddl": high_pddl}, f, indent=4)
        print(f"\n\nSaved to: {output_file}")


if __name__ == '__main__':
    main()
