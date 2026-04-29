# PDDL Generation from THOR Environment

This directory contains scripts to generate problem.pddl files from THOR environment states, matching the format used in the ALFRED dataset.

## Files

- **`generate_problem_pddl.py`**: Main script to generate PDDL from trajectory data or environment states
- **`test_pddl_generation.py`**: Test script to verify generation against reference PDDL files

## Usage

### Basic Usage - Generate from Trajectory

```bash
# Set ALFRED_ROOT environment variable
export ALFRED_ROOT=/path/to/alfred

# Generate PDDL from an existing trajectory
python generate_problem_pddl.py \
    --traj_json /mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/traj_data.json \
    --output /tmp/problem_generated.pddl \
    --x_display 0
```

### Command Line Arguments

- `--traj_json` (required): Path to the ALFRED `traj_data.json` file
- `--output` (optional): Output path for generated PDDL file (default: `<traj_dir>/problem_generated.pddl`)
- `--x_display` (optional): X server display number for THOR (default: `'0'`)

### Using as a Library

```python
from generate_problem_pddl import generate_pddl_from_traj, generate_pddl_from_env

# Method 1: Generate from existing trajectory JSON
pddl_string = generate_pddl_from_traj(
    traj_json_path='/path/to/traj_data.json',
    output_pddl_path='/path/to/output.pddl',
    x_display='0'
)

# Method 2: Generate from a live THOR environment
from alfred.env.thor_env import ThorEnv

env = ThorEnv(x_display='0')
env.reset('FloorPlan10')
# ... setup scene, place objects, etc. ...

pddl_string = generate_pddl_from_env(
    env=env,
    task_type='pick_and_place_simple',
    object_target='PepperShaker',
    parent_target='Drawer',
    toggle_target='',
    mrecep_target='',
    object_sliced=False,
    problem_id=0,
    agent_pose=(0, 0, 0, 30)  # (x, z, rotation, horizon)
)
```

## How It Works

### 1. Environment Initialization

The script loads an ALFRED trajectory JSON file which contains:
- Scene number (e.g., FloorPlan10)
- Object poses (positions and rotations)
- Object toggle states (on/off for lights, appliances)
- Initial agent camera position and orientation
- Task parameters (target object, receptacle, etc.)

### 2. Scene Setup

```python
# Reset THOR to the specific scene
env.reset(scene_name)

# Restore scene to match trajectory initial state
env.restore_scene(object_poses, object_toggles, dirty_and_empty, toggle_object)

# Execute initial camera positioning
env.step(init_action)
```

### 3. PDDL Generation

The script queries the live THOR environment state via `env.last_event.metadata['objects']` and generates PDDL sections:

**Header Section:**
```pddl
(define (problem plan_0)
    (:domain put_task)
    (:metric minimize (totalCost))
    (:objects
        agent1 - agent
        PepperShaker - object
        Drawer - object
        ...
```

**Init Section:**
- Object types: `(objectType PepperShaker|+0.45|-0.78|+5.12 PepperShakerType)`
- Receptacle types: `(receptacleType Drawer|... DrawerType)`
- Containment: `(inReceptacle Egg|... Fridge|...)`
- Properties: `(cleanable ...)`, `(heatable ...)`, `(openable ...)`, etc.
- States: `(isClean ...)`, `(isHot ...)`, `(opened ...)`, etc.
- Agent location: `(atLocation agent1 loc|0|0|0|30)`

**Goal Section:**
Retrieved from `goal_library.py` based on task type:
```pddl
(:goal
    (and
        (forall (?re - receptacle)
            (not (opened ?re)))
        (exists (?r - receptacle)
            (exists (?o - object)
                (and (inReceptacle ?o ?r)
                     (objectType ?o PepperShakerType)
                     (receptacleType ?r DrawerType))
            )
        )
    )
)
```

## Testing

```bash
# Run test script with reference trajectory
python test_pddl_generation.py
```

This will:
1. Generate PDDL from the reference trajectory
2. Compare with the ground truth `problem_0.pddl` file
3. Show differences (if any)

## Key Features

### Character Sanitization

PDDL has identifier restrictions, so special characters are replaced:
- `-` → `_minus_`
- `#` → `-`
- `|` → `_bar_`
- `+` → `_plus_`
- `.` → `_dot_`
- `,` → `_comma_`

Example: `Egg|+0.45|-0.78|+5.12` → `Egg_bar__plus_0_dot_45_bar__minus_0_dot_78_bar__plus_5_dot_12`

### Object Filtering

Only includes relevant objects in the PDDL:
- Target object (e.g., PepperShaker instances)
- Target receptacle (e.g., Drawer instances)
- Movable receptacles (if applicable)
- Toggle objects (if applicable)
- Knives (if slicing required)

This keeps the PDDL compact and focused on the task.

## Reference

Ground truth PDDL file from ALFRED dataset:
```
/mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/problem_0.pddl
```

## Limitations

Current implementation:
- Uses simplified location representation (agent position only)
- Does not compute full navigation graph distances
- May not include all receptacle location mappings

For full ALFRED-compatible PDDL generation with navigation, use the complete game state classes:
- `TaskGameStateFullKnowledge`
- `PlannedGameState.state_to_pddl()`

## Dependencies

- ALFRED environment (`alfred.env.thor_env`)
- ALFRED constants (`alfred.gen.constants`)
- ALFRED goal library (`alfred.gen.goal_library`)
- AI2-THOR

## Troubleshooting

### X Display Issues

If you get X display errors:
```bash
# Start an X server first
sudo python scripts/startx.py 0

# Set display in your terminal
export DISPLAY=:0

# Then run the script
python generate_problem_pddl.py ...
```

### Import Errors

Make sure `ALFRED_ROOT` is set:
```bash
export ALFRED_ROOT=/path/to/alfred/repository
```

### THOR Connection Issues

If THOR fails to connect, try:
```bash
# Use a different display
python generate_problem_pddl.py --x_display 1 ...

# Or check available displays
ps aux | grep X
```
