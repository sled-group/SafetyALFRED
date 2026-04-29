# Plan Rendering Script - Documentation

## Overview

The `render_plan.py` script generates a PDDL problem from a trajectory, creates a plan using Fast Downward, and renders the execution in THOR while saving video frames.

## Features

- ✅ Generates PDDL from trajectory JSON
- ✅ Runs Fast Downward planner to create optimal plans
- ✅ Converts PDDL actions to THOR API calls
- ✅ Executes plan in THOR environment
- ✅ Saves video frames for each step
- ✅ Creates MP4 video of execution
- ✅ Logs execution success/failure for each step

## Usage

```bash
cd /home/josue/Desktop/Research/SLED/MSS/E.T./alfred/gen
source ../../et_env_safety/bin/activate

export ALFRED_ROOT=/home/josue/Desktop/Research/SLED/MSS/E.T./alfred
export DISPLAY=:7

python render_plan.py \
  --traj_json /path/to/traj_data.json \
  --domain /path/to/domain.pddl \
  --output_dir /path/to/output \
  --x_display 7
```

## Example

```bash
python render_plan.py \
  --traj_json /mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/traj_data.json \
  --domain /home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/domain.pddl \
  --output_dir /tmp/plan_render \
  --x_display 7
```

## Output Files

The script creates the following files in the output directory:

```
output_dir/
├── problem.pddl              # Generated PDDL problem file
├── plan.txt                  # Human-readable plan (4 actions)
├── sas_plan                  # Fast Downward plan file
├── execution_log.json        # Detailed execution log with success/failure for each action
├── frames/                   # Directory with PNG frames
│   ├── 000000000.png        # Initial frame
│   ├── 000000001.png        # After action 1
│   ├── ...
│   └── 000000006.png        # Final frame
├── plan_execution.mp4        # Video of plan execution
└── debug.json                # Scene objects metadata (if errors occur)
```

## Execution Log Format

The `execution_log.json` file contains detailed information about each action:

```json
[
  {
    "step": 1,
    "pddl_action": "gotolocation agent1 loc_start loc_end",
    "thor_action": {
      "action": "TeleportFull",
      "x": -0.75,
      "y": 0.9009992,
      "z": -2.0,
      "rotation": 270.0,
      "horizon": 0
    },
    "success": true
  },
  {
    "step": 2,
    "pddl_action": "pickupobjectinreceptacle1 ...",
    "thor_action": {
      "action": "PickupObject",
      "objectId": "PepperShaker|-00.92|+00.93|-01.39"
    },
    "success": true
  }
]
```

## Supported PDDL Actions

The script handles the following PDDL actions:

### Navigation
- `gotolocation` → `TeleportFull` in THOR

### Object Manipulation
- `pickupobjectinreceptacle1` → `PickupObject`
- `pickupobjectnoreceptacle` → `PickupObject`
- `putobjectinreceptacle1` → `OpenObject` (if needed) + `PutObject`

### Receptacle Manipulation
- `openobject` → `OpenObject`
- `closeobject` → `CloseObject`

## PDDL to THOR Conversion

The script automatically converts PDDL naming conventions to THOR object IDs:

### Location Format
**PDDL**: `loc_bar__minus_14_bar_5_bar_3_bar_30`
- Format: `loc|x|y|z|rotation` where coordinates are in grid units
- Conversion: x=-14*0.25=-3.5m, y=5*0.25=1.25m, z=3, rotation=30°*90=270°

**THOR**: `{'x': -3.5, 'y': 0.9009992, 'z': 1.25, 'rotation': 270, 'horizon': 3}`

### Object ID Format
**PDDL**: `peppershaker_bar__minus_00_dot_92_bar__plus_00_dot_93_bar__minus_01_dot_39`

**THOR**: `PepperShaker|-00.92|+00.93|-01.39`

Conversion rules:
1. Extract type from first part: `peppershaker` → `PepperShaker`
2. Handle multi-word types: `countertop` → `CounterTop`
3. Convert coordinates: `_bar_` → `|`, `_minus_` → `-`, `_plus_` → `+`, `_dot_` → `.`

## Pipeline Flow

```
1. Load trajectory JSON
   ↓
2. Generate PDDL (using generate_problem_pddl_full.py)
   ↓
3. Run Fast Downward planner
   ↓
4. Initialize THOR environment
   ↓
5. For each action in plan:
   - Convert PDDL action to THOR actions
   - Execute in THOR
   - Save frame
   - Log success/failure
   ↓
6. Create video from frames
   ↓
7. Save execution log
```

## Example Output

### Console Output
```
================================================================================
PLAN RENDERING
================================================================================

[1/5] Generating PDDL from trajectory...
✓ Generated PDDL: /tmp/plan_render/problem.pddl

[2/5] Running Fast Downward planner...
✓ Plan generated: 4 actions in 0.58s
  1. gotolocation agent1 loc_bar__minus_14_bar_5_bar_3_bar_30 loc_bar__minus_3_bar__minus_8_bar_0_bar_45
  2. pickupobjectinreceptacle1 agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 peppershaker_... countertop_...
  3. gotolocation agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 loc_bar_0_bar__minus_6_bar_0_bar_45
  4. putobjectinreceptacle1 agent1 loc_bar_0_bar__minus_6_bar_0_bar_45 peppershakertype peppershaker_... drawer_...

[3/5] Initializing THOR environment...
✓ Environment initialized: FloorPlan10

[4/5] Executing plan in THOR...

Step 1/4: gotolocation agent1 loc_bar__minus_14_bar_5_bar_3_bar_30 loc_bar__minus_3_bar__minus_8_bar_0_bar_45
  Executing: TeleportFull
  ✓ Success

Step 2/4: pickupobjectinreceptacle1 agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 peppershaker_... countertop_...
  Executing: PickupObject
  ✓ Success

Step 3/4: gotolocation agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 loc_bar_0_bar__minus_6_bar_0_bar_45
  Executing: TeleportFull
  ✓ Success

Step 4/4: putobjectinreceptacle1 agent1 loc_bar_0_bar__minus_6_bar_0_bar_45 peppershakertype peppershaker_... drawer_...
  Executing: OpenObject
  ⚠ Failed: ...
  Executing: PutObject
  ⚠ Failed: No valid Receptacle found

✓ Executed 4 actions, saved 6 frames

[5/5] Creating video...
✓ Video saved: /tmp/plan_render/plan_execution.mp4

================================================================================
RENDERING COMPLETE
================================================================================

Outputs saved to: /tmp/plan_render
  - PDDL: problem.pddl
  - Plan: plan.txt
  - Frames: frames/
  - Video: plan_execution.mp4
  - Log: execution_log.json
```

## Known Limitations

1. **Force Actions**: Some actions use `forceAction=True` which may not match exact physics
2. **Drawer/Cabinet Opening**: Opening actions may fail if agent is not in exact position required by THOR
3. **Put Actions**: May fail if receptacle is not properly opened or agent is not close enough
4. **Navigation**: Uses teleport instead of smooth navigation for speed

## Troubleshooting

### "Object ID appears to be invalid"
- The PDDL object name conversion may need adjustment
- Check the `debug.json` file to see actual object IDs in scene
- Verify object type name mapping in `pddl_action_to_thor_actions()`

### "No valid Receptacle found"
- Agent may not be close enough to receptacle
- Receptacle may need to be opened first
- Check if receptacle is reachable from current location

### Video not created
- Check that ffmpeg is installed
- Verify frames were saved in `frames/` directory
- Check video_util logs

### Plan generation fails
- Verify domain.pddl is compatible
- Check that all required predicates are present in problem PDDL
- Increase planner timeout if needed

## Performance

- **PDDL Generation**: ~5-10 seconds (includes THOR initialization)
- **Planning**: 0.5-1.0 seconds for simple tasks
- **Execution**: ~1 second per action
- **Video Creation**: ~1 second for short videos

## Dependencies

- `alfred.env.thor_env` - THOR environment wrapper
- `alfred.gen.utils.video_util` - Video creation utilities
- `Fast Downward` - PDDL planner
- `PIL` - Image processing
- `ffmpeg` - Video encoding

## Related Scripts

- `generate_problem_pddl_full.py` - Generates PDDL from trajectories
- `verify_pddl_with_planner.py` - Verifies PDDL by comparing plans
- `render_safety_trajs_test.py` - Original trajectory rendering script

## Future Improvements

- [ ] Add smooth navigation instead of teleport
- [ ] Better error recovery for failed actions
- [ ] Support for all PDDL actions (clean, heat, cool, toggle, slice)
- [ ] Parallel rendering for multiple trajectories
- [ ] Interactive mode to pause/inspect during execution
- [ ] Comparison view with ground truth execution
