# Plan Rendering with Navigation - Documentation

## Overview

The `render_plan_with_navigation.py` script extends `render_plan.py` to execute plans using proper smooth navigation instead of teleportation. It generates PDDL from trajectories, creates plans using Fast Downward, and renders the execution in THOR using low-level navigation actions (MoveAhead, RotateLeft, etc.).

## Key Difference from render_plan.py

**`render_plan.py`**: Uses `TeleportFull` for navigation - agent instantly appears at target location
**`render_plan_with_navigation.py`**: Uses navigation graph to decompose `GotoLocation` into sequences of low-level actions:
- `RotateLeft` / `RotateRight` - Turn to face direction
- `MoveAhead` - Move forward 0.25m
- `LookDown` / `LookUp` - Adjust camera horizon

## Navigation Implementation

The script uses ALFRED's navigation graph system (`Graph` from `alfred.gen.graph.graph_obj`) which:

1. **Builds a navigation graph** for the scene with 203 nodes representing reachable positions
2. **Uses A* pathfinding** to find the shortest path between poses
3. **Converts path to actions** using `get_shortest_path()` which returns low-level actions
4. **Handles failures gracefully** by falling back to teleport if navigation fails

### PDDL Location Format

Locations in PDDL are encoded as: `loc_bar__minus_14_bar_5_bar_3_bar_30`

Format breakdown:
- `loc_bar_` prefix
- Four values separated by `_bar_`:
  1. **x**: Grid x coordinate (e.g., `-14` = -14 * 0.25 = -3.5m)
  2. **y**: Grid y coordinate (e.g., `5` = 5 * 0.25 = 1.25m, represents z in THOR)
  3. **rotation_index**: 0-3 (0=0°, 1=90°, 2=180°, 3=270°)
  4. **horizon**: Camera pitch angle in degrees (e.g., `30` = 30°, `45` = 45°)

Example: `loc_bar__minus_14_bar_5_bar_3_bar_30`
- x = -14 grid units = -3.5m
- y = 5 grid units = 1.25m (z in THOR)
- rotation = 3 = 270°
- horizon = 30°

Maps to THOR action:
```python
{
    'action': 'TeleportFull',
    'x': -3.5,
    'y': 0.9009992,
    'z': 1.25,
    'rotation': 270,
    'horizon': 30
}
```

## Usage

```bash
cd /home/josue/Desktop/Research/SLED/MSS/E.T./alfred/gen
source ../../et_env_safety/bin/activate

export ALFRED_ROOT=/home/josue/Desktop/Research/SLED/MSS/E.T./alfred
export DISPLAY=:7

python render_plan_with_navigation.py \
  --traj_json /path/to/traj_data.json \
  --domain /path/to/domain.pddl \
  --output_dir /path/to/output \
  --x_display 7
```

## Example

```bash
python render_plan_with_navigation.py \
  --traj_json /mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/traj_data.json \
  --domain /home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/domain.pddl \
  --output_dir /tmp/plan_render_nav_test2 \
  --x_display 7
```

## Output Files

Same structure as `render_plan.py`:

```
output_dir/
├── problem.pddl              # Generated PDDL problem file
├── plan.txt                  # Human-readable plan (4 actions)
├── sas_plan                  # Fast Downward plan file
├── execution_log.json        # Detailed execution log
├── frames/                   # Directory with PNG frames
│   ├── 000000000.png        # Initial frame
│   ├── 000000001.png        # Frame after low-level action
│   ├── ...
│   └── 000000018.png        # Final frame
├── plan_execution.mp4        # Video of plan execution
└── debug.json                # Scene objects metadata (if errors occur)
```

## Execution Log Format

The execution log shows:
- High-level PDDL actions
- Number of low-level actions each decomposes into
- Success/failure status

Example:
```json
[
  {
    "step": 1,
    "pddl_action": "gotolocation agent1 loc_bar__minus_14_bar_5_bar_3_bar_30 loc_bar__minus_3_bar__minus_8_bar_0_bar_45",
    "num_low_level_actions": 36,
    "success": true
  },
  {
    "step": 2,
    "pddl_action": "pickupobjectinreceptacle1 ...",
    "num_low_level_actions": 1,
    "success": true
  }
]
```

## Example Output

### Console Output
```
================================================================================
PLAN RENDERING WITH NAVIGATION
================================================================================

[1/6] Generating PDDL from trajectory...
✓ Generated PDDL: /tmp/plan_render_nav_test2/problem.pddl

[2/6] Running Fast Downward planner...
✓ Plan generated: 4 actions in 0.62s
  1. gotolocation agent1 loc_bar__minus_14_bar_5_bar_3_bar_30 loc_bar__minus_3_bar__minus_8_bar_0_bar_45
  2. pickupobjectinreceptacle1 agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 peppershaker_... countertop_...
  3. gotolocation agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 loc_bar_0_bar__minus_6_bar_0_bar_45
  4. putobjectinreceptacle1 agent1 loc_bar_0_bar__minus_6_bar_0_bar_45 peppershakertype peppershaker_... drawer_...

[3/6] Initializing THOR environment...
✓ Environment initialized: FloorPlan10

[4/6] Building navigation graph...
✓ Navigation graph built with 203 nodes

[5/6] Executing plan in THOR...

Step 1/4: gotolocation agent1 loc_bar__minus_14_bar_5_bar_3_bar_30 loc_bar__minus_3_bar__minus_8_bar_0_bar_45
  Navigation: (-14, 5, 3, 30) -> (-3, -8, 0, 45)
  Path has 36 low-level actions
  Executing: RotateLeft (and 35 more actions)
  ✓ Step completed successfully

Step 2/4: pickupobjectinreceptacle1 agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 peppershaker_... countertop_...
  Executing: PickupObject (and 0 more actions)
  ✓ Step completed successfully

Step 3/4: gotolocation agent1 loc_bar__minus_3_bar__minus_8_bar_0_bar_45 loc_bar_0_bar__minus_6_bar_0_bar_45
  Navigation: (-3, -8, 0, 45) -> (0, -6, 0, 45)
  Path has 7 low-level actions
  Executing: RotateRight (and 6 more actions)
  ✓ Step completed successfully

Step 4/4: putobjectinreceptacle1 agent1 loc_bar_0_bar__minus_6_bar_0_bar_45 peppershakertype peppershaker_... drawer_...
  Executing: OpenObject (and 1 more actions)
  ✗ Action 2/2 failed: No valid Receptacle found
  ⚠ Step completed with errors

✓ Executed 4 high-level actions, saved 19 frames

[6/6] Creating video...
✓ Video saved: /tmp/plan_render_nav_test2/plan_execution.mp4

================================================================================
RENDERING COMPLETE
================================================================================
```

## Navigation Decomposition Example

A single high-level `GotoLocation` action decomposes into:

**High-level**: `gotolocation agent1 loc_bar__minus_14_bar_5_bar_3_bar_30 loc_bar__minus_3_bar__minus_8_bar_0_bar_45`

**Low-level** (36 actions):
1. RotateLeft (turn to face movement direction)
2. MoveAhead (0.25m forward)
3. MoveAhead (0.25m forward)
4. MoveAhead (0.25m forward)
...
34. MoveAhead (0.25m forward)
35. RotateRight (turn to face target rotation)
36. LookDown (adjust to target horizon)

## Frame Saving Strategy

To keep video size manageable while capturing all important moments:

- **Navigation actions**: Save every 3rd frame (e.g., frame 0, 3, 6, 9...)
- **Manipulation actions**: Save every frame (PickupObject, PutObject, OpenObject, CloseObject)
- **Teleport actions**: Save every frame (fallback case)

This balances video smoothness with file size.

## Supported PDDL Actions

Same as `render_plan.py`:

### Navigation
- `gotolocation` → Sequence of `MoveAhead`, `RotateLeft/Right`, `LookUp/Down`

### Object Manipulation
- `pickupobjectinreceptacle1` → `PickupObject`
- `pickupobjectnoreceptacle` → `PickupObject`
- `putobjectinreceptacle1` → `OpenObject` (if needed) + `PutObject`

### Receptacle Manipulation
- `openobject` → `OpenObject`
- `closeobject` → `CloseObject`

## Navigation Graph Details

The navigation graph (`Graph` class from `alfred.gen.graph.graph_obj`):

### Construction
- Uses ground truth scene layout from `constants.LAYOUTS_PATH`
- Builds directed graph with edges for all valid movements
- Each node represents (x, y, rotation_index) in grid coordinates
- Edges have weights based on traversability (1.0 for free space, higher for obstacles)

### Pathfinding
- Uses NetworkX A* algorithm with custom heuristic
- Heuristic: Manhattan distance + rotation difference
- Returns sequence of actions and poses along the path
- Automatically adjusts horizon at the end of navigation

### Error Handling
- If pathfinding fails, falls back to teleport
- If an action fails during execution, marks spot as impossible and replans
- Handles dynamic obstacles by updating graph weights

## Performance

- **PDDL Generation**: ~5-10 seconds
- **Planning**: 0.5-1.0 seconds for simple tasks
- **Navigation Graph Build**: ~0.1 seconds
- **Navigation Execution**: ~1 second per grid unit
- **Manipulation Execution**: ~0.1 seconds per action
- **Video Creation**: ~1 second

### Comparison with Teleport

| Metric | Teleport (`render_plan.py`) | Navigation (`render_plan_with_navigation.py`) |
|--------|----------------------------|----------------------------------------------|
| Frames saved | 5 frames | 19 frames |
| Execution time | ~2 seconds | ~10 seconds |
| Realism | Low (instant teleport) | High (smooth movement) |
| Actions logged | 4 PDDL actions | 4 PDDL + 43 low-level actions |

## Known Limitations

1. **Drawer/Cabinet Opening**: Some manipulation actions may fail due to precise positioning requirements
2. **Navigation Fallback**: If A* fails, script falls back to teleport
3. **Frame Sampling**: Navigation frames are sampled (every 3rd) to reduce video size
4. **Force Actions**: Uses `forceAction=True` which may not match exact physics

## Troubleshooting

### Navigation fails with empty exception

**Cause**: Pose format mismatch or invalid coordinates

**Fix**: Ensure PDDL location format is correct: `loc_bar_{x}_bar_{y}_bar_{rotation}_bar_{horizon}`

### "No valid Receptacle found"

Same as `render_plan.py` - receptacle opening or positioning issue

### Video shows teleports instead of navigation

**Cause**: Navigation failed and fell back to teleport

**Check**: Look for "Warning: Navigation failed" messages in console output

### Very slow execution

**Cause**: Long navigation paths with many low-level actions

**Fix**: Normal behavior - navigation is slower than teleport but more realistic

## Dependencies

- `alfred.env.thor_env` - THOR environment wrapper
- `alfred.gen.graph.graph_obj` - Navigation graph for pathfinding
- `alfred.gen.utils.video_util` - Video creation utilities
- `alfred.gen.utils.game_util` - Game state utilities (get_pose, etc.)
- `Fast Downward` - PDDL planner
- `NetworkX` - Graph algorithms for A* pathfinding
- `PIL` - Image processing
- `ffmpeg` - Video encoding

## Related Scripts

- `render_plan.py` - Original version using teleport
- `generate_problem_pddl_full.py` - Generates PDDL from trajectories
- `verify_pddl_with_planner.py` - Verifies PDDL by comparing plans
- `semantic_map_planner_agent.py` - Shows how agents use navigation graph

## Key Implementation Details

### How Navigation Works

1. **Parse PDDL location** to extract (x, y, rotation_index, horizon)
2. **Get current pose** using `game_util.get_pose(env.last_event)`
3. **Call `nav_graph.get_shortest_path(current_pose, target_pose)`** which:
   - Runs A* on the navigation graph
   - Returns list of low-level actions
4. **Execute each action** and save frames
5. **Handle failures** by marking impossible spots and replanning

### Code Structure

```python
def pddl_action_to_navigation_sequence(pddl_action, env, nav_graph, agent_loc_history):
    if action_name == 'gotolocation':
        # Parse PDDL location
        x, y, rotation_index, horizon = parse_location(target_loc)

        # Get current pose
        current_pose = game_util.get_pose(env.last_event)

        # Target pose
        target_pose = (x, y, rotation_index, horizon)

        # Get low-level actions from navigation graph
        try:
            actions, path = nav_graph.get_shortest_path(current_pose, target_pose)
            return actions
        except Exception as e:
            # Fallback to teleport
            return [{'action': 'TeleportFull', ...}]
```

## Future Improvements

- [ ] Add visualization of navigation path on top of video
- [ ] Support for more PDDL actions (clean, heat, cool, toggle, slice)
- [ ] Better error recovery for failed manipulation actions
- [ ] Comparison view showing ground truth vs generated plan
- [ ] Interactive mode to pause/step through execution
- [ ] Export navigation metrics (path length, time, collisions)
