# How to Use the PDDL Generator

This guide shows you how to generate problem.pddl files from ALFRED trajectory data using the full ALFRED infrastructure.

## Quick Start

```bash
# 1. Navigate to the gen directory
cd /home/josue/Desktop/Research/SLED/MSS/E.T./alfred/gen

# 2. Activate the virtual environment
source ../../et_env_safety/bin/activate

# 3. Set environment variables
export ALFRED_ROOT=/home/josue/Desktop/Research/SLED/MSS/E.T./alfred
export DISPLAY=:7

# 4. Run the generator
python generate_problem_pddl_full.py \
  --traj_json /mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/traj_data.json \
  --output /tmp/my_problem.pddl \
  --x_display 7
```

## Step-by-Step Setup

### 1. Environment Setup

First, make sure you're in the correct directory and activate the virtual environment:

```bash
cd /home/josue/Desktop/Research/SLED/MSS/E.T./alfred/gen
source ../../et_env_safety/bin/activate
```

### 2. Set Required Environment Variables

```bash
# Point to ALFRED root directory
export ALFRED_ROOT=/home/josue/Desktop/Research/SLED/MSS/E.T./alfred

# Set X server display (for headless THOR rendering)
export DISPLAY=:7
```

**Note**: The X server display number (`:7`) must match the `--x_display` argument you pass to the script.

### 3. Verify X Server is Running

```bash
# Check if X server is running on display 7
ps aux | grep "X :7"
```

If not running, start it:

```bash
sudo Xorg :7 &
# OR if using Xvfb:
Xvfb :7 -screen 0 1024x768x24 &
```

## Usage Examples

### Example 1: Generate PDDL from a Single Trajectory

```bash
python generate_problem_pddl_full.py \
  --traj_json /path/to/traj_data.json \
  --output /path/to/output.pddl \
  --x_display 7
```

**Parameters**:
- `--traj_json`: Path to the trajectory JSON file (required)
- `--output`: Where to save the generated PDDL file (optional, defaults to `<traj_dir>/problem_generated_full.pddl`)
- `--x_display`: X server display number (default: 7)

### Example 2: Generate with Default Output Location

If you don't specify `--output`, the PDDL will be saved in the same directory as the trajectory file:

```bash
python generate_problem_pddl_full.py \
  --traj_json /mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/traj_data.json \
  --x_display 7

# Output will be saved to:
# /mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/problem_generated_full.pddl
```

### Example 3: Using a Different X Display

If your X server is on a different display (e.g., `:0` or `:1`):

```bash
export DISPLAY=:0
python generate_problem_pddl_full.py \
  --traj_json /path/to/traj_data.json \
  --x_display 0
```

### Example 4: Batch Processing Multiple Trajectories

```bash
#!/bin/bash
# Process all trajectories in a directory

TRAJ_DIR="/mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen"

# Find all traj_data.json files and process them
find "$TRAJ_DIR" -name "traj_data.json" | while read traj_file; do
    echo "Processing: $traj_file"
    python generate_problem_pddl_full.py \
        --traj_json "$traj_file" \
        --x_display 7

    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated PDDL for: $traj_file"
    else
        echo "✗ Failed to generate PDDL for: $traj_file"
    fi
done
```

## Using the Script in Python

You can also import and use the generator function directly in your Python code:

```python
import os
import sys

# Add ALFRED paths
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

from generate_problem_pddl_full import generate_pddl_from_traj_full

# Generate PDDL
traj_path = '/path/to/traj_data.json'
output_path = '/path/to/output.pddl'

pddl_string = generate_pddl_from_traj_full(
    traj_json_path=traj_path,
    output_pddl_path=output_path,
    x_display='7'
)

print(f"Generated {len(pddl_string.split(chr(10)))} lines of PDDL")
```

## Required Input Format

The script expects a `traj_data.json` file with the following structure:

```json
{
  "task_type": "pick_and_place_simple",
  "task_id": "trial_T20190906_184021_215264",
  "scene": {
    "scene_num": 10,
    "random_seed": 123456,
    "object_poses": [...],
    "object_toggles": [...],
    "dirty_and_empty": [...],
    "init_action": {...},
    "toggle_object": null
  },
  "pddl_params": {
    "object_target": "PepperShaker",
    "parent_target": "Drawer",
    "toggle_target": "",
    "mrecep_target": "",
    "object_sliced": false
  },
  ...
}
```

**Required fields**:
- `scene.scene_num`: FloorPlan number (e.g., 10 for FloorPlan10)
- `scene.object_poses`: Initial object positions
- `scene.object_toggles`: Which objects should be toggled on/off
- `scene.dirty_and_empty`: Object states (dirty, empty)
- `scene.init_action`: Initial action to execute
- `pddl_params.object_target`: The object to manipulate (e.g., "PepperShaker")
- `pddl_params.parent_target`: Target receptacle or toggle object
- `task_type`: Type of task (e.g., "pick_and_place_simple")

## Output Format

The generated PDDL file will have the following structure:

```pddl
(define (problem plan_XXXXX)
    (:domain put_task)
    (:metric minimize (totalCost))
    (:objects
        agent1 - agent
        [Object type declarations]
        [Object instances with coordinates]
        [Receptacle instances]
        [Location grid points]
    )
    (:init
        (= (totalCost) 0)
        [684 canContain predicates]
        [Object and receptacle types]
        [Object properties (cleanable, heatable, etc.)]
        [Initial object states]
        [Containment relationships]
        [Distance calculations]
        [Location mappings]
    )
    (:goal
        [Task-specific goal condition]
    )
)
```

**Key components**:
- **2000+ lines** of complete scene state
- **684 canContain predicates** (domain knowledge)
- **Full navigation graph** with distance calculations
- **All object instances** with precise coordinates
- **Exact goal specification** matching ALFRED format

## Troubleshooting

### Error: "Cannot connect to X server"

**Cause**: X server not running or wrong display number

**Solution**:
```bash
# Check if X server is running
ps aux | grep "X :7"

# Start X server if needed
Xvfb :7 -screen 0 1024x768x24 &

# Make sure DISPLAY and --x_display match
export DISPLAY=:7
```

### Error: "No module named 'alfred'"

**Cause**: ALFRED_ROOT not set or Python path incorrect

**Solution**:
```bash
# Set ALFRED_ROOT
export ALFRED_ROOT=/home/josue/Desktop/Research/SLED/MSS/E.T./alfred

# Verify it's correct
ls $ALFRED_ROOT/gen/  # Should show alfred module
```

### Error: "KeyError: 'template'"

**Cause**: Old version of script or corrupt trajectory file

**Solution**:
- Make sure you're using `generate_problem_pddl_full.py` (not `generate_problem_pddl.py`)
- Verify your traj_data.json has all required fields
- Re-download the script if needed

### Error: "AI2-THOR binary not found"

**Cause**: THOR binary missing or not executable

**Solution**:
```bash
# Check if binary exists
ls -la $ALFRED_ROOT/../alfred-assets/  # or wherever THOR is installed

# Re-download if needed (this happens automatically on first run)
# Just run the script again, it will download THOR
```

### Warning: "Generated PDDL has different number of lines"

**This is normal!** The generated PDDL may have more or fewer lines than ground truth due to:
- More complete canContain predicates (684 vs 494)
- Different object ordering (Python dict iteration)
- Different random seed (different problem ID)

**What matters**:
- ✓ Goal section matches
- ✓ All objects present
- ✓ Navigation graph complete
- ✓ Valid PDDL syntax

## Verifying Output

### 1. Check File Was Created

```bash
ls -lh /path/to/output.pddl
```

### 2. Check Line Count

```bash
wc -l /path/to/output.pddl
# Should be around 2000+ lines
```

### 3. Check PDDL Structure

```bash
# Check header
head -20 /path/to/output.pddl

# Check goal section
tail -20 /path/to/output.pddl

# Count canContain predicates
grep -c "canContain" /path/to/output.pddl
# Should be 684
```

### 4. Validate with PDDL Parser

```bash
# If you have a PDDL validator installed
validate-pddl /path/to/domain.pddl /path/to/output.pddl
```

### 5. Compare with Ground Truth (Optional)

```bash
# Visual diff
diff /path/to/ground_truth/problem_0.pddl /path/to/output.pddl | less

# Check goal matches exactly
diff <(sed -n '/(:goal/,/^)/p' ground_truth.pddl) \
     <(sed -n '/(:goal/,/^)/p' output.pddl)
```

## Advanced Usage

### Custom Scene Initialization

If you want to modify the scene before generating PDDL:

```python
from generate_problem_pddl_full import generate_pddl_from_traj_full
import json

# Load and modify trajectory
with open('traj_data.json', 'r') as f:
    traj = json.load(f)

# Modify scene parameters
traj['scene']['random_seed'] = 999999  # Different seed
traj['pddl_params']['object_target'] = 'Apple'  # Different object

# Save modified trajectory
with open('/tmp/modified_traj.json', 'w') as f:
    json.dump(traj, f)

# Generate PDDL from modified trajectory
pddl = generate_pddl_from_traj_full('/tmp/modified_traj.json', '/tmp/modified.pddl')
```

### Batch Processing with Error Handling

```python
import os
import glob
from generate_problem_pddl_full import generate_pddl_from_traj_full

# Find all trajectories
traj_files = glob.glob('/mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/**/traj_data.json', recursive=True)

results = {'success': [], 'failed': []}

for traj_file in traj_files:
    try:
        output_file = os.path.join(os.path.dirname(traj_file), 'problem_generated.pddl')
        generate_pddl_from_traj_full(traj_file, output_file, x_display='7')
        results['success'].append(traj_file)
        print(f"✓ {traj_file}")
    except Exception as e:
        results['failed'].append((traj_file, str(e)))
        print(f"✗ {traj_file}: {e}")

# Summary
print(f"\nProcessed {len(traj_files)} trajectories:")
print(f"  Success: {len(results['success'])}")
print(f"  Failed: {len(results['failed'])}")

if results['failed']:
    print("\nFailed trajectories:")
    for traj, error in results['failed']:
        print(f"  - {traj}")
        print(f"    Error: {error}")
```

## Summary

To generate a PDDL file from a trajectory:

1. ✓ Activate virtual environment: `source ../../et_env_safety/bin/activate`
2. ✓ Set environment variables: `export ALFRED_ROOT=...` and `export DISPLAY=:7`
3. ✓ Make sure X server is running on display :7
4. ✓ Run: `python generate_problem_pddl_full.py --traj_json <path> --x_display 7`
5. ✓ Verify output: Check file exists, ~2000+ lines, goal section present

The generated PDDL files are fully compatible with the ALFRED dataset format and can be used with the FF planner for action sequence generation.

## Additional Resources

- **Main Documentation**: See `PDDL_GENERATION_FINAL_REPORT.md` for verification details
- **API Reference**: See `PDDL_GENERATION_README.md` for technical details
- **ALFRED Repository**: https://github.com/askforalfred/alfred
- **AI2-THOR**: https://ai2thor.allenai.org/

## Getting Help

If you encounter issues not covered in this guide:

1. Check the PDDL_GENERATION_FINAL_REPORT.md for known differences with ground truth
2. Verify all prerequisites are installed (see ALFRED README.md)
3. Check the ALFRED GitHub issues: https://github.com/askforalfred/alfred/issues
4. Ensure you're using Python 3.6+ and have all dependencies from requirements.txt
