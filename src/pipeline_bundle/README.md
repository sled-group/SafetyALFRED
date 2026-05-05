# Pipeline bundle

Self-contained copy of everything needed to run:

- `alfred/gen/pipeline_pddl_to_video_thor5.py` — full PDDL → plan → THOR rendering pipeline
- `alfred/gen/test_pipeline_safety_trajs.py` — batch driver that shells out to the pipeline for every safety trajectory it finds
- `alfred/gen/pipeline_llm_live_thor5.py` — live LLM-driven THOR rendering: queries a hosted vLLM server for the next action at every step (uses the same scene init as the PDDL pipeline)

All absolute paths in the runtime code have been rewritten to resolve relative to this directory, so the bundle can be moved/extracted anywhere.

## Layout

```
.
├── alfred/gen/                       # the scripts you call directly
│   ├── pipeline_pddl_to_video_thor5.py
│   ├── test_pipeline_safety_trajs.py
│   ├── pipeline_llm_live_thor5.py
│   └── convert_plan_to_traj.py
├── E.T./alfred/                      # framework imported as `alfred.*`
│   ├── env/                          # ThorEnv (Thor 5.0 wrapper)
│   └── gen/                          # constants, utils, graph, game_states,
│                                     # agents, planner, layouts, ff_planner,
│                                     # generate_problem_pddl_full_thor5.py,
│                                     # safety_initialization.py,
│                                     # render_plan_with_navigation.py
└── alfred_git/alfred/data/DANLI/pddl/
    ├── domain.pddl                   # PDDL domain
    ├── planner.py                    # planner wrapper
    └── fast-downward-24.06.1/        # precompiled Fast Downward
```

Total bundle size: ~550 MB (the planner binary alone is ~545 MB).

## Excluded for size

These were intentionally NOT copied:

- `E.T./alfred/gen/object_detection/` (~777 MB, not imported by these scripts)
- `E.T./alfred/gen/__pycache__/` and `*.pyc`
- `alfred_git/alfred/data/DANLI/pddl/fast-downward-24.06.1.tar.gz` (source archive — the extracted binary is included)
- `alfred_git/alfred/data/DANLI/pddl/sas_plan_temp/` (scratch outputs)

## What you still need on the destination machine

The bundle contains code only — not the Python environment.

1. **Python venv** — Python 3.7 with these packages (matching the original `et_env_safety_modern`):
   - `ai2thor==5.0.0`
   - `numpy`
   - `Pillow`
   - `termcolor`
   - `requests`
   - `scikit-video` (for `alfred.gen.utils.video_util`)
   - `opencv-python`

   Quickest path: copy the existing venv too, or `pip install ai2thor==5.0.0 numpy Pillow termcolor requests scikit-video opencv-python`.

2. **ffmpeg** on `$PATH` (used by `video_util.VideoSaver` to stitch frames into mp4).

3. **An X server / Xvfb** for THOR rendering. THOR will fail without one. The scripts default to `--x_display 7`; pass whatever display your X server uses.

4. **An ALFRED safety-trajectory dataset** — neither script ships data; both expect you to point them at trajectory JSONs you already have.

## Running the pipeline on a single trajectory

From `<bundle>/alfred/gen/`:

```bash
source /path/to/your/venv/bin/activate
cd <bundle>/alfred/gen

python pipeline_pddl_to_video_thor5.py \
    --traj_json /path/to/some/traj_data.json \
    --output_dir /tmp/pipeline_out \
    --x_display 7 \
    --use_teleport
```

Outputs land in `--output_dir`:

- `problem.pddl` — generated PDDL problem
- `plan.txt`, `sas_plan` — planner output
- `plan_execution/plan_execution.mp4` — initial render of the plan
- `converted_trajectory/traj_data.json` — ALFRED-format trajectory
- `final_render/video.mp4` — final smooth-nav render
- `execution_log.json`, `debug.txt` — audit trail

Useful flags (full list via `--help`):

- `--no_render_final` — skip the final smooth-nav render (much faster)
- `--use_teleport` — TeleportFull instead of step-by-step navigation
- `--clear_microwave_objects`, `--clear_sink_objects` — sanitize receptacles
- `--alternative_cabinet N`, `--alternative_object_location N` — variant runs

## Running the live LLM-driven pipeline

Make a conda environment with python 3.7 and run `pip install -r requirements.txt`

`pipeline_llm_live_thor5.py` ignores PDDL planning entirely. It uses the same scene-init code path (so it starts from a real ALFRED trajectory file) but then queries a hosted vLLM server (OpenAI-compatible chat API) for the next action every step until the goal is verified, the agent is stuck in a loop, or the step budget is exhausted. Each successful THOR step becomes one labeled frame in the output video.

Extra requirements over the PDDL pipeline:

- A running vLLM (or any OpenAI-compatible) server reachable from the host. Default URL: `http://localhost:8001/v1`.
- The model must accept image inputs (Qwen2.5-VL, Qwen3-VL, etc.).

Example:

```bash
python pipeline_llm_live_thor5.py \
    --traj_json /path/to/traj_data.json \
    --output_dir /tmp/llm_run \
    --x_display 7 \
    --vllm_url http://localhost:8001/v1 \
    --max_steps 40 \
    --seconds_per_frame 2
```

Outputs in `--output_dir`:

- `frames/*.png` — one PNG per successful THOR step (auto-teleports/auto-opens included), each with a `Predicted: ...` or `Auto: ...` overlay.
- `llm_run.mp4` — frames stitched at the requested seconds-per-frame.
- `llm_run.json` — per-turn audit trail (full prompt, raw response, parsed action, low-level THOR results, success/error).

Useful flags (full list via `--help`):

- `--strict_goto` — disable the auto-teleport-on-not-visible behavior. Non-`GoTo` verbs will fail unless the LLM has already navigated to the target.
- `--history_only_on_success` — only add successful actions to the running history list shown to the LLM.
- `--no_metadata` — vision-only prompt (omit the JSON metadata block).
- `--seconds_per_frame N` — how long each frame is held on screen in the video (default 2 s).

## Running the batch driver

From `<bundle>/alfred/gen/`:

```bash
python test_pipeline_safety_trajs.py --help
```

The batch driver walks a directory of safety trajectories and shells out to `pipeline_pddl_to_video_thor5.py` once per trajectory (it does this with `subprocess.run`, using its own working directory as the CWD — that's why both scripts must live side-by-side, which they do here).

## Path patches applied to the bundled copies

For reference, these are the only edits made to the source files when bundling:

- `alfred/gen/pipeline_pddl_to_video_thor5.py`
  - `et_gen_dir`, DANLI `planner.py` path, default `domain_path`, default `fd_path`, argparse `--domain` default → all resolved relative to the bundle root.
  - Added `sys.path.insert(0, <bundle>/E.T.)` so the `alfred.*` package resolves from the bundled copy.
- `E.T./alfred/gen/render_plan_with_navigation.py`
  - DANLI `planner.py` path and `fd_path` → resolved relative to the bundle root.

No logic was changed.

## Things not in the runtime path

`E.T./alfred/gen/` includes a number of unrelated utility scripts (`render_trajs_*.py`, `add_*_center*.py`, `verify_pddl_with_planner.py`, etc.) that contain stale absolute paths to the original workstation. They are NOT imported by the two pipelines included here, so they are harmless — but if you try to invoke any of them directly, expect to patch their paths first.
