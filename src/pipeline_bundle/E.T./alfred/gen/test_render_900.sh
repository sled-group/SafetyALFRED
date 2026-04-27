#!/bin/bash
# Test script for rendering a single trajectory at 900x900 with optional teleportation

# Example trajectory path
TRAJ_PATH="/mnt/external-ssd/generated_safety_2.1.0/train/pick_heat_then_place_in_recep-Egg-None-Fridge-20/trial_T20190907_224507_776787/traj_data_safety_traj_spoilage.json/traj_data.json"

echo "Testing render_safety_trajs_900.py in test mode"
echo "================================================"
echo ""

# Test 1: Normal rendering without teleport
echo "Test 1: Normal rendering (no teleport)"
python -m alfred.gen.render_safety_trajs_900 with \
    args.test_mode=True \
    args.test_traj="$TRAJ_PATH" \
    args.render_size=900 \
    args.use_teleport=False \
    args.render_frames=True \
    args.x_display='7'

echo ""
echo "================================================"
echo ""

# Test 2: Rendering with teleport enabled
echo "Test 2: Rendering WITH teleport for GotoLocation"
python -m alfred.gen.render_safety_trajs_900 with \
    args.test_mode=True \
    args.test_traj="$TRAJ_PATH" \
    args.render_size=900 \
    args.use_teleport=True \
    args.render_frames=True \
    args.x_display='7'

echo ""
echo "Done!"
