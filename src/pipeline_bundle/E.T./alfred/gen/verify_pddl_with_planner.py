#!/usr/bin/env python3
"""
Verification script to compare generated PDDL with ground truth by:
1. Generating PDDL from trajectory
2. Running Fast Downward planner on both generated and ground truth PDDL
3. Comparing the resulting plans

Usage:
    python verify_pddl_with_planner.py --traj_json <path> --ground_truth_pddl <path> --domain <path>
"""

import os
import sys
import json
import argparse
import tempfile
import re

# Add ALFRED paths
sys.path.append(os.path.join(os.environ.get('ALFRED_ROOT', '.'), 'gen'))

from generate_problem_pddl_full import generate_pddl_from_traj_full

# Import DANLI planner directly to avoid conflicts
danli_planner_path = '/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/planner.py'
import importlib.util
spec = importlib.util.spec_from_file_location("danli_planner", danli_planner_path)
danli_planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(danli_planner)
PDDLPlanner = danli_planner.PDDLPlanner


def fix_ground_truth_pddl(pddl_string):
    """
    Fix ground truth PDDL to match expected format:
    1. Change (:metric minimize (totalCost)) to (:metric minimize (total-cost))
    2. Move metric after goal

    Args:
        pddl_string: Original PDDL string

    Returns:
        str: Fixed PDDL string
    """
    # Replace totalCost with total-cost
    pddl_string = pddl_string.replace('(totalCost)', '(total-cost)')

    # Find and remove the metric line from its current position
    metric_pattern = r'\s*\(:metric[^\n]+\)\s*\n'
    pddl_string = re.sub(metric_pattern, '\n', pddl_string, count=1)

    # Find the goal section and insert metric after it
    # Look for the pattern: "        )\n" followed by "    )" (end of goal, then end of problem)
    goal_end_pattern = r'(        \)\n)(    \))'

    replacement = r'\1        (:metric minimize (total-cost))\n\2'
    pddl_string = re.sub(goal_end_pattern, replacement, pddl_string)

    return pddl_string


def compare_plans(plan1, plan2):
    """
    Compare two plans and return similarity metrics

    Args:
        plan1: List of action tuples from first plan
        plan2: List of action tuples from second plan

    Returns:
        dict: Comparison metrics
    """
    if plan1 is None or plan2 is None:
        return {
            'same_length': False,
            'same_actions': False,
            'length1': len(plan1) if plan1 else 0,
            'length2': len(plan2) if plan2 else 0,
            'overlap_pct': 0.0,
            'identical': False
        }

    # Normalize action tuples for comparison
    def normalize_action(action_tuple):
        # Convert all elements to uppercase and strip whitespace
        return tuple(str(x).upper().strip() for x in action_tuple)

    plan1_norm = [normalize_action(a) for a in plan1]
    plan2_norm = [normalize_action(a) for a in plan2]

    # Calculate metrics
    same_length = len(plan1) == len(plan2)
    same_actions = plan1_norm == plan2_norm

    # Calculate action overlap
    if len(plan1) == 0 and len(plan2) == 0:
        overlap_pct = 100.0
    elif len(plan1) == 0 or len(plan2) == 0:
        overlap_pct = 0.0
    else:
        # Count matching actions at each position
        matches = sum(1 for a1, a2 in zip(plan1_norm, plan2_norm) if a1 == a2)
        max_len = max(len(plan1), len(plan2))
        overlap_pct = (matches / max_len) * 100.0

    return {
        'same_length': same_length,
        'same_actions': same_actions,
        'length1': len(plan1),
        'length2': len(plan2),
        'overlap_pct': overlap_pct,
        'identical': same_actions and same_length
    }


def main():
    parser = argparse.ArgumentParser(
        description='Verify generated PDDL matches ground truth by comparing plans')
    parser.add_argument('--traj_json', type=str, required=True,
                       help='Path to traj_data.json file')
    parser.add_argument('--ground_truth_pddl', type=str, required=True,
                       help='Path to ground truth problem.pddl file')
    parser.add_argument('--domain', type=str, required=True,
                       help='Path to domain.pddl file')
    parser.add_argument('--x_display', type=str, default='7',
                       help='X server display number')
    parser.add_argument('--output_dir', type=str, default='/tmp',
                       help='Directory to save generated files')
    parser.add_argument('--fd_path', type=str,
                       default='/home/josue/Desktop/Research/SLED/MSS/alfred_git/alfred/data/DANLI/pddl/fast-downward-24.06.1/fast-downward.py',
                       help='Path to Fast Downward fast-downward.py')

    args = parser.parse_args()

    print("=" * 80)
    print("PDDL Generation and Plan Verification")
    print("=" * 80)

    # Generate PDDL from trajectory
    print("\n[1/5] Generating PDDL from trajectory...")
    generated_pddl_path = os.path.join(args.output_dir, 'problem_generated_verified.pddl')

    try:
        pddl_string = generate_pddl_from_traj_full(
            args.traj_json,
            generated_pddl_path,
            args.x_display
        )
        print(f"✓ Generated PDDL saved to: {generated_pddl_path}")
        print(f"  Lines: {len(pddl_string.split(chr(10)))}")
    except Exception as e:
        print(f"✗ Failed to generate PDDL: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load and fix ground truth PDDL
    print(f"\n[2/5] Loading and fixing ground truth PDDL...")
    with open(args.ground_truth_pddl, 'r') as f:
        ground_truth_pddl = f.read()

    print(f"✓ Ground truth PDDL loaded from: {args.ground_truth_pddl}")
    print(f"  Original lines: {len(ground_truth_pddl.split(chr(10)))}")

    # Fix the ground truth PDDL format
    fixed_ground_truth_pddl = fix_ground_truth_pddl(ground_truth_pddl)
    fixed_gt_path = os.path.join(args.output_dir, 'problem_ground_truth_fixed.pddl')

    with open(fixed_gt_path, 'w') as f:
        f.write(fixed_ground_truth_pddl)

    print(f"✓ Fixed ground truth PDDL saved to: {fixed_gt_path}")
    print(f"  Fixed lines: {len(fixed_ground_truth_pddl.split(chr(10)))}")

    # Initialize planner
    print(f"\n[3/5] Initializing Fast Downward planner...")
    try:
        planner = PDDLPlanner(fd_path=args.fd_path, alias='max-astar', timeout=60)
        print(f"✓ Planner initialized")
    except Exception as e:
        print(f"✗ Failed to initialize planner: {e}")
        return 1

    # Run planner on generated PDDL
    print(f"\n[4/5] Running planner on generated PDDL...")
    gen_plan_file = os.path.join(args.output_dir, 'sas_plan_generated')
    planner.plan_file = gen_plan_file

    try:
        gen_plan, gen_runtime = planner.plan(args.domain, generated_pddl_path, debug=False)

        if gen_plan is not None:
            print(f"✓ Plan generated successfully")
            print(f"  Plan length: {len(gen_plan)} actions")
            print(f"  Runtime: {gen_runtime:.2f}s")
        else:
            print(f"✗ Failed to generate plan from generated PDDL")
            print(f"  Runtime: {gen_runtime:.2f}s")
            print(f"  No solution found or planning failed")
    except Exception as e:
        print(f"✗ Error running planner on generated PDDL: {e}")
        gen_plan = None
        gen_runtime = 0

    # Run planner on ground truth PDDL
    print(f"\n[5/5] Running planner on ground truth PDDL...")
    gt_plan_file = os.path.join(args.output_dir, 'sas_plan_ground_truth')
    planner.plan_file = gt_plan_file

    try:
        gt_plan, gt_runtime = planner.plan(args.domain, fixed_gt_path, debug=False)

        if gt_plan is not None:
            print(f"✓ Plan generated successfully")
            print(f"  Plan length: {len(gt_plan)} actions")
            print(f"  Runtime: {gt_runtime:.2f}s")
        else:
            print(f"✗ Failed to generate plan from ground truth PDDL")
            print(f"  Runtime: {gt_runtime:.2f}s")
            print(f"  No solution found or planning failed")
    except Exception as e:
        print(f"✗ Error running planner on ground truth PDDL: {e}")
        gt_plan = None
        gt_runtime = 0

    # Compare plans
    print("\n" + "=" * 80)
    print("PLAN COMPARISON")
    print("=" * 80)

    comparison = compare_plans(gen_plan, gt_plan)

    print(f"\nGenerated plan length: {comparison['length1']}")
    print(f"Ground truth plan length: {comparison['length2']}")
    print(f"Plan overlap: {comparison['overlap_pct']:.1f}%")

    if comparison['identical']:
        print("\n" + "✓" * 40)
        print("✓✓✓ PLANS ARE IDENTICAL ✓✓✓")
        print("Generated PDDL produces the same plan as ground truth!")
        print("✓" * 40)
        result = 0
    elif gen_plan is None and gt_plan is None:
        print("\n✗ Both plans failed to generate")
        print("This may indicate an issue with the domain file or planner")
        result = 1
    elif gen_plan is None:
        print("\n✗ Generated PDDL does not produce a valid plan")
        print("This indicates the generated PDDL is missing required predicates or has errors")
        result = 1
    elif gt_plan is None:
        print("\n✗ Ground truth PDDL does not produce a valid plan")
        print("This may indicate an issue with the domain file or ground truth PDDL")
        result = 1
    elif comparison['same_length']:
        print("\n⚠ Plans have same length but different actions")
        print("This may indicate equivalent but different solutions")
        result = 2
    else:
        print("\n⚠ Plans differ in length and/or actions")
        print("Generated PDDL may be missing predicates or have incorrect state")
        result = 2

    # Show first 10 actions of each plan
    if gen_plan:
        print(f"\nFirst {min(10, len(gen_plan))} actions from generated plan:")
        for i, action in enumerate(gen_plan[:10], 1):
            print(f"  {i}. {' '.join(action)}")

    if gt_plan:
        print(f"\nFirst {min(10, len(gt_plan))} actions from ground truth plan:")
        for i, action in enumerate(gt_plan[:10], 1):
            print(f"  {i}. {' '.join(action)}")

    # Show differences
    if not comparison['identical'] and comparison['same_length'] and gen_plan and gt_plan:
        print("\nDifferences found at these steps:")
        diff_count = 0
        for i, (a1, a2) in enumerate(zip(gen_plan, gt_plan), 1):
            if a1 != a2:
                diff_count += 1
                if diff_count <= 10:  # Show first 10 differences
                    print(f"  Step {i}:")
                    print(f"    Generated:    {' '.join(a1)}")
                    print(f"    Ground truth: {' '.join(a2)}")

        if diff_count > 10:
            print(f"  ... and {diff_count - 10} more differences")

    print("\n" + "=" * 80)
    return result


if __name__ == '__main__':
    sys.exit(main())
