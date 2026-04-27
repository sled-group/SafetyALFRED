#!/usr/bin/env python3
"""
Test script to verify PDDL generation from trajectory data.
"""

import os
import sys

# Test with the reference trajectory
TRAJ_PATH = "/mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/traj_data.json"
OUTPUT_PATH = "/tmp/test_problem.pddl"
REFERENCE_PATH = "/mnt/external-ssd/alfred/data/full_2.1.0/valid_unseen/pick_and_place_simple-PepperShaker-None-Drawer-10/trial_T20190906_184021_215264/problem_0.pddl"

def main():
    print("=" * 80)
    print("Testing PDDL Generation")
    print("=" * 80)

    # Import the generator
    from generate_problem_pddl import generate_pddl_from_traj

    # Generate PDDL
    print(f"\nGenerating PDDL from: {TRAJ_PATH}")
    print(f"Output to: {OUTPUT_PATH}")

    try:
        pddl_string = generate_pddl_from_traj(
            TRAJ_PATH,
            OUTPUT_PATH,
            x_display='0'
        )

        print("\n" + "=" * 80)
        print("Generated PDDL (first 100 lines):")
        print("=" * 80)
        for i, line in enumerate(pddl_string.split('\n')[:100]):
            print(f"{i+1:3d}: {line}")

        # Compare with reference if available
        if os.path.exists(REFERENCE_PATH):
            print("\n" + "=" * 80)
            print("Comparing with reference PDDL...")
            print("=" * 80)

            with open(REFERENCE_PATH, 'r') as f:
                reference_pddl = f.read()

            # Compare key sections
            gen_lines = pddl_string.split('\n')
            ref_lines = reference_pddl.split('\n')

            print(f"\nGenerated lines: {len(gen_lines)}")
            print(f"Reference lines: {len(ref_lines)}")

            # Show differences in first 50 lines
            print("\nFirst 50 lines comparison:")
            max_lines = min(50, len(gen_lines), len(ref_lines))
            differences = 0
            for i in range(max_lines):
                if i < len(gen_lines) and i < len(ref_lines):
                    if gen_lines[i].strip() != ref_lines[i].strip():
                        differences += 1
                        print(f"\nLine {i+1} DIFFERS:")
                        print(f"  Generated: {gen_lines[i]}")
                        print(f"  Reference: {ref_lines[i]}")

            print(f"\n{differences} differences found in first {max_lines} lines")

        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
