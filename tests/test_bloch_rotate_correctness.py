"""
Correctness test for bloch_rotate — NOT a golden-file test.

The golden tests in test_rf_shape_golden.py check "does the new code match
the old code's output". That's the wrong tool here: the old code had a bug,
so matching it would just freeze the bug in place.

This test instead checks bloch_rotate against Rodrigues' rotation formula,
a standard, independently-derived way to rotate a vector around an axis.
It does not go through the align/flip/unalign trick that bloch_rotate uses,
so it can't share the same mistake.

Ground truth is rotation by +flip around the B axis, matching the Ernst/
Levitt RF-pulse convention bloch_rotate deliberately follows (see
test_bloch_rotate_levitt_convention.py) -- not -flip, which is what you'd
get from directly integrating the lab-frame Bloch equation instead. Verified
directly: worst-case diff over 5000 random cases is 7.8e-16 with +flip vs.
~2.0 with -flip.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.bloch import bloch_rotate

def rodrigues_rotation(v, axis, angle):
    """Rotate vector v around `axis` by `angle` radians. Independent ground truth."""
    k = axis / np.linalg.norm(axis)
    return (v * np.cos(angle)
            + np.cross(k, v) * np.sin(angle)
            + k * np.dot(k, v) * (1 - np.cos(angle)))

def test_bloch_rotate_matches_rodrigues():
    Gamma = 42.58
    rng = np.random.default_rng(0)
    worst_error = 0.0

    for _ in range(5000):
        B = rng.normal(size=3)
        if np.linalg.norm(B) < 1e-6:
            continue
        T = rng.uniform(1e-4, 1e-2)
        M = rng.normal(size=3)
        M = M / np.linalg.norm(M)

        flip = 2 * np.pi * Gamma * np.linalg.norm(B) * T
        expected =rodrigues_rotation(M, B, flip)
        actual = bloch_rotate(M, T, B, "x", Gamma)

        worst_error = max(worst_error, np.abs(actual - expected).max())

    print(f"worst over over 5000 random cases: {worst_error:.3e}")
    assert worst_error < 1e-9, "bloch_rotate no longer matches independent rotation math"

if __name__ == "__main__":
    test_bloch_rotate_matches_rodrigues()
    print("PASS")