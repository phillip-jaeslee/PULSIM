"""
tests/test_fileshape_resample.py -- regression test for FileShape's
`resample_to` parameter.

Context: sim_import_shaped_pulse (bloch_pulse_simulation.py) used to compute
dt = t_max / len(native_file_points), then reuse that same dt for every one
of the *resampled* target_N steps -- silently wrong whenever the file's
native point count differs from target_N. wave/sine.jhl has 1000 native
points; called with t_max=0.6 that gives target_N=600, so the old code
actually simulated 0.36 ms of duration instead of 0.6 ms.
FileShape(resample_to=...) fixes this by deriving dt from the *resampled*
point count instead.

The achieved flip angle turns out to be self-correcting regardless of dt
(the same dt appears in both the RF calibration and the rotation step, so it
cancels out), which is why the bug was invisible for on-resonance results.
What it actually breaks is off-resonance phase accumulation, so that's what
these tests check -- via pure off-resonance precession (no RF), where the
total accumulated phase is an exact, step-count-independent physics
invariant: 2*pi*phi*duration. That lets the "expected" side avoid depending
on bloch_rotate's general (off-axis) formula entirely -- see the comment
inside test_fileshape_resample_offresonance_precession_is_exact for why.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.bloch import bloch_rotate
from PULSIM.mat_operator import cpu_rot
from PULSIM.rf_shape import RFShape

HERE = os.path.dirname(os.path.abspath(__file__))
WAVE_FILE = os.path.join(os.path.dirname(HERE), "wave", "sine.jhl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(WAVE_FILE),
    reason="requires wave/sine.jhl, not distributed with the repository "
           "(.gitignore excludes wave/ -- see audit item 0-1)",
)

GAMMA = 42.58


def test_fileshape_resample_dt_matches_resampled_point_count():
    """shape.dt must be duration / (post-resample) points, not duration /
    native file points -- the two differ for wave/sine.jhl (1000 native
    points) resampled to 600."""
    duration = 0.6
    target_N = 600

    shape = RFShape.create("file", path=WAVE_FILE, duration=duration, resample_to=target_N)

    assert shape.points == target_N
    assert abs(shape.dt - duration / target_N) < 1e-12


def test_fileshape_resample_offresonance_precession_is_exact():
    """Pure off-resonance precession (no RF) over the full pulse duration
    must rotate M by exactly 2*pi*phi*duration about z, regardless of how
    many points the resampled shape has.

    This deliberately does NOT depend on bloch_rotate's general off-axis
    formula (still unverified per Task #34): with B = [0, 0, phi/Gamma],
    bloch_rotate's eta = arccos(Bz/|Bz|) = 0 and theta = atan2(0, 0) = 0 for
    every branch, so the conjugating rotations collapse to identity and it
    reduces to a plain Rz(flip) -- true regardless of branch or the
    unresolved general-B bug, since that bug only manifests for eta/theta
    != 0 (off-axis B). So this test safely exercises the real production
    code path (bloch_rotate stepped shape.points times at shape.dt) without
    depending on anything Task #34 hasn't fixed yet.
    """
    duration = 0.6
    phi = 2.0  # kHz offset

    for target_N in (300, 600, 1000):
        shape = RFShape.create("file", path=WAVE_FILE, duration=duration, resample_to=target_N)

        M = np.array([1.0, 0.0, 0.0])
        for _ in range(shape.points):
            B = [0.0, 0.0, phi / GAMMA]
            M = bloch_rotate(M, shape.dt, B, "x", GAMMA)

        theta = 2 * np.pi * phi * duration
        M_direct = cpu_rot.Rz(theta) @ np.array([1.0, 0.0, 0.0])

        assert np.max(np.abs(M - M_direct)) < 1e-6, (
            f"target_N={target_N}: stepped result {M} != closed-form {M_direct} "
            f"(theta={theta})"
        )


if __name__ == "__main__":
    test_fileshape_resample_dt_matches_resampled_point_count()
    test_fileshape_resample_offresonance_precession_is_exact()
    print("all tests passed")