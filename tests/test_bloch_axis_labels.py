"""
-x and -y in the Bloch simulator: the same four axis labels as the density-
matrix code (liouville._AXIS_PHASE), so one Pulse(axis="-x") runs in both.

Checked against what the labels must mean: "-x" is exactly the phase pi and
"-y" exactly -pi/2 (the numeric phases bloch_rotate already accepted), the
minus label rotates the opposite way to its plus label, and Pulse.apply gives
the density-matrix result for every label.
"""

import numpy as np
import pytest

import PULSIM
from PULSIM.bloch import bloch_rotate, bloch_rotate_batch
from PULSIM.liouville import LiouvilleSequence, ShapePulseSegment
from PULSIM.spin_operators import SpinOperators
from PULSIM.spin_system import SpinSystem, gyro_ratio

GAMMA = gyro_ratio("H")
RNG = np.random.default_rng(7)
M0 = RNG.normal(size=3)
B = np.array([3e-3, 1e-3, 2e-3])                 # mT, off-resonant field
BB = RNG.normal(size=(6, 3)) * 1e-3              # a batch of fields
MM = RNG.normal(size=(3, 6))


@pytest.mark.parametrize("label, phase", [("-x", np.pi), ("-y", -np.pi / 2),
                                          ("x", 0.0), ("y", np.pi / 2)])
def test_label_is_the_numeric_phase(label, phase):
    np.testing.assert_allclose(bloch_rotate(M0, 0.7, B, label, GAMMA),
                               bloch_rotate(M0, 0.7, B, phase, GAMMA), atol=1e-14)
    np.testing.assert_allclose(bloch_rotate_batch(MM, 0.7, BB, label, GAMMA),
                               bloch_rotate_batch(MM, 0.7, BB, phase, GAMMA), atol=1e-14)


@pytest.mark.parametrize("plus, minus", [("x", "-x"), ("y", "-y")])
def test_minus_label_turns_the_other_way(plus, minus):
    """On resonance a 90 about -x is the inverse of a 90 about +x."""
    b = np.array([1e-3, 0.0, 0.0])
    t = 0.25 / (GAMMA * 1e-3)                    # flip = 2 pi gamma |B| t = pi/2
    there = bloch_rotate(M0, t, b, plus, GAMMA)
    np.testing.assert_allclose(bloch_rotate(there, t, b, minus, GAMMA), M0, atol=1e-12)


def test_unknown_label_still_refused():
    with pytest.raises(NotImplementedError, match="-x"):
        bloch_rotate(M0, 0.1, B, "z", GAMMA)


@pytest.mark.parametrize("axis", ["x", "y", "-x", "-y"])
def test_pulse_apply_matches_the_density_matrix(axis):
    def pulse():
        return PULSIM.Pulse(PULSIM.RFShape.create("hermite", duration=1.0, points=200),
                            np.pi / 2, axis=axis, backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    M = np.array([[0.0], [0.0], [1.0]])
    bloch = pulse().apply(M, np.array([0.0]))[:, 0]
    ss = SpinSystem(nuclei=["H"], offsets=[0.0])
    ops = SpinOperators(ss)
    sigma = LiouvilleSequence([ShapePulseSegment(pulse())], ss).propagate(ops["Iz"])
    dens = np.array([ops.expectation(sigma, "I" + a) for a in "xyz"])
    np.testing.assert_allclose(bloch, dens, atol=1e-10)