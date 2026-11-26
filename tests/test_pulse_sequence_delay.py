"""
Delays in the Bloch simulator: PulseSequence accepts the same liouville.Delay
the density-matrix simulator uses.

Checked against independent ground truth -- the density-matrix simulator,
which propagates a Delay through its own Hamiltonian -- and against what a
delay must do: free precession at each offset, a spin echo that refocuses
every offset, relaxation from the pulses' backend, and the bookkeeping
visualization.py relies on (one frame per RF step or delay step, plus the
start; a continuous time axis; zero RF during a delay).
"""

import numpy as np
import pytest

import PULSIM
from PULSIM.liouville import Delay, LiouvilleSequence, ShapePulseSegment
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.spin_operators import SpinOperators
from PULSIM.spin_system import SpinSystem, gyro_ratio

GAMMA = gyro_ratio("H")
DF = np.linspace(-2.0, 2.0, 9)                       # kHz


def _pulse(name, duration, points, flip_deg, axis="x", backend=None):
    shape = PULSIM.RFShape.create(name, duration=duration, points=points)
    return PULSIM.Pulse(shape, np.deg2rad(flip_deg), axis=axis,
                        backend=backend or PULSIM.NumpyBackend(Gamma=GAMMA))


def _z():
    M = np.zeros((3, DF.size))
    M[2] = 1.0
    return M


def _density(elements):
    """The same sequence in the density-matrix simulator, offset by offset."""
    out = np.empty((3, DF.size))
    for k, f in enumerate(DF):
        ss = SpinSystem(nuclei=["H"], offsets=[2 * np.pi * f])          # rad/ms
        ops = SpinOperators(ss)
        segs = [e if isinstance(e, Delay) else ShapePulseSegment(e) for e in elements]
        sigma = LiouvilleSequence(segs, ss).propagate(ops["Iz"])
        out[:, k] = [ops.expectation(sigma, "I" + a) for a in "xyz"]
    return out


def test_pulse_delay_pulse_matches_the_density_matrix():
    elements = [_pulse("hard", 0.02, 10, 90), Delay(0.37), _pulse("hermite", 0.5, 100, 90, "y")]
    bloch = PulseSequence(elements).run(_z(), DF)
    np.testing.assert_allclose(bloch, _density(elements), atol=1e-10)


def test_a_delay_is_free_precession():
    """From x, a delay T turns each offset by 2 pi df T about z, length kept."""
    M = np.zeros((3, DF.size))
    M[0] = 1.0
    T = 0.13
    out = PulseSequence([Delay(T)]).run(M, DF)
    angle = np.arctan2(out[1], out[0])
    np.testing.assert_allclose(np.abs(np.angle(np.exp(1j * (np.abs(angle) - 2 * np.pi * np.abs(DF) * T)))),
                               0.0, atol=1e-12)
    np.testing.assert_allclose(np.hypot(out[0], out[1]), 1.0, atol=1e-12)
    np.testing.assert_allclose(out[2], 0.0, atol=1e-12)


def test_spin_echo_refocuses_every_offset():
    """90x - tau - 180y - tau with hard pulses: every offset ends up together.
    The pulses last 1 us, so offsets up to 2 kHz still precess ~0.01 rad during
    them -- the tolerance; without refocusing the spread would be the full circle."""
    tau = 0.8
    seq = PulseSequence([_pulse("hard", 0.001, 4, 90), Delay(tau),
                         _pulse("hard", 0.001, 4, 180, "y"), Delay(tau)])
    out = seq.run(_z(), DF)
    np.testing.assert_allclose(out, np.repeat(out[:, [DF.size // 2]], DF.size, axis=1), atol=1e-2)


def test_include_offset_false_does_not_precess():
    M = np.zeros((3, DF.size))
    M[0] = 1.0
    np.testing.assert_allclose(PulseSequence([Delay(0.3, include_offset=False)]).run(M, DF),
                               M, atol=1e-14)


def test_a_delay_relaxes_like_the_pulses_backend():
    backend = PULSIM.NumpyBackend(Gamma=GAMMA, T1=50.0, T2=2.0)
    seq = PulseSequence([_pulse("hard", 0.001, 4, 90, backend=backend), Delay(1.0)])
    out = seq.run(_z(), np.array([0.0]))
    assert np.hypot(out[0, 0], out[1, 0]) == pytest.approx(np.exp(-1.0 / 2.0), rel=1e-3)


def test_trajectory_ends_where_run_ends_and_bookkeeping_holds():
    seq = PulseSequence([_pulse("hard", 0.02, 10, 90), Delay(0.5), _pulse("hard", 0.02, 10, 90)],
                        delay_steps=25)
    traj = seq.run(_z(), DF, trajectory=True)
    assert traj.shape == (len(seq.rf) + 1, 3, DF.size)
    assert len(seq.time) == len(seq.rf) == len(seq.phase) == 10 + 25 + 10
    np.testing.assert_allclose(traj[-1], seq.run(_z(), DF), atol=1e-12)
    np.testing.assert_array_equal(seq.rf[10:35], 0)            # no RF during the delay
    assert np.all(np.isnan(seq.phase[10:35]))
    np.testing.assert_allclose(seq.time[35], 0.02 + 0.5, atol=1e-12)   # second pulse starts after it
    assert np.all(np.diff(seq.time) > 0)