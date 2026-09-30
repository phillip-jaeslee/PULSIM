"""
PulseSequence: several Pulses applied one after another, with the RF, phase
and time traces concatenated for plotting.

The physics check is composition: a sequence must do exactly what calling
each Pulse's apply() in turn does, and two 90-degree pulses in a row must be
one 180. The bookkeeping checks pin the conventions visualization.py relies
on: a trajectory has one frame per RF step plus the starting frame, and
.time is continuous across pulse boundaries.
"""

import numpy as np
import pytest

import PULSIM
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.spin_system import gyro_ratio

BACKEND = PULSIM.NumpyBackend(Gamma=gyro_ratio('H'))
DF = np.linspace(-3.0, 3.0, 13)                      # kHz


def _pulse(name, duration, points, flip, axis="x"):
    shape = PULSIM.RFShape.create(name, duration=duration, points=points)
    return PULSIM.Pulse(shape, flip, axis=axis, backend=BACKEND)


def _z(n=DF.size):
    M = np.zeros((3, n))
    M[2] = 1.0
    return M


@pytest.fixture
def mixed():
    """Different shapes, lengths, point counts and axes."""
    return PulseSequence([
        _pulse("eburp2", 1.0, 400, np.pi / 2, axis="y"),
        _pulse("hard", 0.012, 12, np.pi, axis="x"),
        _pulse("gausscasq5", 0.8, 300, np.pi / 2, axis="x"),
    ])


def test_equals_applying_each_pulse_in_turn(mixed):
    M = _z()
    for pulse in mixed:
        M = pulse.apply(M, DF)
    assert np.array_equal(mixed.run(_z(), DF), M)


def test_two_90s_are_one_180():
    """Same RF amplitude, same steps: 2 x (90x, 0.5 ms) == 180x, 1 ms."""
    half = _pulse("hard", 0.5, 250, np.pi / 2)
    whole = _pulse("hard", 1.0, 500, np.pi)
    assert half.nu1_max == pytest.approx(whole.nu1_max, rel=1e-12)
    assert np.allclose(PulseSequence([half, half]).run(_z(), DF),
                       whole.apply(_z(), DF), atol=1e-12)


def test_trajectory_frames(mixed):
    traj = mixed.run(_z(), DF, trajectory=True)
    assert traj.shape == (len(mixed.rf) + 1, 3, DF.size)
    assert np.array_equal(traj[0], _z())
    assert np.allclose(traj[-1], mixed.run(_z(), DF), atol=1e-14)
    # the frame at the first pulse boundary is that pulse's own end state
    n1 = mixed[0].shape.points
    assert np.allclose(traj[n1], mixed[0].apply(_z(), DF), atol=1e-14)


def test_time_is_continuous_across_pulses(mixed):
    t = mixed.time
    assert len(t) == len(mixed.rf) == len(mixed.phase)
    assert t[0] == 0.0
    assert np.all(np.diff(t) > 0)
    start = 0.0
    idx = 0
    for pulse in mixed:
        n, dt = pulse.shape.points, pulse.shape.dt
        assert t[idx] == pytest.approx(start, abs=1e-12)
        assert np.diff(t[idx:idx + n]) == pytest.approx(np.full(n - 1, dt))
        start += pulse.shape.duration
        idx += n


def test_rf_and_phase_are_the_pulses_in_order(mixed):
    assert np.array_equal(mixed.rf, np.concatenate([p.calibrated_rf() for p in mixed]))
    assert np.array_equal(mixed.phase, np.concatenate([p.shape.phase_profile for p in mixed]))


def test_container_behaviour():
    a, b = _pulse("hard", 0.1, 10, np.pi / 2), _pulse("hard", 0.1, 10, np.pi)
    seq = PulseSequence()
    assert seq.append(a) is seq                       # chainable
    seq.append(b)
    assert len(seq) == 2 and seq[0] is a and list(seq) == [a, b]