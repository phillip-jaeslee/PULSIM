"""
Frame j of a trajectory is the state after j RF steps, i.e. at the instant
step j begins: time[j]. The RF panels must label it so, not one step early.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from PULSIM.visualization import _rf_panels       # noqa: E402


def test_frame_j_is_labelled_time_j():
    time = np.array([0.0, 0.1, 0.2, 0.3])             # 4 RF steps
    rf = np.array([1.0, 2.0, 3.0, 4.0])
    phase = np.zeros(4)
    t, amp, pha = _rf_panels(time, rf, phase, n_frames=5)
    assert t == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert amp == pytest.approx([1.0, 2.0, 3.0, 4.0, 4.0])


def test_time_need_not_start_at_zero():
    """A centred time grid (as sample_times gives) keeps its own origin."""
    time = np.array([-0.2, -0.1, 0.0, 0.1])
    t, _, _ = _rf_panels(time, np.ones(4), np.zeros(4), n_frames=5)
    assert t == pytest.approx([-0.2, -0.1, 0.0, 0.1, 0.2])


def test_equal_lengths_pass_through_unchanged():
    time = np.array([0.0, 0.1, 0.2])
    t, _, _ = _rf_panels(time, np.ones(3), np.zeros(3), n_frames=3)
    assert t == pytest.approx(time)