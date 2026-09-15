"""
Tests for sequence_figure: the diagram must agree with the sequence.

The point of deriving a figure from the segment list is that it cannot drift
from the simulation. These test the two ways it could drift anyway -- a wrong
total duration, and simultaneous pulses drawn as though they were sequential.
The second one is a real regression: the first version of draw_sequence put
the two 180 degree pulses at different x positions.

matplotlib is an optional extra, so this module skips entirely without it --
the base-only CI job installs numpy and scipy alone.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PULSIM
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, ShapePulseSegment, LiouvilleSequence, RawShapedPulseSegment
from PULSIM.sequence_figure import draw_sequence, describe_sequence, _group_simultaneous, _layout

PI = np.pi
GAMMA_H = gyro_ratio("H")
DELTA = 1.724


def inept(shaped_duration=2.0):
    ss = SpinSystem(nuclei=["H", "13C"], offsets=[0.0, 0.0], couplings={(0, 1): 145.0})
    shape = PULSIM.RFShape.create("gausscasq5", duration=shaped_duration, points=64)
    pulse = PULSIM.Pulse(shape, flip=PI / 2, axis="x",
                         backend=PULSIM.NumpyBackend(Gamma=GAMMA_H))
    return LiouvilleSequence([
        ShapePulseSegment(pulse),
        Delay(DELTA),
        IdealPulse(PI, phase=0.0, channel="H"),
        IdealPulse(PI, phase=0.0, channel="13C"),
        Delay(DELTA),
        IdealPulse(PI / 2, phase=PI / 2, channel="H"),
        IdealPulse(PI / 2, phase=0.0, channel="13C"),
    ], ss)


def test_describe_reports_the_real_durations_and_channels():
    items = describe_sequence(inept())
    assert [it["kind"] for it in items] == [
        "shaped", "delay", "ideal", "ideal", "delay", "ideal", "ideal"]
    assert items[0]["duration"] == pytest.approx(2.0)
    assert items[1]["duration"] == pytest.approx(DELTA)
    assert items[0]["channels"] == (0,)      # the shaped pulse is on 1H only
    assert items[2]["channels"] == (0,)
    assert items[3]["channels"] == (1,)


def test_simultaneous_pulses_become_one_event():
    """The regression. Two 180s on different channels are separate segments --
    correct, since an ideal pulse has no duration -- but drawing them in
    separate slots asserts that they do not happen together."""
    events = _group_simultaneous(describe_sequence(inept()))
    assert [e["kind"] for e in events] == ["shaped", "delay", "ideal", "delay", "ideal"]
    assert len(events[2]["pulses"]) == 2
    assert set(events[2]["channels"]) == {0, 1}


def test_pulses_on_the_same_channel_do_not_merge():
    """A composite 90x-90y on one channel really is two events in a row, and
    merging them would be just as wrong as splitting a simultaneous pair."""
    ss = SpinSystem(nuclei=["H"], offsets=[0.0])
    seq = LiouvilleSequence([IdealPulse(PI / 2, phase=0.0, channel="H"),
                             IdealPulse(PI / 2, phase=PI / 2, channel="H")], ss)
    assert len(_group_simultaneous(describe_sequence(seq))) == 2


def test_to_scale_axis_is_the_real_time_axis():
    """With no minimum-width floor, drawn width IS duration -- every segment
    and the total."""
    events = _group_simultaneous(describe_sequence(inept()))
    positions, total, faked = _layout(events, to_scale=True, min_pulse_frac=0.0)
    assert faked == []
    assert total == pytest.approx(sum(e["duration"] for e in events), abs=1e-12)
    for e, (_, w) in zip(events, positions):
        assert w == pytest.approx(e["duration"], abs=1e-12)


def test_the_minimum_width_is_reported_not_hidden():
    """An ideal pulse is drawn wider than it is, because its true duration is
    zero. draw_sequence prints a note saying so; this asserts the note has
    something to report."""
    events = _group_simultaneous(describe_sequence(inept()))
    _, total, faked = _layout(events, to_scale=True, min_pulse_frac=0.012)
    assert faked, "ideal pulses were widened, so the caller must be told"
    assert total > sum(e["duration"] for e in events)


def test_the_shaped_pulse_outlasts_the_delay():
    """The reason a to-scale rendering exists: a 2 ms selective pulse is
    longer than the 1.724 ms delay after it, and the picture should show it."""
    events = _group_simultaneous(describe_sequence(inept(shaped_duration=2.0)))
    positions, _, _ = _layout(events, to_scale=True, min_pulse_frac=0.0)
    assert positions[0][1] > positions[1][1]


def test_an_unknown_segment_type_raises():
    """A new Segment subclass must force a decision, not vanish silently
    from the diagram."""
    class Unhandled:
        pass
    seq = LiouvilleSequence([Unhandled()], SpinSystem(nuclei=["H"], offsets=[0.0]))
    with pytest.raises(TypeError, match="does not know how to draw"):
        describe_sequence(seq)


def test_draw_sequence_runs_in_both_modes():
    import matplotlib.pyplot as plt
    seq = inept()
    for to_scale in (False, True):
        fig, ax = plt.subplots()
        draw_sequence(seq, ax=ax, to_scale=to_scale)
        plt.close(fig)

def test_raw_shaped_segment_reports_channels():
    """The branch the INEPT tutorial actually uses. ShapePulseSegment and
    RawShapedPulseSegment are separate branches, so a key spelled wrong in
    one of them is invisible to every test that exercises the other."""
    ss = SpinSystem(nuclei=["H", "13C"], offsets=[0.0, 0.0])
    rf = np.full(64, 5.0, dtype=complex)
    seq = LiouvilleSequence([RawShapedPulseSegment(rf, 1.5 / 64, channel="H")], ss)

    (item,) = describe_sequence(seq)
    assert item["kind"] == "shaped"
    assert item["channels"] == (0,)
    assert item["duration"] == pytest.approx(1.5)
    draw_sequence(seq, to_scale=True)     # the KeyError was here, not above