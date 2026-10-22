"""
Tests for SimultaneousPulses: finite pulses on different channels played
at the same time.

Back-to-back ShapePulseSegments are NOT simultaneous: each runs with the
full offset + J Hamiltonian, so the pair lasts the sum of the durations and
each spin free-evolves while the other is pulsed. These tests pin down the
segment that fixes that, against independent ground truth:

  - one pulse alone is exactly a ShapePulseSegment;
  - with J = 0 the two spins do not talk, so a simultaneous pair must equal
    each spin on its own -- the shorter pulse as delay / pulse / delay
    (align="center") or pulse / delay (align="start");
  - with J on, two rectangular pulses must equal ONE exponential of the full
    Hamiltonian (offsets + J + both RF fields), and differ from playing the
    same two pulses one after the other;
  - the sequence diagram draws the pair as one event, each pulse in its own
    place.
"""

import os
import sys

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.backend import NumpyBackend
from PULSIM.liouville import (Delay, LiouvilleSequence, ShapePulseSegment,
                              SimultaneousPulses, _static_hamiltonian)
from PULSIM.pulse_oo import Pulse
from PULSIM.rf_shape import RFShape
from PULSIM.spin_operators import Ix, Iy, Iz, SpinOperators, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio

TWO_PI = 2 * np.pi
OFF_H, OFF_C = TWO_PI * 0.7, TWO_PI * -0.3          # rad/ms


def pulse(shape, channel, flip_deg, duration, points, axis="x"):
    return Pulse(RFShape.create(shape, duration=duration, points=points),
                 np.deg2rad(flip_deg), axis=axis,
                 backend=NumpyBackend(Gamma=gyro_ratio(channel)))


def run(segments, ss, sigma0):
    return LiouvilleSequence(segments, ss).propagate(sigma0)


def test_one_pulse_alone_is_a_shape_pulse_segment():
    ss = SpinSystem(nuclei=["H", "13C"], offsets=[OFF_H, OFF_C], couplings={(0, 1): 140.0})
    s0 = embed(Iz(), 0, 2) + embed(Iz(), 1, 2)
    p = pulse("hermite", "H", 90, 1.0, 256)
    np.testing.assert_allclose(run([SimultaneousPulses([p])], ss, s0),
                               run([ShapePulseSegment(p)], ss, s0), atol=1e-12)


@pytest.mark.parametrize("align, c_alone", [
    ("center", lambda pC: [Delay(0.3), ShapePulseSegment(pC), Delay(0.3)]),
    ("start",  lambda pC: [ShapePulseSegment(pC), Delay(0.6)]),
])
def test_without_coupling_each_spin_evolves_on_its_own(align, c_alone):
    pH = pulse("eburp1", "H", 90, 1.0, 200, "x")
    pC = pulse("hermite", "13C", 180, 0.4, 97, "y")          # shorter, off-grid
    ss = SpinSystem(nuclei=["H", "13C"], offsets=[OFF_H, OFF_C], couplings={})
    both = run([SimultaneousPulses([pH, pC], align=align)], ss,
               embed(Iz(), 0, 2) + embed(Iz(), 1, 2))
    ops = SpinOperators(ss)

    ssH = SpinSystem(nuclei=["H"], offsets=[OFF_H])
    ssC = SpinSystem(nuclei=["13C"], offsets=[OFF_C])
    rH = run([ShapePulseSegment(pH)], ssH, embed(Iz(), 0, 1))
    rC = run(c_alone(pC), ssC, embed(Iz(), 0, 1))
    oH, oC = SpinOperators(ssH), SpinOperators(ssC)
    for a in "xyz":
        assert ops.expectation(both, "I" + a) == pytest.approx(oH.expectation(rH, "I" + a), abs=1e-10)
        assert ops.expectation(both, "S" + a) == pytest.approx(oC.expectation(rC, "I" + a), abs=1e-10)


def test_hard_pair_with_coupling_matches_one_exponential():
    tau = 0.2                                               # ms, each 180 pulse
    pH = pulse("hard", "H", 180, tau, 10)
    pC = pulse("hard", "13C", 180, tau, 10)
    ss = SpinSystem(nuclei=["H", "13C"], offsets=[TWO_PI * 1.0, TWO_PI * 0.5],
                    couplings={(0, 1): 140.0})
    H = _static_hamiltonian(ss) + (np.pi / tau) * (embed(Ix(), 0, 2) + embed(Ix(), 1, 2))
    U = expm(-1j * H * tau)
    s0 = product_operator(Iy(), 0, Iz(), 1, 2)              # INEPT, before the 180 pair

    np.testing.assert_allclose(run([SimultaneousPulses([pH, pC])], ss, s0),
                               U @ s0 @ U.conj().T, atol=1e-12)
    # and the old way -- one after the other -- is measurably different
    seq = run([ShapePulseSegment(pH), ShapePulseSegment(pC)], ss, s0)
    ops = SpinOperators(ss)
    assert abs(ops.expectation(seq, "2IySz") - ops.expectation(U @ s0 @ U.conj().T, "2IySz")) > 0.5


def test_duration_is_the_longest_pulse():
    seg = SimultaneousPulses([pulse("eburp1", "H", 90, 1.0, 200),
                              pulse("hermite", "13C", 180, 0.4, 97)])
    assert seg.duration == pytest.approx(1.0)


def test_two_pulses_on_one_channel_are_refused():
    with pytest.raises(ValueError, match="share a channel"):
        SimultaneousPulses([pulse("hermite", "H", 90, 1.0, 64), pulse("hermite", "H", 90, 1.0, 64)])


def test_bad_align_is_refused():
    with pytest.raises(ValueError, match="align"):
        SimultaneousPulses([pulse("hermite", "H", 90, 1.0, 64)], align="end")


def test_diagram_draws_the_pair_as_one_event():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PULSIM.sequence_figure import describe_sequence, draw_sequence

    ss = SpinSystem(nuclei=["H", "13C"], offsets=[0.0, 0.0])
    seq = LiouvilleSequence([SimultaneousPulses([pulse("reburp", "H", 180, 1.2, 200),
                                                 pulse("hermite", "13C", 180, 0.6, 128)])], ss)
    (item,) = describe_sequence(seq)
    assert item["kind"] == "shaped"
    assert item["channels"] == (0, 1)
    assert item["duration"] == pytest.approx(1.2)
    assert [(s["start"], s["duration"]) for s in item["subs"]] == [
        pytest.approx((0.0, 1.2)), pytest.approx((0.3, 0.6))]      # 13C centred
    for to_scale in (False, True):
        fig, ax = plt.subplots()
        draw_sequence(seq, ax=ax, to_scale=to_scale)
        plt.close(fig)