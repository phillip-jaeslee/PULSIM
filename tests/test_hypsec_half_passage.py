"""
HypSec half passage: the first half of the full-passage design, ending on
resonance -- an adiabatic 90-degree excitation (audit Part 5, item 5.10-1).

Closed form: a half passage of length T is the first half of a full passage of
length 2T, so  nu1_half(T) = nu1_full(2T)  (a = 1/2 in AdiabaticCalibration).
Held to the numerical calibration, which measures the sweep rate at the edge
crossing independently, and to the Bloch engine.
"""

import numpy as np
import pytest

import PULSIM
from PULSIM.calibration import NumericAdiabaticCalibration, resonance_crossing

GAMMA = 42.577478518        # kHz/mT, 1H
T_MS = 2.0


def _half(points=2000, **kw):
    return PULSIM.RFShape.create("hypsec", duration=T_MS, points=points,
                                 passage="half", **kw)


def test_half_passage_is_first_half_of_full_2T():
    half = _half().calibration.nu1_for(T_MS)
    full = PULSIM.RFShape.create("hypsec", duration=2 * T_MS).calibration.nu1_for(2 * T_MS)
    assert half == pytest.approx(full, rel=1e-12)


@pytest.mark.parametrize("points", [1000, 4000])
def test_closed_form_matches_numerical_calibration(points):
    shape = _half(points)
    closed = shape.calibration.nu1_for(T_MS)
    numeric = NumericAdiabaticCalibration.from_envelope(shape.envelope(), 5.0).nu1_for(T_MS)
    assert numeric == pytest.approx(closed, rel=2e-5)


def test_waveform_ends_on_resonance_at_peak_amplitude():
    env = _half().envelope()
    assert abs(env[0]) == pytest.approx(0.01, rel=1e-6)     # truncation level
    assert abs(env[-1]) == pytest.approx(1.0, rel=1e-12)
    c = resonance_crossing(env)
    assert c.at_edge and c.u == 1.0


def _m_on_resonance(q):
    pulse = PULSIM.Pulse(_half(q_mid=q), backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    return pulse.apply(np.array([[0.0], [0.0], [1.0]]), np.array([0.0]))[:, 0]


def test_q5_excites_90_degrees_on_resonance():
    M = _m_on_resonance(5.0)
    assert abs(M[2]) < 0.02
    assert np.hypot(M[0], M[1]) > 0.999


def test_q2_does_not_fully_excite():
    assert _m_on_resonance(2.0)[2] > 0.1


def test_unknown_passage_is_refused():
    with pytest.raises(ValueError, match="passage"):
        PULSIM.RFShape.create("hypsec", duration=T_MS, passage="quarter").envelope()