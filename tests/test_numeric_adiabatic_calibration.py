"""
NumericAdiabaticCalibration: Q calibration from the measured resonance
crossing. Held to the HypSec closed form (AdiabaticCalibration), which it must
reproduce, and to the linear-chirp closed form for WURST.
"""

import numpy as np
import pytest

from PULSIM.calibration import NumericAdiabaticCalibration
from PULSIM.rf_shape import RFShape

T_MS = 2.0


@pytest.fixture
def hypsec():
    return RFShape.create("hypsec", duration=T_MS, points=4001)


@pytest.mark.parametrize("q", [2.0, 5.0, 10.0])
def test_matches_hypsec_closed_form(hypsec, q):
    closed = hypsec.calibration                     # AdiabaticCalibration
    numeric = NumericAdiabaticCalibration.from_envelope(hypsec.envelope(), q_mid=q)
    expected = closed.nu1_over_sqrt_q(T_MS) * np.sqrt(q)
    assert numeric.nu1_for(T_MS) == pytest.approx(expected, rel=1e-5)


def test_nu1_over_sqrt_q_is_q_independent(hypsec):
    a = NumericAdiabaticCalibration.from_envelope(hypsec.envelope(), q_mid=2.0)
    b = NumericAdiabaticCalibration.from_envelope(hypsec.envelope(), q_mid=10.0)
    assert a.nu1_over_sqrt_q(T_MS) == b.nu1_over_sqrt_q(T_MS)


def test_q_for_inverts_nu1_for(hypsec):
    cal = NumericAdiabaticCalibration.from_envelope(hypsec.envelope(), q_mid=5.0)
    assert cal.q_for(cal.nu1_for(T_MS), T_MS) == pytest.approx(5.0, rel=1e-12)


def test_linear_chirp_closed_form():
    """Linear sweep SW over T, amplitude 1 at the centre:
    Q = omega1^2 / (2*pi*SW/T)  ->  nu1 = sqrt(Q * SW / (2*pi*T))   [kHz]"""
    shape = RFShape.create("wurst", duration=T_MS, points=4000)
    cal = NumericAdiabaticCalibration.from_envelope(shape.envelope(), q_mid=5.0)
    expected = np.sqrt(5.0 * 40.0 / (2 * np.pi * T_MS))       # SW = 40 kHz
    assert cal.nu1_for(T_MS) == pytest.approx(expected, rel=5e-4)