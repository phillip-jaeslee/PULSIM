"""
resonance_crossing(): the sweep rate and amplitude at resonance, measured
from the waveform. It is the basis for calibrating every adiabatic family by
Q without a per-family formula, so it is checked here against the two cases
that DO have a closed form: HypSec (tanh sweep) and a linear chirp (WURST).
"""

import numpy as np
import pytest

from PULSIM.calibration import resonance_crossing
from PULSIM.rf_shape import RFShape

T_MS = 2.0


def _hypsec(points):
    return RFShape.create("hypsec", duration=T_MS, points=points)


@pytest.mark.parametrize("points", [1001, 4001])
def test_hypsec_full_passage_matches_closed_form(points):
    shape = _hypsec(points)
    cal = shape.calibration
    # closed form: omega1^2 / Q = rate_norm / T^2  ->  rate_norm = T^2 omega1^2 / Q
    expected = T_MS**2 * (2 * np.pi * cal.nu1_for(T_MS)) ** 2 / cal.q_mid
    c = resonance_crossing(shape.envelope())
    assert c.rate_norm == pytest.approx(expected, rel=5e-5)
    assert c.u == pytest.approx(0.5, abs=1e-3)
    assert c.amp_rel == pytest.approx(1.0, abs=1e-3)
    assert not c.at_edge


@pytest.mark.parametrize("points", [1001, 4001])
def test_hypsec_half_passage_found_at_edge(points):
    """First half of a HypSec, odd points so it ends exactly on resonance.
    Duration T/2 at the same physical sweep rate -> rate_norm is 1/4."""
    env = _hypsec(points).envelope()
    full = resonance_crossing(env)
    half = resonance_crossing(env[: points // 2 + 1])
    assert half.at_edge
    assert half.u == 1.0
    assert half.amp_rel == pytest.approx(1.0, abs=1e-3)
    assert 4 * half.rate_norm == pytest.approx(full.rate_norm, rel=1e-4)


def test_linear_chirp_matches_closed_form():
    """WURST sweeps SW linearly over T: rate = 2*pi*SW/T -> rate_norm = 2*pi*SW*T."""
    shape = RFShape.create("wurst", duration=T_MS, points=4000)
    c = resonance_crossing(shape.envelope())
    assert c.rate_norm == pytest.approx(2 * np.pi * 40.0 * T_MS, rel=1e-3)   # SW in kHz


def test_composite_chirp_is_refused():
    shape = RFShape.create("compositesmoothedchirp", duration=T_MS, points=4000)
    with pytest.raises(ValueError, match="no\\s+single Q"):
        resonance_crossing(shape.envelope())


def test_unswept_pulse_is_refused():
    """A constant-phase pulse never sweeps through resonance."""
    env = RFShape.create("gausscasq5", duration=T_MS, points=1000).envelope()
    with pytest.raises(ValueError):
        resonance_crossing(env * np.exp(1j * 0.3))