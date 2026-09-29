"""
Every analytic adiabatic shape now calibrates from Q (or, for SinCos, from its
own sweep), and the check that matters is physical: at Q = 5 the pulse must
actually invert on resonance in the Bloch engine. Bruker's rule of thumb is
Q = 5 for inversion; these tests hold every family to it, not just HypSec.
"""

import numpy as np
import pytest

import PULSIM

GAMMA = 42.577478518        # kHz/mT, 1H
T_MS = 2.0
POINTS = 2000

# Shapes calibrated by a different rule, or not at all -- tested separately.
SPECIAL = {"hypsec", "sincos", "compositesmoothedchirp"}


def _adiabatic_names():
    names = []
    for name in sorted(PULSIM.RFShape.available()):
        if name in ("file", "composite"):
            continue
        if PULSIM.RFShape.create(name, duration=T_MS, points=100).intent == "adiabatic":
            names.append(name)
    return names


def _pulse(name, **kw):
    shape = PULSIM.RFShape.create(name, duration=T_MS, points=POINTS, **kw)
    return PULSIM.Pulse(shape, backend=PULSIM.NumpyBackend(Gamma=GAMMA))


def _mz_on_resonance(pulse):
    M = np.array([[0.0], [0.0], [1.0]])
    return float(pulse.apply(M, np.array([0.0]))[2, 0])


def test_the_special_cases_are_still_adiabatic():
    """Guard the SPECIAL set: if a name is renamed, the loop below would
    silently stop covering it."""
    assert SPECIAL <= set(_adiabatic_names())


@pytest.mark.parametrize("name", sorted(set(_adiabatic_names()) - SPECIAL))
def test_q5_inverts_on_resonance(name):
    pulse = _pulse(name)
    assert pulse.realized_q == pytest.approx(5.0, rel=1e-9)
    assert _mz_on_resonance(pulse) < -0.99


@pytest.mark.parametrize("name", sorted(set(_adiabatic_names()) - SPECIAL))
def test_q2_does_not_fully_invert(name):
    """The other half of the rule of thumb -- otherwise Q = 5 passing proves
    nothing about the calibration, only that the RF was large enough."""
    assert _mz_on_resonance(_pulse(name, q_mid=2.0)) > -0.97


def test_composite_chirp_refuses_with_a_way_out():
    """One resonance crossing per segment: no single Q exists."""
    shape = PULSIM.RFShape.create("compositesmoothedchirp", duration=T_MS, points=POINTS)
    with pytest.raises(NotImplementedError, match="nu1_max"):
        shape.calibration


def test_sincos_amplitude_matches_its_sweep():
    """Bendall & Pegg: nu1_max = peak offset = factor / T."""
    pulse = _pulse("sincos")
    assert pulse.nu1_max == pytest.approx(4.0 / T_MS, rel=1e-4)
    assert pulse.realized_q == pytest.approx(8.0, rel=1e-3)
    assert _mz_on_resonance(pulse) < -0.999


def test_sincos_explicit_q_overrides_the_matched_amplitude():
    pulse = _pulse("sincos", q_mid=5.0)
    assert pulse.realized_q == pytest.approx(5.0, rel=1e-9)
    assert pulse.nu1_max < 4.0 / T_MS


def test_sincos_half_passage_excites():
    """Half passage ends on resonance: a 90-degree adiabatic excitation."""
    pulse = _pulse("sincos", full_passage=False)
    assert pulse.nu1_max == pytest.approx(4.0 / T_MS, rel=1e-4)
    assert abs(_mz_on_resonance(pulse)) < 0.05