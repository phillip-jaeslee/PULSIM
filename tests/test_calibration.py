"""
Test for PULSIM/calibration.py

Two jobs:

1. AreaCalibration must reproduce the legacy Pulse.calibrated_rf() exactly for
   every amplitude-modulatd shape. This is the equivalence that lets Phase 2
   delete the old calibration path without changing a single published number.

2. AdiabaticCalibration must reproduce Bruker's own stored design constants,
   and must actually invert at the Q the Bruker manual recommends.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PULSIM
from PULSIM.calibration import (
    AreaCalibration,
    AdiabaticCalibration,
    beta_from_truncation,
    mu_from_sweep_width,
    signed_integral_of,
)

GAMMA = 42.577478518        # kHz/mT, 1H
DURATION = 2.0              # ms
POINTS = 1000

# Values stored in a real Bruker HypSec file (SHL_BETA / SHL_MU)
# generated with SHL_SW = 20 Hz and SHL_TRUNCLEV = 1%.
BRUKER_BETA = 5.298292
BRUKER_MU = 5.929443

def amplitude_modulated_shapes():
    names = []
    for name in sorted(PULSIM.RFShape.available()):
        if name in ("file", "composite"):
            continue
        shape = PULSIM.RFShape.create(name, duration=DURATION, points=POINTS)
        if not np.any(shape.envelope().imag != 0):
            names.append(name)
    return names

@pytest.mark.parametrize("name", amplitude_modulated_shapes())
@pytest.mark.parametrize("flip_deg", [90, 180])
def test_area_calibration_matches_legacy(name, flip_deg):
    """The new form is the old form, rearranged. It must not move a number."""
    shape = PULSIM.RFShape.create(name, duration=DURATION, points=POINTS)
    flip = np.deg2rad(flip_deg)

    legacy_peak = np.abs(
        PULSIM.Pulse(shape, flip=flip, axis="x", backend=PULSIM.NumpyBackend(Gamma=GAMMA)).calibrated_rf()
    ).max()

    cal = AreaCalibration(signed_integral_of(shape.envelope()))
    new_peak = cal.nu1_for(flip, DURATION) / GAMMA

    assert new_peak == pytest.approx(legacy_peak, rel=1e-12)

def test_bruker_design_constants():
    """beta and mu must reproduce the vendor file's own stored values."""
    beta = beta_from_truncation(1.0)
    mu = mu_from_sweep_width(20.0, beta)

    assert beta == pytest.approx(BRUKER_BETA, abs=1e-6)
    assert mu == pytest.approx(BRUKER_MU, abs=1e-5)

    # Reverse check: the design must give back the sweep width it came from.
    assert 2 * mu * beta / np.pi == pytest.approx(20.0, rel=1e-9)

def _invert_on_resonance(nu1_khz, duration, points=4000):
    shape = PULSIM.RFShape.create("hypsec", duration=duration, points=points)
    env = shape.envelope()
    rf = env / np.abs(env).max() * (nu1_khz / GAMMA)

    df = np.array([0.0])
    M = np.zeros((3, 1))
    M[2] = 1.0
    backend = PULSIM.NumpyBackend(Gamma=GAMMA)

    for n in range(len(rf)):
        B = np.stack([np.full_like(df, rf[n].real),
                      np.full_like(df, rf[n].imag),
                      df / GAMMA], axis=1)
        M = backend.rotate(M, shape.dt, B, "x")

    return float(M[2, 0])

def _hypsec_calibration(q_mid):
    beta = beta_from_truncation(1.0)
    return AdiabaticCalibration(q_mid=q_mid, mu=mu_from_sweep_width(20.0, beta), beta=beta)

def test_q5_inverts_and_q1_does_not():
    """Bruker's rule of thumb, reproduced from the physics."""
    assert _invert_on_resonance(_hypsec_calibration(5.0).nu1_for(2.0), 2.0) < -0.99
    assert _invert_on_resonance(_hypsec_calibration(1.0).nu1_for(2.0), 2.0) > -0.70

@pytest.mark.parametrize("duration", [2.0, 4.0, 8.0])
def test_inversion_is_duration_invariant_at_fixed_q(duration):
    """The defining signature of adiabatic behaviour -- and the reason Q, not
    flip angle, is the right control parameter."""
    mz = _invert_on_resonance(_hypsec_calibration(5.0).nu1_for(duration), duration)
    assert mz == pytest.approx(-0.9997, abs=2e-4)

def test_nu1_over_sqrt_q_is_q_independent():
    """Bruker's integradia reports this quantity precisely because it does not
    depend on Q."""
    a = _hypsec_calibration(2.0).nu1_over_sqrt_q(2.0)    
    b = _hypsec_calibration(9.0).nu1_over_sqrt_q(2.0)
    assert a == pytest.approx(b, rel=1e-15)

def phase_modulated_shapes():
    names = []
    for name in sorted(PULSIM.RFShape.available()):
        if name in ("file", "composite"):
            continue
        shape = PULSIM.RFShape.create(name, duration=DURATION, points=POINTS)
        if np.any(shape.envelope().imag != 0):
            names.append(name)
    return names

def adiabatic_without_calibration():
    """Every adiabatic shape except the one whose sweep rate we have derived."""
    return [n for n in phase_modulated_shapes() if n != "hypsec"]

@pytest.mark.parametrize("name", adiabatic_without_calibration())
def test_adiabatic_families_without_a_sweep_rate_refuse(name):
    """Ten of the eleven have no derived sweep-rate expression. They must say
    so rather than silently falling back to area calibration."""
    shape = PULSIM.RFShape.create(name, duration=DURATION, points=POINTS)
    with pytest.raises(NotImplementedError, match="adiabatic calibration"):
        PULSIM.Pulse(shape, backend=PULSIM.NumpyBackend(Gamma=GAMMA)).nu1_max

def test_hypsec_calibrates_without_a_flip_angle():
    """The point of the whole redesign: no flip angle, and it still works."""
    shape = PULSIM.RFShape.create("hypsec", duration=2.0, points=1000)
    pulse = PULSIM.Pulse(shape, backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    assert pulse.nu1_max == pytest.approx(4.5914306592, rel=1e-9)
    assert pulse.realized_q == pytest.approx(5.0, rel=1e-9)

def test_flip_angle_on_adiabatic_shape_warns_and_changes_nothing():
    shape = PULSIM.RFShape.create("hypsec", duration=2.0, points=1000)
    backend = PULSIM.NumpyBackend(Gamma=GAMMA)
    with pytest.warns(UserWarning, match="does not scale the RF amplitude"):
        with_flip = PULSIM.Pulse(shape, flip=np.pi / 2, backend=backend).nu1_max
    assert with_flip == PULSIM.Pulse(shape, backend=backend).nu1_max

def test_explicit_nu1_max_wins_and_realized_q_reports_it():
    """Precedence: explicit amplitude beats the design Q, and the achieved Q
    is recomputed as a diagnostic rather than silently assumed."""
    shape = PULSIM.RFShape.create("hypsec", duration=2.0, points=1000)
    pulse = PULSIM.Pulse(shape, nu1_max=2.0, backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    assert pulse.nu1_max == 2.0
    assert pulse.realized_q == pytest.approx(0.9487109995, rel=1e-6)

def test_no_empirical_factor_of_two():
    """The historical 'x2 if is_adiabatic' had no derivation. Guard against it
    coming back for any amplitude-modulated shape."""
    shape = PULSIM.RFShape.create("gausscasq5", duration=DURATION, points=POINTS)    
    pulse = PULSIM.Pulse(shape, flip=np.pi / 2, axis="x", backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    expected = AreaCalibration(signed_integral_of(shape.envelope())).nu1_for(np.pi / 2, DURATION) / GAMMA
    assert np.abs(pulse.calibrated_rf()).max() == pytest.approx(expected, rel=1e-12)

def _sweep_times_duration(shape, duration):
    """The sweep width the stored phase actually realises, times duration.

    nu_RF(t) = (1/(2*pi*T)) * dphi/du with u = t/T, so this product is the
    duration-invariant "sweep width for a 1 second pulse" -- Bruker's SHL_SW.
    """
    phase = np.unwrap(np.angle(shape.envelope()))
    u = np.linspace(0.0, 1.0, shape.points)
    nu = np.gradient(phase, u) / (2.0 * np.pi * duration)
    return (nu.max() - nu.min()) * duration

def test_hypsec_reproduces_the_bruker_design_constants():
    """beta and mu are consequences of truncation and sweep width, not free
    parameters. Replaces the legacy-equivalence fixture for hypsec.
    """
    shape = PULSIM.RFShape.create("hypsec", duration=0.6, points=1000)
    assert shape.beta == pytest.approx(BRUKER_BETA, abs=1e-6)
    assert shape.mu == pytest.approx(BRUKER_MU, abs=1e-6)

@pytest.mark.parametrize("sweep_width_1s", [10.0, 20.0, 40.0])
def test_hypsec_waveform_realizes_its_declared_sweep_width(sweep_width_1s):
    """The design input must actaully reach the waveform.

    The old code advertised a 'sweepwidth' parameter in its docstring and never
    read it: passing sweepwidth=99999 produced a byte-identical envelop.
    """
    shape = PULSIM.RFShape.create("hypsec", duration=0.6, points=1000, sweep_width_1s=sweep_width_1s)
    assert _sweep_times_duration(shape, 0.6) == pytest.approx(sweep_width_1s, rel=1e-3)

@pytest.mark.parametrize("truncation_percent", [0.5, 1.0, 5.0])
def test_hypsec_truncation_sets_the_edge_amplitude(truncation_percent):
    """beta = arccosh(1/truncation), so the envelope starts and ends just above
    the truncation level.

    Not *exactly* at it: _build searches a 10,000-point grid for the first
    sample exceeding the threshold, so the edge value is quantized by that grid
    (~3e-3 relative here). A tighter bound is machine-dependent -- a last-bit
    difference in cosh(beta*t)**(-1-1j*mu) moves the boundary by one index
    between numpy versions, which is how CI caught this.
    """
    shape = PULSIM.RFShape.create("hypsec", duration=0.6, points=1000,
                                  truncation_percent=truncation_percent)
    amp = np.abs(shape.envelope())
    target = truncation_percent / 100.0

    assert target <= amp[0] < target * 1.01
    assert target <= amp[-1] < target * 1.01