"""
SincShape must be one central lobe in normalized time.

The sinc argument is 2t/T, so the first zero crossings land on the pulse
edges at every duration. These tests exist because it was previously
np.sinc(t) with t in absolute milliseconds: the zeros sat at +/- 1 ms
regardless of duration, so a 1 ms "sinc" had no zero crossing at all, a
4 ms one had four lobes, and the calibration did not scale as 1/T.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PULSIM
from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.rf_shape import RFShape
from PULSIM.spin_system import gyro_ratio

PI = np.pi
POINTS = 512
GAMMA_H = gyro_ratio("H")
DURATIONS = [0.25, 0.5, 1.0, 2.0, 4.0]


def envelope(duration, points=POINTS):
    return np.real(RFShape.create("sinc", duration=duration, points=points).envelope())


def normalized(duration, points=POINTS):
    e = envelope(duration, points)
    return e / np.abs(e).max()


def nu1_max(duration, points=POINTS):
    shape = RFShape.create("sinc", duration=duration, points=points)
    return Pulse(shape, flip=PI / 2, backend=NumpyBackend(Gamma=GAMMA_H)).nu1_max


def fwhm(duration, points=POINTS):
    """Full width of |Mxy| at half its on-resonance value, kHz."""
    shape = RFShape.create("sinc", duration=duration, points=points)
    pulse = Pulse(shape, flip=PI / 2, axis="x", backend=NumpyBackend(Gamma=GAMMA_H))
    df = np.linspace(0.0, 12.0 / duration, 8000)
    M = np.zeros((3, df.size))
    M[2] = 1.0
    M = pulse.apply(M, df)
    mxy = np.hypot(M[0], M[1])
    i = int(np.argmax(mxy < 0.5))
    assert i > 0, "profile never falls below half height in the scanned range"
    f = (mxy[i - 1] - 0.5) / (mxy[i - 1] - mxy[i])
    return 2.0 * (df[i - 1] + f * (df[i] - df[i - 1]))


# --- the shape is duration-independent -----------------------------------

def test_normalized_envelope_is_duration_invariant():
    """Stretching the pulse must not reshape it. This is what the absolute
    time argument got wrong, and it is the property every other test here
    depends on."""
    base = normalized(DURATIONS[0])
    for T in DURATIONS[1:]:
        np.testing.assert_allclose(normalized(T), base, rtol=0, atol=1e-12,
                                   err_msg=f"envelope changed shape at duration {T} ms")


def test_the_pulse_is_a_single_lobe():
    """One central lobe: the envelope never changes sign in the interior."""
    e = envelope(1.0)
    interior = e[1:-1]
    assert np.all(interior > -1e-12), "sinc envelope has an interior sign change"


def test_first_zero_sits_at_the_pulse_edge():
    """sample_times() starts at exactly -T/2, so the first sample is the
    first zero crossing -- at every duration."""
    for T in DURATIONS:
        e = envelope(T)
        assert abs(e[0]) < 1e-12 * np.abs(e).max(), f"edge sample is not a zero at {T} ms"


# --- calibration and bandwidth then follow --------------------------------

def test_nu1max_scales_as_one_over_duration():
    ref = nu1_max(DURATIONS[0]) * DURATIONS[0]
    for T in DURATIONS:
        assert nu1_max(T) * T == pytest.approx(ref, rel=1e-12), \
            f"nu1_max * T is not constant at duration {T} ms"


def test_bandwidth_duration_product_is_constant():
    ref = fwhm(DURATIONS[0]) * DURATIONS[0]
    for T in DURATIONS:
        assert fwhm(T) * T == pytest.approx(ref, rel=1e-4), \
            f"bandwidth * duration is not constant at duration {T} ms"


# --- nothing already published moves --------------------------------------

def test_two_millisecond_waveform_is_unchanged():
    """2t/T == t exactly when T == 2, so the normalized definition and the
    old absolute-time one agree there. Every stored 2 ms reference -- the
    lesson pages' shapes.js among them -- must therefore still be valid."""
    T = 2.0
    shape = RFShape.create("sinc", duration=T, points=POINTS)
    t = shape.sample_times()
    legacy = np.hamming(POINTS).T * np.sinc(t)
    np.testing.assert_allclose(np.real(shape.envelope()), legacy, rtol=0, atol=1e-15)


def test_two_millisecond_calibration_is_unchanged():
    """Measured from the package, not derived: if this moves, a stored
    reference somewhere else has silently gone stale."""
    assert nu1_max(2.0) == pytest.approx(0.2964185889419243, rel=1e-12)


# --- the specific regression ----------------------------------------------

def test_a_short_pulse_is_not_a_windowed_hump():
    """Under the old definition a 1 ms sinc contained no zero crossing, so
    it was a Hamming hump rather than a sinc. Its normalized envelope would
    then differ from the 2 ms one."""
    np.testing.assert_allclose(normalized(1.0), normalized(2.0), rtol=0, atol=1e-12)
