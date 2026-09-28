"""
Regression test: frequency-swept adiabatic shapes sweep the width they declare.

Nine shapes built their phase as offset_hz * dt with dt in ms, so every sweep
ran 1000x too fast and aliased to about +/-1 MHz. Nothing caught it, because
no test looked at the phase. These tests do.

The instantaneous offset is taken from the phase step between samples, and
only where the amplitude is non-negligible -- at a zero-amplitude edge the
phase is undefined.
"""

import numpy as np
import pytest

from PULSIM.rf_shape import RFShape

DURATION_MS = 2.0
POINTS = 4000
SWEEP_WIDTH_HZ = 40000.0          # the shared default: +/-20 kHz

SINGLE_SWEEP = [
    "wurst", "smoothedchirp", "tanhtan",
    "cawurst", "casmoothedchirp", "cagauss", "calorentz", "capowhsec",
]
ALL_SWEPT = SINGLE_SWEEP + ["compositesmoothedchirp"]


def _offset_khz(shape):
    """Instantaneous frequency offset (kHz) between samples, amplitude-masked."""
    env = shape.envelope()
    step = np.angle(env[1:] * np.conj(env[:-1]))          # rad per sample, wrap-safe
    amp = np.abs(env)
    keep = np.minimum(amp[1:], amp[:-1]) > 1e-3 * amp.max()
    return step[keep] / (2 * np.pi * shape.dt), step[keep], np.flatnonzero(keep)


@pytest.mark.parametrize("name", ALL_SWEPT)
def test_phase_is_not_aliased(name):
    """The bug itself: ~63 rad per sample before the fix, ~0.06 after."""
    shape = RFShape.create(name, duration=DURATION_MS, points=POINTS)
    _, step, _ = _offset_khz(shape)
    assert np.max(np.abs(step)) < np.pi / 4


@pytest.mark.parametrize("name", SINGLE_SWEEP)
def test_sweep_spans_declared_width(name):
    # 5%, not tighter: tanh/tan sweeps fastest at its zero-amplitude edges, so
    # the amplitude mask drops ~0.6 kHz there. The bug this guards was 50x.
    shape = RFShape.create(name, duration=DURATION_MS, points=POINTS)
    off, _, _ = _offset_khz(shape)
    half_khz = SWEEP_WIDTH_HZ / 2 / 1000
    assert off.max() == pytest.approx(half_khz, rel=0.05)
    assert off.min() == pytest.approx(-half_khz, rel=0.05)


@pytest.mark.parametrize("name", SINGLE_SWEEP)
def test_single_resonance_crossing_at_centre(name):
    shape = RFShape.create(name, duration=DURATION_MS, points=POINTS)
    off, _, idx = _offset_khz(shape)
    crossings = idx[np.flatnonzero(np.diff(np.sign(off)) != 0)]
    assert len(crossings) == 1
    assert crossings[0] / POINTS == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("name", ["wurst", "cagauss"])
@pytest.mark.parametrize("points", [1000, 4000])
def test_sweep_independent_of_sampling(name, points):
    """A time-unit error scales with dt, so it shows up as points-dependence."""
    shape = RFShape.create(name, duration=DURATION_MS, points=points)
    off, _, _ = _offset_khz(shape)
    assert off.max() - off.min() == pytest.approx(SWEEP_WIDTH_HZ / 1000, rel=0.02)