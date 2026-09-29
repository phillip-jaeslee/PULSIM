"""
time_reversed=True: any shape played backwards (e.g. E-BURP-2 "tr", the
flip-back partner of an excitation pulse).

Checked by what reversal must and must not change: the envelope is exactly
reversed; the area calibration is unchanged; an adiabatic half passage moves
its resonance crossing from the end to the start; and on resonance, a pulse
followed by its time-reversed copy with inverted phase is the identity.
"""

import numpy as np
import pytest

import PULSIM
from PULSIM.calibration import resonance_crossing
from PULSIM.liouville import LiouvilleSequence, ShapePulseSegment
from PULSIM.rf_shape import FileShape
from PULSIM.spin_operators import Iz, embed
from PULSIM.spin_system import SpinSystem, gyro_ratio

T_MS = 1.5
FIXTURE = "tests/fixtures/waveforms/pulsim_sine.jhl"


def _shape(name, **kw):
    return PULSIM.RFShape.create(name, duration=T_MS, points=500, **kw)


@pytest.mark.parametrize("name", ["eburp2", "hypsec", "wurst"])
def test_envelope_is_exactly_reversed(name):
    assert np.array_equal(_shape(name, time_reversed=True).envelope(),
                          _shape(name).envelope()[::-1])


def test_area_calibration_is_unchanged():
    backend = PULSIM.NumpyBackend(Gamma=gyro_ratio('H'))
    fwd = PULSIM.Pulse(_shape("eburp2"), np.pi / 2, backend=backend).nu1_max
    rev = PULSIM.Pulse(_shape("eburp2", time_reversed=True), np.pi / 2, backend=backend).nu1_max
    assert rev == pytest.approx(fwd, rel=1e-12)


def test_reversed_half_passage_crosses_resonance_at_the_start():
    shape = _shape("hypsec", passage="half", time_reversed=True)
    assert resonance_crossing(shape.envelope()).u == 0.0
    assert shape.calibration.nu1_for(T_MS) == _shape("hypsec", passage="half").calibration.nu1_for(T_MS)


def test_file_shape_accepts_it():
    fwd = FileShape(path=FIXTURE, duration=1.0).envelope()
    assert np.array_equal(FileShape(path=FIXTURE, duration=1.0, time_reversed=True).envelope(), fwd[::-1])


def test_pulse_then_reversed_inverted_copy_is_identity_on_resonance():
    """On resonance H_rev(t) = -H(T - t), so U_rev = U^dagger exactly."""
    backend = PULSIM.NumpyBackend(Gamma=gyro_ratio('H'))
    fwd = PULSIM.Pulse(_shape("eburp2"), np.pi / 2, axis="x", backend=backend)
    rev = PULSIM.Pulse(_shape("eburp2", time_reversed=True), np.pi / 2, axis="-x", backend=backend)
    Z = embed(Iz(), 0, 1)
    ss = SpinSystem(nuclei=['H'], offsets=[0.0], couplings={})
    mid = LiouvilleSequence([ShapePulseSegment(fwd)], ss).propagate(Z)
    end = LiouvilleSequence([ShapePulseSegment(fwd), ShapePulseSegment(rev)], ss).propagate(Z)
    assert abs(np.trace(mid @ Z).real) < 1e-6                  # genuinely excited in between
    assert np.allclose(end, Z, atol=1e-9)