"""
File-import tests that run from a clean clone.

The rest of the file-import suite is written against wave/, which is vendor
material excluded from the repository, so those tests skip for anyone outside
the group. These do not skip: they run against
tests/fixtures/waveforms/pulsim_sine.jhl, a waveform this project generates
from a closed-form expression (tools/generate_test_waveforms.py).

That makes the parser reproducibly tested without redistributing anything.
It does NOT cover the 203-shape corpus: the census and parameter agreement in
PHYSICS_SPECIFICATION section 9 still rest on wave/ and remain a separate
reproducibility task.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.rf_shape import RFShape
from PULSIM.spin_system import gyro_ratio

REPO = _here.parent
FIXTURE = _here / "fixtures" / "waveforms" / "pulsim_sine.jhl"
GENERATOR = REPO / "tools" / "generate_test_waveforms.py"
N_POINTS = 513
PI = np.pi


def _closed_form(n=N_POINTS):
    k = np.arange(n, dtype=float)
    return np.sin(PI * k / (n - 1)) * 100.0


# ------------------------------------------------------- it is actually here
def test_fixture_is_committed():
    """The point of the exercise: no skipif on this file."""
    assert FIXTURE.is_file(), f"{FIXTURE} is missing -- run tools/generate_test_waveforms.py"


def test_fixture_matches_its_generator():
    """Catches a hand-edited fixture, which would make the documented
    expression a lie."""
    result = subprocess.run([sys.executable, str(GENERATOR), "--check"],
                            capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stdout + result.stderr


# ------------------------------------------------------------ what it reads
def test_point_count_comes_from_the_file():
    shape = RFShape.create("file", path=str(FIXTURE), duration=1.0)
    assert shape.points == N_POINTS


def test_passing_points_is_an_error():
    with pytest.raises(TypeError):
        RFShape.create("file", path=str(FIXTURE), duration=1.0, points=256)


def test_amplitudes_match_the_documented_expression():
    shape = RFShape.create("file", path=str(FIXTURE), duration=1.0)
    # the file stores 6 significant figures, so compare at text precision
    assert shape.xy[:, 0] == pytest.approx(_closed_form(), abs=1e-4)


def test_phase_column_is_all_zero():
    shape = RFShape.create("file", path=str(FIXTURE), duration=1.0)
    assert np.abs(shape.xy[:, 1]).max() == 0.0


def test_envelope_is_symmetric_and_peaks_at_the_centre():
    shape = RFShape.create("file", path=str(FIXTURE), duration=1.0)
    amplitude = shape.xy[:, 0]
    assert amplitude[0] == pytest.approx(0.0, abs=1e-9)
    assert amplitude[(N_POINTS - 1) // 2] == pytest.approx(100.0, abs=1e-9)
    assert amplitude == pytest.approx(amplitude[::-1], abs=1e-9)


# ----------------------------------------------- it carries no vendor traces
def test_header_is_empty_so_calibration_falls_back_to_area():
    """No vendor header keys, by design: the shape must get no intent from the
    file and calibrate by signed area."""
    shape = RFShape.create("file", path=str(FIXTURE), duration=1.0)
    assert not shape.header.scalars and not shape.header.arrays
    assert shape.intent is None
    assert shape.calibration_mode == "area"


def test_fixture_contains_no_vendor_private_keys():
    """Bruker shape files carry ##$KEY= private fields. This file must have
    none -- both because it is not one, and because copying them is exactly
    what this fixture exists to avoid."""
    text = FIXTURE.read_text()
    assert "##$" not in text
    for key in ("SHAPE_EXMODE", "SHAPE_TYPE", "SHAPE_BWFAC", "SHAPE_INTEGFAC",
                "SHAPE_MODE", "SHAPE_TOTROT"):
        assert key not in text


def test_provenance_notice_is_present():
    """It should not be possible to drop the notice without a test noticing."""
    text = FIXTURE.read_text()
    for phrase in ("independently generated",
                   "no waveform samples or metadata",
                   "trademarks of",
                   "not affiliated with or endorsed by"):
        assert phrase in text, f"provenance notice is missing: {phrase!r}"


# ------------------------------------------------------------- it simulates
def test_resampling_changes_the_point_count():
    for target in (128, 200, 1000):
        shape = RFShape.create("file", path=str(FIXTURE), duration=1.0,
                               resample_to=target)
        assert shape.points == target


def test_resampling_preserves_the_envelope_shape():
    coarse = RFShape.create("file", path=str(FIXTURE), duration=1.0,
                            resample_to=257)
    amplitude = coarse.xy[:, 0]
    assert amplitude.max() == pytest.approx(100.0, rel=1e-3)
    assert amplitude[0] == pytest.approx(0.0, abs=1e-3)


def test_a_pulse_built_from_the_fixture_obeys_the_sign_convention():
    """Section 1.8 of PHYSICS_SPECIFICATION: a 90x on +Mz gives -My."""
    shape = RFShape.create("file", path=str(FIXTURE), duration=1.0)
    pulse = Pulse(shape, PI / 2, axis="x",
                  backend=NumpyBackend(Gamma=gyro_ratio("H")))
    M = pulse.apply(np.array([[0.0], [0.0], [1.0]]), np.array([0.0]))[:, 0]
    assert M[0] == pytest.approx(0.0, abs=1e-9)
    assert M[1] == pytest.approx(-1.0, abs=1e-6)
    assert M[2] == pytest.approx(0.0, abs=1e-6)
