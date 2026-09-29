"""
Flip angle on an adiabatic pulse (audit Part 5, section 5.2).

flip never scales the RF of an adiabatic pulse; it records the intended
operation. PULSIM therefore stays SILENT when flip matches the shape's nominal
operation (180 degrees for a full passage, 90 for a half passage) and warns
only on a mismatch, or when the operation is unknown. A warning that fires on
every adiabatic pulse teaches users to ignore it.
"""

import warnings

import numpy as np
import pytest

import PULSIM
from PULSIM.rf_shape import FileShape

GAMMA = 42.577478518        # kHz/mT, 1H
T_MS = 2.0


def _pulse(name, flip, **kw):
    shape = PULSIM.RFShape.create(name, duration=T_MS, points=1000, **kw)
    return PULSIM.Pulse(shape, flip=flip, backend=PULSIM.NumpyBackend(Gamma=GAMMA))


def _silent(pulse):
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        return pulse.nu1_max


@pytest.mark.parametrize("name, kw, flip", [
    ("hypsec", {}, np.pi),
    ("hypsec", {"passage": "half"}, np.pi / 2),
    ("sincos", {}, np.pi),
    ("sincos", {"full_passage": False}, np.pi / 2),
    ("wurst", {}, np.pi),
])
def test_matching_flip_is_silent(name, kw, flip):
    _silent(_pulse(name, flip, **kw))


@pytest.mark.parametrize("name, kw, flip, nominal", [
    ("hypsec", {}, np.pi / 2, "180"),
    ("hypsec", {"passage": "half"}, np.pi, "90"),
    ("wurst", {}, np.pi / 2, "180"),
])
def test_mismatched_flip_warns_and_names_the_nominal_operation(name, kw, flip, nominal):
    with pytest.warns(UserWarning, match=f"designed as a {nominal}-degree operation"):
        _pulse(name, flip, **kw).nu1_max


def test_flip_never_changes_the_amplitude():
    silent = _silent(_pulse("hypsec", np.pi))
    with pytest.warns(UserWarning):
        mismatched = _pulse("hypsec", np.pi / 2).nu1_max
    assert silent == mismatched == _pulse("hypsec", None).nu1_max


def test_area_shapes_have_no_nominal_flip():
    assert PULSIM.RFShape.create("gausscasq5", duration=T_MS).nominal_flip is None


def test_file_without_declared_rotation_warns_as_unknown(tmp_path):
    """No SHAPE_TOTROT in the file: the operation cannot be checked, so say so."""
    path = tmp_path / "plain.jhl"
    env = PULSIM.RFShape.create("hypsec", duration=T_MS, points=501).envelope()
    rows = [f"{abs(e) / np.abs(env).max() * 100:.6e}, {(-np.degrees(np.angle(e))) % 360:.6e}"
            for e in env]
    path.write_text("##TITLE= t\n##NPOINTS= 501\n##XYPOINTS= (XY..XY)\n"
                    + "\n".join(rows) + "\n##END=\n")
    shape = FileShape(path=str(path), duration=T_MS, intent="adiabatic", q_mid=5.0)
    assert shape.nominal_flip is None
    pulse = PULSIM.Pulse(shape, flip=np.pi, backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    with pytest.warns(UserWarning, match="not declared"):
        pulse.nu1_max