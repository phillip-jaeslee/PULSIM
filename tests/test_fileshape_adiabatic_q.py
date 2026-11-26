"""
An imported adiabatic waveform calibrates from Q -- but only when the caller
supplies Q, because a shape file never stores it (audit Part 5, section 5.7).

No vendor file is needed: the test writes its own waveform, PULSIM's analytic
HypSec, to a temporary file in the same amplitude/phase column format the
importer reads, then checks the imported pulse against HypSec's closed form.
"""

import numpy as np
import pytest

import PULSIM
from PULSIM.rf_shape import FileShape

GAMMA = 42.577478518        # kHz/mT, 1H
T_MS = 2.0


@pytest.fixture
def hypsec_file(tmp_path):
    """HypSec written as a shape file: amplitude in percent, phase in degrees.

    Deliberately no header fields, so intent comes from the caller -- this
    tests the calibration, not the header parser (test_bruker covers that)."""
    env = PULSIM.RFShape.create("hypsec", duration=T_MS, points=1001).envelope()
    amp = np.abs(env) / np.abs(env).max() * 100.0
    phase = (-np.degrees(np.angle(env))) % 360.0     # FileShape uses exp(-i*phase)
    path = tmp_path / "hypsec_test.jhl"
    lines = ["##TITLE= PULSIM test waveform (analytic HypSec)",
             f"##NPOINTS= {len(env)}", "##XYPOINTS= (XY..XY)"]
    lines += [f"{a:.6e}, {p:.6e}" for a, p in zip(amp, phase)]
    lines.append("##END=")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_imported_adiabatic_file_calibrates_from_given_q(hypsec_file):
    shape = FileShape(path=str(hypsec_file), duration=T_MS, intent="adiabatic", q_mid=5.0)
    pulse = PULSIM.Pulse(shape, backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    analytic = PULSIM.RFShape.create("hypsec", duration=T_MS).calibration.nu1_for(T_MS)
    assert pulse.nu1_max == pytest.approx(analytic, rel=1e-4)
    assert pulse.realized_q == pytest.approx(5.0, rel=1e-9)


def test_imported_adiabatic_file_without_q_refuses_and_names_q_mid(hypsec_file):
    shape = FileShape(path=str(hypsec_file), duration=T_MS, intent="adiabatic")
    assert shape.q_mid is None                       # no silent default of 5
    with pytest.raises(NotImplementedError, match="q_mid"):
        shape.calibration


def test_explicit_nu1_max_still_works_and_q_stays_unknown(hypsec_file):
    shape = FileShape(path=str(hypsec_file), duration=T_MS, intent="adiabatic")
    pulse = PULSIM.Pulse(shape, nu1_max=4.0, backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    assert pulse.nu1_max == 4.0
    assert pulse.realized_q is None