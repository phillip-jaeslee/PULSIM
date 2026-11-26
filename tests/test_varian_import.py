"""
Varian/Agilent VnmrJ shape files (*.RF) read through VarianShape.

The fixtures in tests/fixtures/varian/ are copied unchanged from OpenVnmrJ
(https://github.com/OpenVnmrJ/OpenVnmrJ, Apache-2.0; see the LICENSE file
next to them), so these tests run from a clean clone -- no skipif.

    eburp1.RF     amplitude-only; negative lobes stored as phase 180
    wurstIR.RF    duration counts of 2
    rna_S888g.RF  gate = 0 rows at full amplitude
    pbx.RF        a zero-duration row that only sets the full scale
    expPuls.RF    full scale 4095 instead of 1023
    square.RF     a single row

The expected values come from the files' rows, parsed here independently
of VarianShape, and from VnmrJ's own reader (OpenVnmrJ src/psg/wg.c).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

from PULSIM.rf_shape import RFShape, VarianShape

FIXTURES = _here / "fixtures" / "varian"
ALL = ["eburp1.RF", "wurstIR.RF", "rna_S888g.RF", "pbx.RF", "expPuls.RF", "square.RF"]


def _rows(name):
    with open(FIXTURES / name) as handle:
        return [[float(v) for v in line.split()]
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")]


def _load(name):
    return RFShape.create("varian", path=str(FIXTURES / name), duration=1.0)


@pytest.mark.parametrize("name", ALL)
def test_fixture_is_committed(name):
    assert (FIXTURES / name).is_file()


@pytest.mark.parametrize("name", ALL)
def test_every_fixture_builds(name):
    shape = _load(name)
    assert isinstance(shape, VarianShape)
    assert shape.intent is None                 # no SHAPE_* header in a .RF file
    assert np.all(np.isfinite(shape._build()))


# ------------------------------------------------------------- eburp1.RF
def test_columns_are_swapped_into_amplitude_phase_order():
    rows = _rows("eburp1.RF")
    shape = _load("eburp1.RF")
    np.testing.assert_allclose(shape.xy[:, 0], [r[1] / 1023 * 100 for r in rows])
    np.testing.assert_allclose(shape.xy[:, 1], [r[0] for r in rows])


def test_phase_180_rows_become_negative_lobes():
    shape = _load("eburp1.RF")
    wave = shape._build()
    negative = shape.xy[:, 1] == 180.0
    assert negative.any()
    assert np.all(wave.real[negative] <= 0)
    assert np.all(wave.real[~negative] >= 0)
    np.testing.assert_allclose(wave.imag, 0, atol=1e-9)


def test_matches_the_analytic_eburp1():
    """Same pulse from two sources: the published coefficients (EBurp1) and
    VnmrJ's tabulated file. The file keeps three decimals, so allow 3 %."""
    file_wave = _load("eburp1.RF")._build().real
    analytic = RFShape.create("eburp1", duration=1.0, points=len(file_wave))._build().real
    file_wave = file_wave / np.abs(file_wave).max()
    analytic = analytic / np.abs(analytic).max()
    assert np.abs(file_wave - analytic).max() < 0.03


# ------------------------------------------------------------ wurstIR.RF
def test_duration_counts_expand_into_equal_steps():
    rows = _rows("wurstIR.RF")
    assert any(r[2] == 2 for r in rows)            # the case being tested
    assert _load("wurstIR.RF").points == sum(int(r[2]) for r in rows)


# ---------------------------------------------------------- rna_S888g.RF
def test_gated_off_rows_take_time_at_zero_amplitude():
    rows = _rows("rna_S888g.RF")
    gated = [len(r) > 3 and r[3] == 0 for r in rows]
    assert any(g and r[1] > 0 for g, r in zip(gated, rows))   # the case being tested
    shape = _load("rna_S888g.RF")
    assert shape.points == len(rows)               # gated rows are kept, not dropped
    np.testing.assert_array_equal(shape.xy[gated, 0], 0.0)
    ungated = [not g for g in gated]
    np.testing.assert_allclose(shape.xy[ungated, 0],
                               [r[1] / 1023 * 100 for r, u in zip(rows, ungated) if u])


# ---------------------------------------------------------------- pbx.RF
def test_zero_duration_row_sets_full_scale_but_is_not_a_step():
    rows = _rows("pbx.RF")
    scaling = [r for r in rows if r[2] == 0]
    steps = [r for r in rows if r[2] > 0]
    assert scaling and max(r[1] for r in scaling) > max(r[1] for r in steps)
    shape = _load("pbx.RF")
    assert shape.points == sum(int(r[2]) for r in steps)
    full = max(r[1] for r in rows)
    assert shape.xy[:, 0].max() == pytest.approx(max(r[1] for r in steps) / full * 100)
    assert shape.xy[:, 0].max() < 100.0


# ------------------------------------------------------------ expPuls.RF
def test_full_scale_is_the_file_maximum_not_1023():
    assert max(r[1] for r in _rows("expPuls.RF")) == 4095
    assert _load("expPuls.RF").xy[:, 0].max() == pytest.approx(100.0)


# ------------------------------------------------------------- square.RF
def test_single_row_file_becomes_a_two_point_constant():
    shape = _load("square.RF")
    assert shape.points == 2
    np.testing.assert_allclose(shape.xy, [[100.0, 0.0], [100.0, 0.0]])
