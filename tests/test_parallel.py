"""
parallel_map: fn(*params) for each entry, results in input order, and a
clear message when the optional joblib extra is missing.
"""

import builtins

import numpy as np
import pytest

import PULSIM
from PULSIM.parallel import parallel_map, require_joblib


def _power(base, exponent):
    return base ** exponent


def _excite(df_khz):
    """A whole independent simulation: one 90x hard pulse at one offset."""
    shape = PULSIM.RFShape.create("hard", duration=0.1, points=100)
    pulse = PULSIM.Pulse(shape, np.pi / 2, backend=PULSIM.NumpyBackend(Gamma=42.577478518))
    return pulse.apply(np.array([[0.0], [0.0], [1.0]]), np.array([df_khz]))[:, 0]


def test_missing_joblib_names_the_extra(monkeypatch):
    real_import = builtins.__import__

    def no_joblib(name, *args, **kwargs):
        if name == "joblib":
            raise ModuleNotFoundError("No module named 'joblib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_joblib)
    with pytest.raises(ModuleNotFoundError, match=r"\.\[parallel\]"):
        require_joblib()


def test_results_come_back_in_input_order():
    pytest.importorskip("joblib")
    params = [(b, e) for b in range(2, 6) for e in range(4)]
    assert parallel_map(_power, params, n_jobs=2) == [b ** e for b, e in params]


def test_matches_a_serial_loop_on_real_simulations():
    pytest.importorskip("joblib")
    offsets = [(-2.0,), (0.0,), (1.5,), (4.0,)]
    parallel = parallel_map(_excite, offsets, n_jobs=2)
    serial = [_excite(*p) for p in offsets]
    for a, b in zip(parallel, serial):
        assert np.array_equal(a, b)