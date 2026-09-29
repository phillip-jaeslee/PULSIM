"""
TanhTan against the published definition.

de Graaf & Nicolay, Concepts Magn. Reson. 9, 247 (1997), Eqs. 14-15, give
tanh/tan as a HALF passage (the BIR-4 segment, Garwood & Ke 1991), t in [0, T]:

    B1(t)  = B1max * tanh(xi * t / T)
    dw(t)  = dwmax * tan(kappa * (1 - t/T)) / tan(kappa)

with xi = 10, kappa = arctan(20). A full passage is that half passage followed
by its time reverse, which is what PULSIM builds: on x = 2t/T - 1 in [-1, 1],
amplitude tanh(xi * (1 - |x|)) and offset (SW/2) * tan(kappa * x) / tan(kappa).
The first half of PULSIM's pulse must therefore BE the published half passage
(duration T/2, dwmax = SW/2), up to the sweep direction.
"""

import numpy as np
import pytest

import PULSIM

T_MS = 2.0
POINTS = 4001                  # odd: the centre sample is exactly on resonance
SW_KHZ = 40.0                  # default sweep_width 40000 Hz
XI, TAN_KAPPA = 10.0, 20.0     # PULSIM defaults = the published values


@pytest.fixture
def shape():
    return PULSIM.RFShape.create("tanhtan", duration=T_MS, points=POINTS)


def test_defaults_are_the_published_values(shape):
    assert shape.params.get("zeta", 10.0) == XI
    assert shape.params.get("tan_kappa", 20.0) == TAN_KAPPA


def test_first_half_amplitude_is_published_tanh(shape):
    half = POINTS // 2 + 1
    t_over_T = np.linspace(0.0, 1.0, half)           # time within the half passage
    amp = np.abs(shape.envelope()[:half])
    # compared unnormalized: the peak is tanh(xi) = 0.999999996, not 1
    assert amp == pytest.approx(np.tanh(XI * t_over_T), abs=1e-12)


def test_first_half_frequency_is_published_tan(shape):
    env = shape.envelope()
    half = POINTS // 2 + 1
    # from sample 1 on: sample 0 has zero amplitude, so its phase is undefined
    offset_khz = np.angle(env[2:half] * np.conj(env[1:half - 1])) / (2 * np.pi * shape.dt)
    t_over_T = np.linspace(0.0, 1.0, half)[2:]       # step k -> k+1 carries offset[k+1]
    kappa = np.arctan(TAN_KAPPA)
    published = (SW_KHZ / 2) * np.tan(kappa * (1 - t_over_T)) / TAN_KAPPA
    # PULSIM sweeps -SW/2 -> +SW/2; the published half passage starts at +dwmax.
    assert -offset_khz == pytest.approx(published, abs=1e-9)