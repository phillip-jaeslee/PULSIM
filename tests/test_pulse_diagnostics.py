"""
pulse_diagnostics: the first-order account of J evolution during a pulse,
held to closed forms and to the exact propagator, and the delay optimizer.

The decisive check is scaling. If the returned vector c is the right
first-order generator, then U_exact - U_rf(T) exp(-i kappa (c.I) Sz) is
second order in J: ten times more J gives a hundred times more error. A
vector that is wrong at first order leaves an error that grows only tenfold.
"""

import numpy as np
import pytest
from scipy.linalg import expm

import PULSIM
from PULSIM.liouville import RawShapedPulseSegment, _axis_phase, _static_hamiltonian
from PULSIM.pulse_diagnostics import effective_coupling_generator, optimize_delay
from PULSIM.spin_operators import Ix, Iy, Iz, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio

GAMMA = gyro_ratio('H')
T_MS = 1.5


def _pulse(name, flip=np.pi / 2, axis="x", points=500, **kw):
    shape = PULSIM.RFShape.create(name, duration=T_MS, points=points, **kw)
    return PULSIM.Pulse(shape, flip, axis=axis, backend=PULSIM.NumpyBackend(Gamma=GAMMA))


def _rf(pulse):
    """RF in rad/ms, exactly as ShapePulseSegment feeds the Hamiltonian."""
    return (2 * np.pi * GAMMA * pulse.calibrated_rf() * np.exp(1j * _axis_phase(pulse.axis)),
            pulse.shape.dt)


def _propagator(rf, dt, J_hz):
    ss = SpinSystem(nuclei=['H', '15N'], offsets=[0.0, 0.0], couplings={(0, 1): J_hz})
    U = np.eye(4, dtype=complex)
    for H, step in RawShapedPulseSegment(rf, dt, channel='H').hamiltonians(ss):
        U = expm(-1j * H * step) @ U
    return U, ss


def _first_order_error(rf, dt, J_hz, side):
    U_exact, ss = _propagator(rf, dt, J_hz)
    U_rf, _ = _propagator(rf, dt, 0.0)
    B = product_operator(Iz(), 0, Iz(), 1, 2)
    H_J = _static_hamiltonian(ss, include_offset=False)
    kappa = np.trace(H_J @ B).real / np.trace(B @ B).real
    ops = [product_operator(Ix(), 0, Iz(), 1, 2), product_operator(Iy(), 0, Iz(), 1, 2), B]
    c = effective_coupling_generator(rf, dt, side=side)
    E = expm(-1j * kappa * sum(ci * op for ci, op in zip(c, ops)))
    U_pred = U_rf @ E if side == "before" else E @ U_rf
    return np.abs(U_pred - U_exact).max()


def test_no_rf_is_a_plain_delay():
    c = effective_coupling_generator(np.zeros(100, dtype=complex), 0.01)
    assert c == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


@pytest.mark.parametrize("flip", [np.pi / 2, np.pi])
def test_rectangular_pulse_closed_form(flip):
    """Toggling-frame Iz under a constant x pulse is Iz cos(wt) + Iy sin(wt)."""
    rf, dt = _rf(_pulse("hard", flip=flip, points=2000))
    w = flip / T_MS
    expected = (0.0, (1 - np.cos(flip)) / w, np.sin(flip) / w)
    assert effective_coupling_generator(rf, dt) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("name, kw", [
    ("hard", {}),
    ("eburp2", {}),
    ("hypsec", {"points": 1500, "flip": None}),     # phase-modulated
    ("wurst", {"points": 3000, "flip": None}),      # phase-modulated
])
@pytest.mark.parametrize("side", ["before", "after"])
def test_first_order_in_J_against_exact_propagator(name, kw, side):
    # 10 -> 100 Hz, not 1 -> 10: at 1 Hz the error reaches a ~1e-7 floor set
    # by the trapezoid rule, not by the physics, and the ratio stops meaning much
    rf, dt = _rf(_pulse(name, **kw))
    e10, e100 = _first_order_error(rf, dt, 10.0, side), _first_order_error(rf, dt, 100.0, side)
    assert e10 < 2e-4
    assert 50 < e100 / e10 < 200                    # second order: ~100x


def test_flip_back_pulse_carries_its_delay_after_itself():
    """eb2try is eb2x time-reversed: the delay eb2x carries before itself,
    eb2try carries after itself (the E-BURP-2 refocusing element)."""
    eb2x = _rf(_pulse("eburp2"))
    eb2try = _rf(_pulse("eburp2", axis="y", time_reversed=True))
    before_x = effective_coupling_generator(*eb2x, side="before")[2]
    after_try = effective_coupling_generator(*eb2try, side="after")[2]
    assert before_x == pytest.approx(1.0772, abs=1e-4)
    assert after_try == pytest.approx(before_x, abs=1e-9)


def test_unknown_side_is_refused():
    with pytest.raises(ValueError, match="side"):
        effective_coupling_generator(np.zeros(4, dtype=complex), 0.1, side="middle")


def test_optimize_delay_finds_an_interior_optimum():
    result = optimize_delay(lambda d: d, lambda d: -(d - 1.3) ** 2, bounds=(0.5, 2.0))
    assert result.x == pytest.approx(1.3, abs=1e-4)


def test_optimize_delay_refuses_a_bound_limited_answer():
    """The bug the delay tutorial had: the optimum outside the search range."""
    with pytest.raises(ValueError, match="bound"):
        optimize_delay(lambda d: d, lambda d: -(d - 3.0) ** 2, bounds=(0.5, 2.0))