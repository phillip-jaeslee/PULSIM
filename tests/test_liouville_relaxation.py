"""
Analytic and cross-formalism tests for phenomenological relaxation in
Liouville space (PULSIM/relaxation.py).

Nothing here is a golden test. Every assertion is against a closed-form
solution, against the Bloch backend that already implements the same physics
for an uncoupled spin, or against a convergence rate the numerical scheme is
required to have -- so these can fail even when PULSIM reproduces its own
past output exactly.

The two claims worth stating up front, because both were wrong at some point
while this was being written:

  * with no Relaxation attached, propagation must be BIT FOR BIT what it was
    before relaxation existed -- not approximately, not to 1e-15;
  * relaxation and coherent evolution do NOT commute once the spins are
    coupled, because evolution moves amplitude between operators of different
    rate (Ix relaxes at 1/T2, the 2IySz it evolves into at 1/T2 + 1/T1). A
    Delay is a single step however long it is, so it has to be subdivided.
    Equal T1 and T2 does not rescue this: the two-spin rate is a sum either
    way. test_long_delay_needs_subdivision is that bug, nailed down.
"""

import os
import sys

import numpy as np
import pytest

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))

from PULSIM.backend import NumpyBackend
from PULSIM.bloch import bloch_delay
from PULSIM.liouville import (Delay, IdealPulse, LiouvilleSequence,
                              ShapePulseSegment)
from PULSIM.pulse_oo import Pulse
from PULSIM.relaxation import Relaxation
from PULSIM.rf_shape import RFShape
from PULSIM.spin_operators import SpinOperators
from PULSIM.spin_system import SpinSystem, gyro_ratio

PI = np.pi
GAMMA_H = gyro_ratio("H")


# ---------------------------------------------------------------- helpers
def _one_spin(offset_khz=0.0):
    ss = SpinSystem(nuclei=["H"], offsets=[2 * PI * offset_khz])
    return ss, SpinOperators(ss)


def _two_spins(J=140.0, off_I=0.0, off_S=0.0):
    couplings = {(0, 1): J} if J else None
    ss = SpinSystem(nuclei=["H", "13C"], offsets=[off_I, off_S], couplings=couplings)
    return ss, SpinOperators(ss)


def _inept(J=140.0):
    delta = 1.0 / (4 * J / 1000.0)
    return [IdealPulse(PI / 2, phase="x", channel="H"),
            Delay(delta),
            IdealPulse(PI, phase="x", channel="H"),
            IdealPulse(PI, phase="x", channel="13C"),
            Delay(delta),
            IdealPulse(PI / 2, phase="y", channel="H"),
            IdealPulse(PI / 2, phase="x", channel="13C")]


# ------------------------------------------------- the no-op must be a no-op
def test_no_relaxation_argument_is_bit_for_bit_unchanged():
    ss, ops = _two_spins(off_I=1.1, off_S=-0.7)
    sigma0 = ops["Iz"].astype(complex)
    without = LiouvilleSequence(_inept(), ss).propagate(sigma0)
    empty = LiouvilleSequence(_inept(), ss, relaxation=Relaxation()).propagate(sigma0)
    assert np.array_equal(without, empty), "an empty Relaxation must not perturb a single bit"


def test_relaxation_with_only_one_time_still_relaxes():
    """T1 alone or T2 alone is not 'no relaxation'."""
    ss, ops = _one_spin()
    start = ops["Ix"].astype(complex)
    plain = LiouvilleSequence([Delay(20.0)], ss).propagate(start)
    t2_only = LiouvilleSequence([Delay(20.0)], ss,
                                relaxation=Relaxation(T2=30.0)).propagate(start)
    assert not np.allclose(plain, t2_only)


# ------------------------------------------------------------ closed forms
@pytest.mark.parametrize("t", [0.0, 10.0, 50.0, 150.0])
def test_t2_decay_matches_closed_form(t):
    T2 = 50.0
    ss, ops = _one_spin()
    sigma = LiouvilleSequence([Delay(t)], ss,
                              relaxation=Relaxation(T2=T2)).propagate(ops["Ix"].astype(complex))
    assert ops.expectation(sigma, "Ix") == pytest.approx(np.exp(-t / T2), abs=1e-12)


@pytest.mark.parametrize("t", [0.0, 50.0, 200.0, 600.0])
def test_t1_recovery_from_saturation_matches_closed_form(t):
    T1, M0 = 200.0, 1.0
    ss, ops = _one_spin()
    saturated = np.zeros((2, 2), dtype=complex)
    sigma = LiouvilleSequence([Delay(t)], ss,
                              relaxation=Relaxation(T1=T1, M0=M0)).propagate(saturated)
    assert ops.expectation(sigma, "Iz") == pytest.approx(M0 * (1 - np.exp(-t / T1)), abs=1e-12)


def test_inversion_recovery_null_at_t1_ln2():
    """The one number an NMR reader checks by eye, same as the Bloch test."""
    T1, M0 = 800.0, 1.0
    ss, ops = _one_spin()
    inverted = (-ops["Iz"]).astype(complex)
    sigma = LiouvilleSequence([Delay(T1 * np.log(2.0))], ss,
                              relaxation=Relaxation(T1=T1, M0=M0)).propagate(inverted)
    assert ops.expectation(sigma, "Iz") == pytest.approx(0.0, abs=1e-12)


def test_t1_and_t2_stay_in_their_own_lanes():
    ss, ops = _one_spin()
    start = (ops["Ix"] + ops["Iz"]).astype(complex)
    only_t2 = LiouvilleSequence([Delay(5.0)], ss,
                                relaxation=Relaxation(T2=10.0)).propagate(start)
    only_t1 = LiouvilleSequence([Delay(5.0)], ss,
                                relaxation=Relaxation(T1=10.0, M0=0.0)).propagate(start)
    assert ops.expectation(only_t2, "Iz") == pytest.approx(1.0, abs=1e-12)   # T2 leaves z alone
    assert ops.expectation(only_t1, "Ix") == pytest.approx(1.0, abs=1e-12)   # T1 leaves xy alone


def test_multi_spin_operator_rate_is_the_sum_of_its_factors():
    """The defining claim of the model: 2Iz(I)Iy(S) carries 1/T1 + 1/T2.

    Run with no offsets and no coupling so the Hamiltonian is exactly zero and
    only relaxation acts -- otherwise this would be testing two things at once.
    """
    T1, T2, t = 300.0, 60.0, 40.0
    ss, ops = _two_spins(J=0.0)
    sigma = LiouvilleSequence([Delay(t)], ss,
                              relaxation=Relaxation(T1=T1, T2=T2, M0=0.0)
                              ).propagate(ops["IzSy"].astype(complex))
    want = np.exp(-t * (1.0 / T1 + 1.0 / T2))
    assert ops.expectation(sigma, "IzSy") == pytest.approx(want, abs=1e-12)


def test_identity_component_is_untouched_so_trace_is_preserved():
    ss, ops = _two_spins()
    sigma0 = (ops["Iz"] + 0.5 * np.eye(4)).astype(complex)
    sigma = LiouvilleSequence([Delay(25.0)], ss,
                              relaxation=Relaxation(T1=200.0, T2=40.0)).propagate(sigma0)
    assert np.trace(sigma).real == pytest.approx(np.trace(sigma0).real, abs=1e-12)


def test_long_time_limit_is_equilibrium():
    T1, M0 = 100.0, 0.8
    ss, ops = _one_spin()
    sigma = LiouvilleSequence([Delay(50 * T1)], ss,
                              relaxation=Relaxation(T1=T1, T2=10.0, M0=M0)
                              ).propagate(ops["Ix"].astype(complex))
    assert ops.expectation(sigma, "Iz") == pytest.approx(M0, abs=1e-9)
    assert ops.expectation(sigma, "Ix") == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------- cross-formalism
@pytest.mark.parametrize("offset_khz", [0.0, 0.15, -0.3])
def test_matches_bloch_backend_for_one_uncoupled_spin(offset_khz):
    """A single spin is where the two formalisms must agree exactly. The Bloch
    backend's relaxation is the reference implementation; this is the whole
    justification for the Liouville version's conventions."""
    T1, T2 = 300.0, 80.0
    shape = RFShape.create("gausscasq5", duration=2.0, points=400)

    M = Pulse(shape, PI / 2, axis="x",
              backend=NumpyBackend(Gamma=GAMMA_H, T1=T1, T2=T2)
              ).apply(np.array([[0.0], [0.0], [1.0]]), np.array([offset_khz]))
    M = bloch_delay(M, 20.0, np.array([offset_khz]), GAMMA_H, T1, T2, 1.0)[:, 0]

    ss, ops = _one_spin(offset_khz)
    segments = [ShapePulseSegment(Pulse(shape, PI / 2, axis="x",
                                        backend=NumpyBackend(Gamma=GAMMA_H))),
                Delay(20.0)]
    sigma = LiouvilleSequence(segments, ss,
                              relaxation=Relaxation(T1=T1, T2=T2)).propagate(ops["Iz"].astype(complex))
    got = np.array([ops.expectation(sigma, k) for k in ("Ix", "Iy", "Iz")])
    assert got == pytest.approx(M, abs=1e-11)


# ------------------------------------------------------- numerical scheme
def test_strang_splitting_is_second_order():
    """Halving the step must quarter the error. The shape is held fixed and
    only the propagation step is subdivided, so this measures the splitting
    and not the shape discretization."""
    ss, ops = _one_spin(0.2)
    relax = Relaxation(T1=300.0, T2=80.0, max_step=None)
    shape = RFShape.create("gausscasq5", duration=2.0, points=200)
    pulse = Pulse(shape, PI / 2, axis="x", backend=NumpyBackend(Gamma=GAMMA_H))

    def run(n_sub):
        r = Relaxation(T1=300.0, T2=80.0, max_step=shape.dt / n_sub)
        sigma = LiouvilleSequence([ShapePulseSegment(pulse)], ss, relaxation=r
                                  ).propagate(ops["Iz"].astype(complex))
        return ops.expectation(sigma, "Iy")

    reference = run(64)
    errors = [abs(run(m) - reference) for m in (1, 2, 4, 8)]
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.25)


def test_uncoupled_delay_commutes_so_one_step_is_enough():
    """With no coupling, evolution stays inside a rate class and the split is
    exact -- which is why the Bloch side never needed subdivision."""
    ss, ops = _one_spin(0.3)
    relax = lambda ms: Relaxation(T1=400.0, T2=70.0, max_step=ms)
    start = ops["Ix"].astype(complex)
    coarse = LiouvilleSequence([Delay(30.0)], ss, relaxation=relax(None)).propagate(start)
    fine = LiouvilleSequence([Delay(30.0)], ss, relaxation=relax(0.01)).propagate(start)
    assert np.abs(coarse - fine).max() < 1e-12


def test_long_delay_needs_subdivision_when_spins_are_coupled():
    """The bug this parameter exists for: J-coupling moves amplitude between
    operators of different rate, so one 30 ms step is visibly wrong."""
    ss, ops = _two_spins(J=140.0, off_I=2 * PI * 0.3, off_S=-2 * PI * 0.2)
    start = ops["Ix"].astype(complex)

    def run(max_step):
        return LiouvilleSequence([Delay(30.0)], ss,
                                 relaxation=Relaxation(T1=400.0, T2=70.0, max_step=max_step)
                                 ).propagate(start)

    reference = run(0.005)
    assert np.abs(run(None) - reference).max() > 1e-3      # one step: wrong
    assert np.abs(run(0.05) - reference).max() < 1e-6      # the default: fine


def test_equal_t1_and_t2_does_not_make_it_commute():
    """Isotropic relaxation commutes with rotation on the Bloch sphere, so the
    natural guess is that T1 == T2 removes the commutator here too. It does
    not -- a two-spin operator's rate is a sum either way."""
    ss, ops = _two_spins(J=140.0, off_I=2 * PI * 0.3, off_S=-2 * PI * 0.2)
    start = ops["Ix"].astype(complex)
    one_step = LiouvilleSequence([Delay(30.0)], ss,
                                 relaxation=Relaxation(T1=70.0, T2=70.0, max_step=None)).propagate(start)
    fine = LiouvilleSequence([Delay(30.0)], ss,
                             relaxation=Relaxation(T1=70.0, T2=70.0, max_step=0.005)).propagate(start)
    assert np.abs(one_step - fine).max() > 1e-3


# --------------------------------------------------------------- the API
def test_times_may_be_given_per_spin():
    ss, ops = _two_spins(J=0.0)
    t = 30.0
    sigma = LiouvilleSequence([Delay(t)], ss,
                              relaxation=Relaxation(T2=[20.0, 80.0], M0=0.0)
                              ).propagate((ops["Ix"] + ops["Sx"]).astype(complex))
    assert ops.expectation(sigma, "Ix") == pytest.approx(np.exp(-t / 20.0), abs=1e-12)
    assert ops.expectation(sigma, "Sx") == pytest.approx(np.exp(-t / 80.0), abs=1e-12)


@pytest.mark.parametrize("kwargs", [{"T1": 0.0}, {"T2": -5.0}, {"T1": [10.0, 0.0]}])
def test_non_positive_times_raise(kwargs):
    ss, ops = _two_spins()
    with pytest.raises(ValueError):
        LiouvilleSequence([Delay(1.0)], ss,
                          relaxation=Relaxation(**kwargs)).propagate(ops["Iz"].astype(complex))


def test_max_step_must_be_positive():
    with pytest.raises(ValueError):
        Relaxation(T2=10.0, max_step=0.0)


def test_wrong_length_time_list_raises():
    ss, ops = _two_spins()
    with pytest.raises(ValueError):
        LiouvilleSequence([Delay(1.0)], ss,
                          relaxation=Relaxation(T2=[10.0, 20.0, 30.0])
                          ).propagate(ops["Iz"].astype(complex))
