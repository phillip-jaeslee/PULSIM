"""
Analytic tests for B2a: phenomenological T1/T2 relaxation

Nothing here is a golden test. Every assertion is against a closed-form
solution of the Bloch equation, or against a convergence rate the numerical
scheme is required to have -- so these tests can fail even if PULSIM
reproduces its own past output exactly.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.bloch import affine_propagate, bloch_relax, bloch_relax_rotate_batch, bloch_rotate, bloch_rotate_batch, relaxation_matrix

GAMMA = 42.577

# -- Free T2 decay

def test_t2_decay_matches_closed_form():
    T2 = 25.0
    M_start = np.array([0.6, -0.8, 0.3])
    for t in (0.0, 1.0, 12.5, 25.0, 100):
        got = bloch_relax(M_start, t, M0=1.0, T1=None, T2=T2)
        decay = np.exp(-t / T2)
        assert got[0] == pytest.approx(M_start[0] * decay, abs=1e-14)
        assert got[1] == pytest.approx(M_start[1] * decay, abs=1e-14)
        assert got[2] == M_start[2]  # T1 is None, so Mz must not move at all

def test_t2_and_t1_stay_in_their_own_lanes():
    M = np.array([1.0, 1.0, 1.0])
    only_t2 = bloch_relax(M, 0.5, M0=1.0, T1=None, T2=10.0)
    only_t1 = bloch_relax(M, 0.5, M0=1.0, T1=10.0, T2=None)    
    assert only_t2[2] == 1.0                        # Mz untouched by T2
    assert only_t1[0] == 1.0 and only_t1[1] == 1.0  # Mxy untouched by T1


#-- Free T1 recovery

def test_t1_recovery_matches_closed_form():
    T1, M0 = 800.0, 1.0
    start = np.array([0.0, 0.0, -M0])       #inverted
    for t in (0.0, 100.0, 554.5, 1600.0, 8000.0):
        got = bloch_relax(start, t, M0=M0, T1=T1, T2=None)
        want = M0 + (start[2] - M0) * np.exp(-t / T1)
        assert got[2] == pytest.approx(want, abs=1e-14)

def test_inversion_recovery_null_at_t1_ln2():
    """The one number an NMR reader checks by eye: Mz crosses zero at T1 ln 2."""
    T1, M0 = 800.0, 1.0
    t_null = T1 * np.log(2.0)
    got = bloch_relax(np.array([0.0, 0.0, -M0]), t_null, M0=M0, T1=T1, T2=None)
    assert got[2] == pytest.approx(0.0, abs=1e-14)

def test_recovery_saturates_at_m0():
    """From fully saturated (M = 0), long-time limit is equilibrium, not zero."""
    M0 = 1.0
    got = bloch_relax(np.array([0.0, 0.0, 0.0]), 1e6, M0=M0, T1=10.0, T2=1.0)
    assert got == pytest.approx([0.0, 0.0, M0], abs=1e-14)


# -- Relaxation OFF

def test_relaxation_off_is_the_exact_identity():
    M = np.array([0.3, -0.4, 0.87])
    out = bloch_relax(M, 37.0, M0=1.0, T1=None, T2=None)
    assert np.array_equal(out, M)       # bit for bit, not approx

def test_rotation_path_is_bit_identical_when_relaxation_is_off():
    """The early return in bloch_relax_rotate_batch, asserted as written."""
    rng = np.random.default_rng(20260911)
    M = rng.normal(size=(3, 12))
    B = rng.normal(size=(12, 3)) * 0.01
    dt = 0.005
    plain = bloch_rotate_batch(M, dt, B, "x", GAMMA)
    through_relax = bloch_relax_rotate_batch(M, dt, B, "x", GAMMA, T1=None, T2=None)
    assert np.array_equal(plain, through_relax)

def test_strang_step_is_not_the_same_as_no_relaxation():
    """The other half: the early return must fire ONLY when both are None.

    T2-only and T1-only are tested separately, because an `and` where an `or`
    belongs passes the both-given case and silently drops the other two.
    """
    B = np.array([[0.01, 0.0, 0.0]])
    M = np.array([[0.0], [0.0], [1.0]])
    plain = bloch_rotate_batch(M, 0.5, B, "x", GAMMA)
    for kwargs in ({"T2": 10.0}, {"T1": 50.0}, {"T1": 50.0, "T2": 10.0}):
        relaxed = bloch_relax_rotate_batch(M, 0.5, B, "x", GAMMA, **kwargs)
        assert np.max(np.abs(relaxed - plain)) > 1e-6, kwargs

def test_infinite_relaxation_times_are_accepted():
    """np.inf is the documented alternative spelling of None"""
    M = np.array([0.3, -0.4, 0.87])
    out = bloch_relax(M, 37.0, M0=1.0, T1=np.inf, T2=np.inf)
    assert out == pytest.approx(M, abs=0.0)
    assert np.array_equal(relaxation_matrix(np.inf, np.inf), np.zeros((3, 3)))


# -- affine_propagate, both limits

def test_affine_reduces_to_pure_rotation():
    """R = 0: the 4x4 must agree with bloch_rotate, which the Levitt and
    Rodrigues tests already pin down independently.
    """
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(200):
        M = rng.normal(size=3)
        B = rng.normal(size=3) * 0.02
        dt = rng.uniform(0.001, 0.05)
        Omega = 2 * np.pi * GAMMA * B
        got = affine_propagate(M, dt, Omega, T1=None, T2=None)
        want = bloch_rotate(M, dt, B, "x", GAMMA)
        worst = max(worst, np.max(np.abs(got - want)))
    assert worst < 1e-12, f"worst |affine - bloch_rotate| = {worst:g}"

def test_affine_reduces_to_free_relaxation():
    """Omega = 0: the 4x4 must agree with the closed form, over a range of dt
    spanning much-less-than to much-greater-than T2.
    """
    M = np.array([0.6, -0.2, -0.9])
    T1, T2, M0 = 700.0, 40.0, 1.0
    for dt in (0.01, 1.0, 25.0, 200.0):
        got = affine_propagate(M, dt, np.zeros(3), T1=T1, T2=T2, M0=M0)
        want = bloch_relax(M, dt, M0=M0, T1=T1, T2=T2)
        assert got == pytest.approx(want, abs=1e-12), f"dt={dt}"


# -- Converge

def test_strang_converges_to_affine_at_second_order():
    """Halving dt must cut the error by FOUR, not two.

    Rotation and relaxation do not commute unless T1 == T2, so a single
    split step is not excat -- affine_propagate is. What the production
    path owes us is not exactness but a convergence rate: second order.

    A first-order splitting (relax then rotate, no half-steps) would land
    near order 1.0 here, so this test distinguishes the two schemes. It is
    not merely an accuracy check.
    """
    T1, T2, M0 = 50.0, 10.0, 1.0            # T1 != T2 on purpose
    B = np.array([[0.005871, 0.0, 0.002]])  # tilted axis: RF plus offset
    total = 2.0
    Omega = 2 * np.pi * GAMMA * B[0]
    M_init = np.array([[0.0], [0.0], [1.0]])

    exact = affine_propagate(M_init[:, 0], total, Omega, T1, T2, M0)

    errors = {}
    for n_steps in (16, 32, 64, 128):
        dt = total / n_steps
        M = M_init.copy()
        for _ in range(n_steps):
            M = bloch_relax_rotate_batch(M, dt, B, "x", GAMMA, T1, T2, M0)
        errors[n_steps] = np.max(np.abs(M[:, 0] - exact))

    for coarse, fine in ((16, 32), (32, 64), (64, 128)):
        order = np.log2(errors[coarse] / errors[fine])
        assert 1.9 < order < 2.1, f"observed order {order:.3f} from {coarse}->{fine}"

    assert errors[128] < 1e-5, f"error at 128 steps = {errors[128]:g}"


# -- Precession

def test_free_precession_sign_at_quarter_cycle():
    """M+(t) = M+(0) exp(+i 2 pi df t): a quarter cycle takes +x to +y.

    df*t = 0.25 is chosen deliberately. At df*t = 0.5, or any integer, both
    candidate phase conventions give the same vector, so a test placed there
    cannot fail whichever convention the code implements. See
    PHYSICS_SPECIFICATION.md section 1.8 item 3 and section 1.11.
    """
    df, dt = 0.25, 1.0                              # kHz, ms -> df*dt = 0.25 cycles
    Omega = np.array([0.0, 0.0, 2 * np.pi * df])    # section 1.6: Omega_z = 2 * pi * df

    got = affine_propagate(np.array([1.0, 0.0, 0.0]), dt, Omega, T1=None, T2=None)
    assert got == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)

    # A negative offset must turn the other way. Without this half, a sign
    # error in Omega_hat AND in the offset would cancel and go unnoticed.
    got_neg = affine_propagate(np.array([1.0, 0.0, 0.0]), dt, -Omega, T1=None, T2=None)
    assert got_neg == pytest.approx([0.0, -1.0, 0.0], abs=1e-12)

def test_relaxation_scales_the_transverse_magnitude_without_turning_it():
    """Relaxation may shrink M+, never rotate it.

    R is diagonal, so it must not mix Mx into My,. If the relaxation and
    rotation blocks were combined wrongly in the 4x4, the quarter-cycle
    result would pick up a small Mx component -- this asserts it stays zero.
    """
    df, dt, T2 = 0.25, 1.0, 20.0
    Omega = np.array([0.0, 0.0, 2 * np.pi * df])
    got = affine_propagate(np.array([1.0, 0.0, 0.0]), dt, Omega, T1=None, T2=T2)
    assert got[0] == pytest.approx(0.0, abs=1e-12)
    assert got[1] == pytest.approx(np.exp(-dt / T2), abs=1e-12)


# -- Fixed points and the driven limit

def test_equilibrium_is_the_fixed_point():
    """M_eq must not move. With Omega along z it is also a rotation fixed
    point so any drift here comes from the relaxation half.
    """
    M0 = 1.0
    M_eq = np.array([0.0, 0.0, M0])
    Omega = np.array([0.0, 0.0, 2 * np.pi * 3.0])
    got = affine_propagate(M_eq, 5.0, Omega, T1=50.0, T2=10.0, M0=M0)
    assert got == pytest.approx(M_eq, abs=1e-12)

def test_long_time_limit_with_rf_off_returns_to_equilibrium():
    M0 = 1.0
    Omega = np.array([0.0, 0.0, 2 * np.pi * 3.0])       # offset only, no RF
    got = affine_propagate(np.array([0.7, -0.5, -0.9]), 5000.0, Omega, T1=10.0, T2=5.0, M0=M0)
    assert got == pytest.approx([0.0, 0.0, M0], abs=1e-9)

def test_cw_saturation_steady_state():
    """With RF left on, the long-time limit is NOT equilibrium.

    On resonance, setting dM/dt = 0 in

        dMx/dt = -Mx/T2
        dMy/dt = -w1 Mz - My/T2
        dMz/dt = +w1 My - (Mz - M0)/T1

    gives the textbook continuous-wave saturation result

        Mx = 0,   Mz = M0 / (1 + w1^2 T1 T2),   My = -w1 T2 Mz.

    Derived by hand, so it is an independent check of the driven limit and
    of the sign of the RF term at the same time.
    """
    M0, T1, T2 = 1.0, 40.0, 8.0
    w1 = 2 * np.pi * 0.05               # rad/ms
    Omega = np.array([w1, 0.0, 0.0])    # RF along +x, on resonance

    mz = M0 / (1.0 + w1 ** 2 * T1 * T2)
    my = -w1 * T2 * mz

    got = affine_propagate(np.array([0.0, 0.0, M0]), 5000.0, Omega, T1=T1, T2=T2, M0=M0)
    assert got == pytest.approx([0.0, my, mz], abs=1e-9)


# -- Bad input

@pytest.mark.parametrize("kwargs", [{"T1": 0.0}, {"T2": 0.0}, {"T1": -1.0}, {"T2": -5.0}])
def test_non_positive_relaxation_times_are_rejected(kwargs):
    """Zero is an error, not shorthand for instantaneous; None means infinite.

    Both entry points must agree: _decay guards bloch_relax, and
    relaxation_matrix guards the affine path.
    """
    with pytest.raises(ValueError):
        bloch_relax(np.array([0.0, 0.0, 1.0]), 1.0, M0=1.0, **kwargs)
    with pytest.raises(ValueError):
        relaxation_matrix(**kwargs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__], "-q"))