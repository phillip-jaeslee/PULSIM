"""
Analytic tests for B2b: gradients as position-dependent offsets
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.bloch import bloch_delay
from PULSIM.gradients import gradient_offsets, uniform_positions, ensemble_average, ideal_spoil

GAMMA = 42.577          # kHz/mT

# -- Units

def test_offset_at_a_known_position():
    """A gradient of 10 mT/m at 5 mm gives 0.05 mT, i.e. Gamma * 0.05 kHz
    
    Hand-computable on purpose: 42.577 * 10 * 0.005 = 2.12885 kHz. A stray
    2 * pi or a m/mm mixup survive this.
    """
    df = gradient_offsets([0.0, 0.0, 10.0], [[0.0, 0.0, 0.005]], GAMMA)
    assert df == pytest.approx([2.12885], abs=1e-12)

def test_zero_gradient_gives_zero_offset_everywhere():
    r = uniform_positions(0.01, 16)
    df = gradient_offsets([0.0, 0.0, 0.0], r, GAMMA)
    assert  np.array_equal(df, np.zeros(16))

def test_offset_is_odd_about_the_slab_center():
    """+z and -z must dephase in opposite senses, or nothing refocuses."""
    r = np.array([[0.0, 0.0, 0.003], [0.0, 0.0, -0.003]])
    df = gradient_offsets([0.0, 0.0, 7.0], r, GAMMA)
    assert df[0] == pytest.approx(-df[1], abs=1e-12)
    assert df[0] > 0

def test_only_the_gradient_axis_contributes():
    """A z-gradient must not see an x-displacement."""
    df = gradient_offsets([0.0, 0.0, 10.0], [[0.5, -0.3, 0.0]], GAMMA)
    assert df == pytest.approx([0.0], abs=1e-15)

def test_sampled_span_is_one_cell_short_of_the_slab():
    """Midpoint sampling, stated as a test so the convention cannot drift."""
    g, L, n = 10.0, 0.01, 64
    r = uniform_positions(L, n)
    df = gradient_offsets([0.0, 0.0, g], r, GAMMA)
    assert df.max() - df.min() == pytest.approx(GAMMA * g * L * (1 - 1 / n), abs=1e-12)

def test_uniform_positions_shape_and_symmetry():
    r = uniform_positions(0.02, 33, axis="z", center=0.0)
    assert r.shape == (33, 3)
    assert np.array_equal(r[:, 0], np.zeros(33))    # x untouched
    assert np.array_equal(r[:, 1], np.zeros(33))    # y untouched
    assert r[:, 2].mean() == pytest.approx(0.0, abs=1e-15)

def test_single_position_sits_at_the_center():
    r = uniform_positions(0.01, 1, center=0.004)
    assert r.shape == (1, 3)
    assert r[0] == pytest.approx([0.0, 0.0, 0.004], abs=1e-15)    

@pytest.mark.parametrize("kwargs", [{"n_positions": 0}, {"n_positions": -3}])
def test_bad_sample_count_rejected(kwargs):
    with pytest.raises(ValueError):
        uniform_positions(0.01, **kwargs)

def test_bad_shapes_and_axis_rejected():
    with pytest.raises(ValueError):
        gradient_offsets([0.0, 0.0], [[0.0, 0.0, 0.001]], GAMMA)        # G not a 3-vector
    with pytest.raises(ValueError):
        gradient_offsets([0.0, 0.0, 1.0], [0.0, 0.0, 0.001], GAMMA)     # r not 2D
    with pytest.raises(ValueError):
        uniform_positions(0.01, 8, axis="q")


# -- Averaging and spoiling

def test_ensemble_average_of_identical_spins_is_that_spin():
    M = np.tile(np.array([[0.3], [-0.4], [0.87]]), (1, 25))
    assert ensemble_average(M) == pytest.approx([0.3, -0.4, 0.87], abs=1e-15)

def test_ensemble_average_cancels_opposed_spins():
    M = np.array([[1.0, -1.0], [0.5, -0.5], [0.2, 0.2]])
    assert ensemble_average(M) == pytest.approx([0.0, 0.0, 0.2], abs=1e-15)

def test_weights_need_not_be_normalized():
    M = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    a = ensemble_average(M, weights=[3.0, 1.0])
    b = ensemble_average(M, weights=[0.75, 0.25])
    assert a == pytest.approx([0.75, 0.0, 0.25], abs=1e-15)
    assert a == pytest.approx(b, abs=1e-15)

def test_ideal_spoil_zeroes_both_transverse_components():
    """The old spoil_magnetization zeroed only My. This is the regression."""
    M = np.array([[0.6, -0.2], [0.3, 0.9], [0.5, -0.1]])
    out = ideal_spoil(M)
    assert np.array_equal(out[0], np.zeros(2))
    assert np.array_equal(out[1], np.zeros(2))
    assert np.array_equal(out[2], M[2])


def test_ideal_spoil_does_not_mutate_its_argument():
    """The old one aliased instead of copying and destroyed the caller's array."""
    M = np.array([[0.6], [0.3], [0.5]])
    before = M.copy()
    ideal_spoil(M)
    assert np.array_equal(M, before)


def test_ideal_spoil_accepts_a_single_vector():
    assert ideal_spoil(np.array([0.6, 0.3, 0.5])) == pytest.approx([0.0, 0.0, 0.5])


def test_bloch_delay_precesses_with_the_established_sign():
    """df*t = 0.25 takes +x to +y, same convention as test_relaxation.py."""
    df = np.array([0.25])
    M = np.array([[1.0], [0.0], [0.0]])
    out = bloch_delay(M, 1.0, df, GAMMA)
    assert out[:, 0] == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)    


# -- Dephasing physics

def _dephase(twists, n_positions, length=0.01, gradient=10.0, **kwargs):
    """M along +x everywhere, dephased for `twists` full turns across the slab.

    `twists` = Gamma * g * L * t is the natural dimensionless variable: the
    number of complete 2*pi phase cycles accumulated from one face of the slab
    to the other. One twist means the magnetization has wound exactly once,
    which is the classic condition for complete cancellation.
    """
    t = twists / (GAMMA * gradient * length)
    r = uniform_positions(length, n_positions)
    df = gradient_offsets([0.0, 0.0, gradient], r, GAMMA)
    M = np.zeros((3, n_positions))
    M[0] = 1.0
    return ensemble_average(bloch_delay(M, t, df, GAMMA, **kwargs))

### The sinc law

def test_slab_dephasing_follows_the_sinc_law():
    """Uniform slab, linear gradient:  Mbar+ = sinc(Gamma g L t).

    Derived by integrating exp(i 2 pi Gamma g z t) over z in [-L/2, L/2].
    One formula constraining the offset map, the units, the propagator and
    the averaging at once -- the strongest single test in B2b.
    """
    for twists in (0.25, 0.5, 1.5, 2.5):
        bar = _dephase(twists, 2048)
        assert bar[0] == pytest.approx(np.sinc(twists), abs=1e-5), f"{twists} twists"
        assert bar[1] == pytest.approx(0.0, abs=1e-12)    # odd in z, cancels
        assert bar[2] == pytest.approx(0.0, abs=1e-12)    # nothing tips Mz


def test_dephasing_matches_the_exact_discrete_sum():
    """The sinc law is the continuum limit; a finite sample has its own
    closed form, and we can assert THAT to machine precision.

    For n midpoint samples and k twists the geometric sum collapses to

        Mbar_x = sin(pi k) / (n sin(pi k / n)),

    purely real, tending to sinc(k) as n grows. Asserting the discrete form
    is sharper than asserting sinc with a tolerance, and it pins the
    sampling rule as well as the physics.
    """
    for twists, n in ((0.25, 64), (1.5, 64), (2.5, 128)):
        bar = _dephase(twists, n)
        exact = np.sin(np.pi * twists) / (n * np.sin(np.pi * twists / n))
        assert bar[0] == pytest.approx(exact, abs=1e-12), f"{twists} twists, n={n}"


def test_complete_dephasing_at_integer_twists():
    """One full winding cancels exactly, not approximately.

    The sampled phases are n-th roots of unity, so their sum is identically
    zero whenever n does not divide k. This is the test that would break if
    uniform_positions were changed to sample the slab edges.
    """
    for twists in (1, 2, 3, 5):
        bar = _dephase(twists, 64)
        assert np.hypot(bar[0], bar[1]) < 1e-12, f"{twists} twists"


### Gradient echo

def test_gradient_echo_refocuses_exactly():
    """+g then -g for equal times undoes the dephasing completely.

    This is the property every gradient experiment rests on, and it fails
    loudly if the offset sign, the position sign or the phase accumulation
    is wrong. Checked mid-sequence too: the signal really is gone before it
    comes back, so this cannot pass by never having dephased at all.
    """
    length, gradient, n = 0.01, 10.0, 64
    t = 1.0 / (GAMMA * gradient * length)           # exactly one twist
    r = uniform_positions(length, n)

    M = np.zeros((3, n))
    M[0] = 1.0

    M = bloch_delay(M, t, gradient_offsets([0.0, 0.0, gradient], r, GAMMA), GAMMA)
    dephased = ensemble_average(M)
    assert np.hypot(dephased[0], dephased[1]) < 1e-12      # signal gone

    M = bloch_delay(M, t, gradient_offsets([0.0, 0.0, -gradient], r, GAMMA), GAMMA)
    assert ensemble_average(M) == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test_gradient_echo_under_t2_loses_amplitude_but_not_phase():
    """Relaxation attenuates the echo; it does not spoil the refocusing.

    T2 scales every isochromat equally regardless of its phase, so the echo
    comes back at exp(-2t/T2) with its direction intact. Ties B2b to B2a,
    and exercises the claim in bloch_delay's docstring that the splitting is
    exact for a purely longitudinal field.
    """
    length, gradient, n, T2 = 0.01, 10.0, 64, 40.0
    t = 1.0 / (GAMMA * gradient * length)
    r = uniform_positions(length, n)

    M = np.zeros((3, n))
    M[0] = 1.0

    M = bloch_delay(M, t, gradient_offsets([0, 0,  gradient], r, GAMMA), GAMMA, T2=T2)
    M = bloch_delay(M, t, gradient_offsets([0, 0, -gradient], r, GAMMA), GAMMA, T2=T2)

    assert ensemble_average(M) == pytest.approx([np.exp(-2 * t / T2), 0.0, 0.0], abs=1e-12)

def test_physical_dephasing_is_not_ideal_spoiling():
    """The distinction the docs are required to make, as an assertion.

    After one twist the AVERAGE transverse signal is zero -- but every
    individual isochromat still carries its full transverse magnetization
    and can be refocused (see the gradient-echo test above). ideal_spoil
    destroys it outright. The two agree on the bulk vector and disagree
    completely on the ensemble behind it.
    """
    length, gradient, n = 0.01, 10.0, 64
    t = 1.0 / (GAMMA * gradient * length)
    r = uniform_positions(length, n)

    M = np.zeros((3, n))
    M[0], M[2] = 0.8, 0.6                      # tilted, so Mz is non-trivial

    dephased = bloch_delay(M, t, gradient_offsets([0, 0, gradient], r, GAMMA), GAMMA)
    spoiled = ideal_spoil(M)

    # Same bulk observable, and Mz untouched by either
    assert ensemble_average(dephased) == pytest.approx([0.0, 0.0, 0.6], abs=1e-12)
    assert ensemble_average(spoiled) == pytest.approx([0.0, 0.0, 0.6], abs=1e-12)

    # Utterly different ensembles: dephasing preserves |Mxy| per isochromat
    assert np.hypot(dephased[0], dephased[1]).min() == pytest.approx(0.8, abs=1e-12)
    assert np.abs(spoiled[:2]).max() == 0.0

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
