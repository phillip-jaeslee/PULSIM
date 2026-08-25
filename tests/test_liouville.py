"""
Correctness tests for liouville.py -- NOT golden-file tests.

liouville.py had zero test coverage before this file: test_bloch_density_
equivalence.py checks the *physics* (Bloch vector vs density matrix) but
reimplements the propagation by hand with a bare scipy.linalg.expm loop --
it never actually calls Segment/LiouvilleSequence. These tests exercise the
real classes (Delay, IdealPulse, ShapePulseSegment, LiouvilleSequence,
SpinSystem) against independent ground truth:

  - Delay and IdealPulse are checked against Rodrigues' rotation formula on
    the classical Bloch vector recovered from the density matrix. Ground
    truth is +offset*duration (Delay) and +flip (IdealPulse), the same
    Ernst/Levitt sign convention established for bloch_rotate in
    test_bloch_rotate_levitt_convention.py.

  - ShapePulseSegment is cross-checked against the already-verified
    classical stack (Pulse.apply -> bloch_rotate_batch) for the same RF
    waveform and offsets.

  - The J-coupling term is checked against Levitt's Spin Dynamics in-phase
    -> antiphase product-operator formula:
        Ix1  -->  Ix1*cos(pi*J*t) + 2*Iy1*Iz2*sin(pi*J*t)
    under a coupling-only Delay (include_offset=False).

  - A general unitarity/Hermiticity check across a composed sequence
    guards against future segments breaking the U sigma U^dagger contract.

  - Channel selectivity (new): a ShapePulseSegment/IdealPulse targeted at
    one spin's gamma leaves every other spin's density-matrix contribution
    exactly untouched -- this is what makes heteronuclear-selective pulses
    (1H-only, 13C-only, as in INEPT) possible.
"""

import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, ShapePulseSegment, LiouvilleSequence
from PULSIM.rf_shape import RFShape
from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse

PI = np.pi
GAMMA_H = gyro_ratio('H')
GAMMA_C = gyro_ratio('13C')


def rodrigues_rotation(v, axis, angle):
    """Rotate vector v around `axis` by `angle` radians. Independent ground truth."""
    k = axis / np.linalg.norm(axis)
    return (v * np.cos(angle)
            + np.cross(k, v) * np.sin(angle)
            + k * np.dot(k, v) * (1 - np.cos(angle)))


def spin_expectation(sigma, spin_index, n_spins):
    """<Ix>, <Iy>, <Iz> of one spin in an n_spins density matrix, via the
    trace formula <A> = Tr(sigma A) / Tr(A A). Reduces to the old
    single-spin element-wise-ratio trick exactly when n_spins == 1."""
    out = []
    for op in (Ix(), Iy(), Iz()):
        A = embed(op, spin_index, n_spins)
        out.append(np.trace(sigma @ A).real / np.trace(A @ A).real)
    return np.array(out)


def test_delay_matches_rodrigues_z_rotation():
    offset = 3.7
    duration = 0.3
    ss = SpinSystem(nuclei=['H'], offsets=[offset], couplings={})
    seq = LiouvilleSequence([Delay(duration)], ss)
    M_final = spin_expectation(seq.propagate(Ix()), 0, 1)
    expected = rodrigues_rotation(np.array([1.0, 0.0, 0.0]), np.array([0, 0, 1.0]), offset * duration)
    diff = np.abs(M_final - expected).max()
    print(f"Delay vs Rodrigues z-rotation: diff={diff:.3e}")
    assert diff < 1e-9, "Delay no longer matches +offset*duration z-rotation"


def test_ideal_pulse_matches_rodrigues_x_rotation():
    flip = PI / 3
    ss = SpinSystem(nuclei=['H'], offsets=[0.0], couplings={})
    seq = LiouvilleSequence([IdealPulse(flip, phase=0.0)], ss)
    M_final = spin_expectation(seq.propagate(Iz()), 0, 1)
    expected = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0, 0]), flip)
    diff = np.abs(M_final - expected).max()
    print(f"IdealPulse (phase=0) vs Rodrigues x-rotation: diff={diff:.3e}")
    assert diff < 1e-9, "IdealPulse no longer matches +flip x-rotation"


def test_ideal_pulse_phase_selects_rotation_axis():
    flip = PI / 3
    ss = SpinSystem(nuclei=['H'], offsets=[0.0], couplings={})
    seq = LiouvilleSequence([IdealPulse(flip, phase=PI / 2)], ss)
    M_final = spin_expectation(seq.propagate(Iz()), 0, 1)
    expected = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), np.array([0, 1.0, 0]), flip)
    diff = np.abs(M_final - expected).max()
    print(f"IdealPulse (phase=pi/2) vs Rodrigues y-rotation: diff={diff:.3e}")
    assert diff < 1e-9, "IdealPulse phase no longer selects the y-axis correctly"


def test_shape_pulse_segment_matches_classical_stack():
    offsets_khz = np.array([0.0, 1.0, 3.0])

    shape = RFShape.create("hard", duration=1.0, points=500)
    pulse = Pulse(shape, PI / 2, axis="x", backend=NumpyBackend(Gamma=GAMMA_H))

    M0 = np.zeros((3, len(offsets_khz)))
    M0[2, :] = 1.0
    M_classical = pulse.apply(M0, offsets_khz)

    seg = ShapePulseSegment(pulse)
    worst = 0.0
    for f, off in enumerate(offsets_khz):
        ss = SpinSystem(nuclei=['H'], offsets=[2 * PI * off], couplings={})
        seq = LiouvilleSequence([seg], ss)
        M_final = spin_expectation(seq.propagate(Iz()), 0, 1)
        worst = max(worst, np.abs(M_final - M_classical[:, f]).max())

    print(f"ShapePulseSegment vs classical Pulse.apply: worst diff={worst:.3e}")
    assert worst < 1e-9, "ShapePulseSegment disagrees with the already-verified classical stack"


def test_j_coupling_inphase_to_antiphase():
    J = 140.0   # real Hz, e.g. a 1J(CH)-scale coupling
    t = 0.05    # ms
    ss = SpinSystem(nuclei=['H', 'H'], offsets=[0, 0], couplings={(0, 1): J})
    seq = LiouvilleSequence([Delay(t, include_offset=False)], ss)
    sigma_final = seq.propagate(embed(Ix(), 0, 2))

    J_internal = J / 1000.0   # Hz -> PULSIM's ms time base
    expected = (np.cos(PI * J_internal * t) * embed(Ix(), 0, 2)
                + np.sin(PI * J_internal * t) * product_operator(Iy(), 0, Iz(), 1, 2))
    diff = np.abs(sigma_final - expected).max()
    print(f"J-coupling vs Levitt in-phase->antiphase formula: diff={diff:.3e}")
    assert diff < 1e-9, "Delay's J-coupling term no longer matches Levitt's product-operator formula"

def test_composed_sequence_stays_unitary():
    ss = SpinSystem(nuclei=['H'], offsets=[2.0], couplings={})
    seq = LiouvilleSequence(
        [IdealPulse(PI / 2, phase=0.0), Delay(0.1), IdealPulse(PI / 3, phase=PI / 2)], ss,
    )
    sigma0 = Iz()
    sigma_final = seq.propagate(sigma0)

    tr0 = np.trace(sigma0 @ sigma0.conj().T).real
    trf = np.trace(sigma_final @ sigma_final.conj().T).real
    herm_diff = np.abs(sigma_final - sigma_final.conj().T).max()

    print(f"tr(sigma0^2)={tr0:.6f}, tr(sigma_f^2)={trf:.6f}, Hermiticity diff={herm_diff:.3e}")
    assert abs(tr0 - trf) < 1e-9, "propagate() is not trace-preserving across composed segments"
    assert herm_diff < 1e-9, "propagate() is not Hermiticity-preserving across composed segments"


def test_shape_pulse_segment_is_channel_selective():
    shape = RFShape.create("hard", duration=1.0, points=200)
    pulse_H = Pulse(shape, PI / 2, axis="x", backend=NumpyBackend(Gamma=GAMMA_H))
    seg_H = ShapePulseSegment(pulse_H)

    ss = SpinSystem(nuclei=['H', '13C'], offsets=[0.0, 0.0], couplings={})
    sigma0 = embed(Iz(), 0, 2) + embed(Iz(), 1, 2)
    seq = LiouvilleSequence([seg_H], ss)
    sigma_final = seq.propagate(sigma0)

    M_H = spin_expectation(sigma_final, 0, 2)
    M_C = spin_expectation(sigma_final, 1, 2)
    print(f"1H (targeted): {M_H}, 13C (untargeted): {M_C}")

    expected_H = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0, 0]), PI / 2)
    assert np.abs(M_H - expected_H).max() < 1e-4, "channel-targeted spin no longer rotates correctly"
    assert np.abs(M_C - np.array([0.0, 0.0, 1.0])).max() < 1e-9, "a channel-selective pulse touched a spin outside its channel"


def test_ideal_pulse_is_channel_selective():
    ss = SpinSystem(nuclei=['H', '13C'], offsets=[0.0, 0.0], couplings={})
    seq = LiouvilleSequence([IdealPulse(PI / 2, phase=0.0, duration=1e-6, channel=GAMMA_H)], ss)
    sigma0 = embed(Iz(), 0, 2) + embed(Iz(), 1, 2)
    sigma_final = seq.propagate(sigma0)

    M_H = spin_expectation(sigma_final, 0, 2)
    M_C = spin_expectation(sigma_final, 1, 2)
    print(f"1H (targeted): {M_H}, 13C (untargeted): {M_C}")

    expected_H = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0, 0]), PI / 2)
    assert np.abs(M_H - expected_H).max() < 1e-9, "channel-targeted IdealPulse no longer rotates correctly"
    assert np.abs(M_C - np.array([0.0, 0.0, 1.0])).max() < 1e-9, "IdealPulse's channel selection touched an untargeted spin"

def test_raw_shaped_pulse_segment_matches_rodrigues():
    from PULSIM.liouville import RawShapedPulseSegment
    w1 = 0.5    # rad/ms, constant CW field along x -- no calibration involved
    n_pts, T = 200, 3.0
    rf = np.full(n_pts, w1, dtype=complex)
    seg = RawShapedPulseSegment(rf, T / n_pts, channel='H')
    ss = SpinSystem(nuclei=['H'], offsets=[0.0], couplings={})
    seq = LiouvilleSequence([seg], ss)
    M_final = spin_expectation(seq.propagate(Iz()), 0, 1)
    expected = rodrigues_rotation(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0, 0]), w1 * T)
    diff = np.abs(M_final - expected).max()
    print(f"RawShapedPulseSegment vs Rodrigues: diff={diff:.3e}")
    assert diff < 1e-9, "RawShapedPulseSegment no longer matches a direct constant-RF rotation"

if __name__ == "__main__":
    test_delay_matches_rodrigues_z_rotation()
    test_ideal_pulse_matches_rodrigues_x_rotation()
    test_ideal_pulse_phase_selects_rotation_axis()
    test_shape_pulse_segment_matches_classical_stack()
    test_j_coupling_inphase_to_antiphase()
    test_composed_sequence_stays_unitary()
    test_shape_pulse_segment_is_channel_selective()
    test_ideal_pulse_is_channel_selective()
    test_raw_shaped_pulse_segment_matches_rodrigues()
    print("PASS")