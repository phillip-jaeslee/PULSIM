"""
Locked-down benchmark: bloch_rotate against Levitt's Spin Dynamics (Ernst)
rotation convention, R_alpha(beta) = exp(-i*beta*I_alpha).

This is deliberately NOT a physics-first derivation (see the lab-frame RK4
check in the PR history for why that's a different, insufficient question --
free precession of a real spin and the Ernst RF/rotating-frame convention
use different rotation senses, reconciled via the rotating-frame and RF-phase
definitions). Instead, this benchmarks bloch_rotate directly against parts of
this codebase already independently verified to match Levitt's own
product-operator table: IdealPulse and Delay (liouville.py), both checked
against Iz -> -Iy for a 90x pulse, Iz -> +Ix for 90y, etc. earlier in this
project. Classical Bloch vectors are compared to <2Ix>, <2Iy>, <2Iz>
expectation values from the density-matrix oracle.

Covers all six cases requested for Task #34:
  1. 90x pulse
  2. 90y pulse
  3. 180x pulse
  4. free precession, positive and negative offset
  5. a pulse with a 45-degree phase
  6. a general effective field (RF + offset combined)
"""

import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_pulsim_root = os.path.dirname(_here)
sys.path.insert(0, _pulsim_root)

from PULSIM.bloch import bloch_rotate
from PULSIM.spin_operators import Ix, Iy, Iz, embed

GAMMA = 42.577478


def _expm_herm(H, t):
    w, V = np.linalg.eigh(H)
    return V @ np.diag(np.exp(-1j * w * t)) @ V.conj().T


def _sigma_to_M(sigma):
    Ix1, Iy1, Iz1 = embed(Ix(), 0, 1), embed(Iy(), 0, 1), embed(Iz(), 0, 1)
    return np.array([
        np.real(np.trace(sigma @ (2 * Ix1))),
        np.real(np.trace(sigma @ (2 * Iy1))),
        np.real(np.trace(sigma @ (2 * Iz1))),
    ])


def _levitt_ideal_pulse(sigma0, flip, phase):
    """Ground truth oracle: IdealPulse's own already-verified formula."""
    Ix1, Iy1 = embed(Ix(), 0, 1), embed(Iy(), 0, 1)
    duration = 1e-6
    w1 = flip / duration
    H = w1 * (np.cos(phase) * Ix1 + np.sin(phase) * Iy1)
    U = _expm_herm(H, duration)
    return U @ sigma0 @ U.conj().T


def _levitt_delay(sigma0, offset, duration):
    """Ground truth oracle: Delay's own already-verified formula."""
    Iz1 = embed(Iz(), 0, 1)
    H = offset * Iz1
    U = _expm_herm(H, duration)
    return U @ sigma0 @ U.conj().T


def test_90x_pulse():
    sigma_z = embed(Iz(), 0, 1)
    M_levitt = _sigma_to_M(_levitt_ideal_pulse(sigma_z, np.pi / 2, phase=0.0))
    T = 0.01
    Bmag = (np.pi / 2) / (2 * np.pi * GAMMA * T)
    M_br = bloch_rotate(np.array([0., 0., 1.]), T, [Bmag, 0, 0], "x", GAMMA)
    assert np.max(np.abs(M_br - M_levitt)) < 1e-9


def test_90y_pulse():
    sigma_z = embed(Iz(), 0, 1)
    M_levitt = _sigma_to_M(_levitt_ideal_pulse(sigma_z, np.pi / 2, phase=np.pi / 2))
    T = 0.01
    Bmag = (np.pi / 2) / (2 * np.pi * GAMMA * T)
    M_br = bloch_rotate(np.array([0., 0., 1.]), T, [Bmag, 0, 0], "y", GAMMA)
    assert np.max(np.abs(M_br - M_levitt)) < 1e-9


def test_180x_pulse():
    sigma_z = embed(Iz(), 0, 1)
    M_levitt = _sigma_to_M(_levitt_ideal_pulse(sigma_z, np.pi, phase=0.0))
    T = 0.01
    Bmag = np.pi / (2 * np.pi * GAMMA * T)
    M_br = bloch_rotate(np.array([0., 0., 1.]), T, [Bmag, 0, 0], "x", GAMMA)
    assert np.max(np.abs(M_br - M_levitt)) < 1e-9


def test_free_precession_both_offset_signs():
    duration = 0.02
    Ix1 = embed(Ix(), 0, 1)
    for offset in (5.0, -5.0):
        M_levitt = _sigma_to_M(_levitt_delay(Ix1.copy(), offset, duration))
        Bz = offset / (2 * np.pi * GAMMA)
        M_br = bloch_rotate(np.array([1., 0., 0.]), duration, [0, 0, Bz], "x", GAMMA)
        assert np.max(np.abs(M_br - M_levitt)) < 1e-8, f"offset={offset}"


def test_45_degree_phase_pulse():
    sigma_z = embed(Iz(), 0, 1)
    M_levitt = _sigma_to_M(_levitt_ideal_pulse(sigma_z, np.pi / 2, phase=np.pi / 4))
    T = 0.01
    Bmag = (np.pi / 2) / (2 * np.pi * GAMMA * T)
    Bx, By = Bmag * np.cos(np.pi / 4), Bmag * np.sin(np.pi / 4)
    M_br = bloch_rotate(np.array([0., 0., 1.]), T, [Bx, By, 0], "x", GAMMA)
    assert np.max(np.abs(M_br - M_levitt)) < 1e-9


def test_general_effective_field():
    sigma_z = embed(Iz(), 0, 1)
    Ix1, Iy1, Iz1 = embed(Ix(), 0, 1), embed(Iy(), 0, 1), embed(Iz(), 0, 1)
    offset, w1, phase, duration = 3.0, 40.0, np.pi / 6, 0.008
    H = offset * Iz1 + w1 * (np.cos(phase) * Ix1 + np.sin(phase) * Iy1)
    U = _expm_herm(H, duration)
    M_levitt = _sigma_to_M(U @ sigma_z @ U.conj().T)

    Bx = w1 * np.cos(phase) / (2 * np.pi * GAMMA)
    By = w1 * np.sin(phase) / (2 * np.pi * GAMMA)
    Bz = offset / (2 * np.pi * GAMMA)
    M_br = bloch_rotate(np.array([0., 0., 1.]), duration, [Bx, By, Bz], "x", GAMMA)
    assert np.max(np.abs(M_br - M_levitt)) < 1e-9


if __name__ == "__main__":
    test_90x_pulse()
    test_90y_pulse()
    test_180x_pulse()
    test_free_precession_both_offset_signs()
    test_45_degree_phase_pulse()
    test_general_effective_field()
    print("all six Levitt-convention cases passed")