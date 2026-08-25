"""
Cross-formalism correctness test: classical Bloch-vector propagation 
(bloch_rotate, via PULSIM) vs. quantum density-matrix propagation
(spin_operators.py + scipy.linalg.expm), for a single uncoupled spin.

Independent ground truth, not a golden-file test: this does not compare new
code to old code, it compares two different physics formalisms that must
agree exactly for an isolated spin-1/2 (the SU(2)/SO(3) isomorphism).

Hamiltonian used is H = +2*pi*Gamma*(B. I) -- positive sign matches the
Ernst/Levitt RF-pulse convention bloch_rotate was deliberately fixed to
follow (see tests/test_bloch_rotate_levitt_convention.py, benchmarked
directly against Levitt's Spin Dynamics product-operator table). The
lab-frame classical Bloch equation gives the opposite mathematical
rotation sense for a positive-gamma nucleus, but that's a different
question -- see test_bloch_rotate_levitt_convention.py's docstring.
"""

import os
import sys

import numpy as np
from scipy.linalg import expm

_here = os.path.dirname(os.path.abspath(__file__))
_pulsim_root = os.path.dirname(_here)
sys.path.insert(0, _pulsim_root)
sys.path.insert(0, os.path.join(_pulsim_root, "..", "..", "densityMatSim"))

from PULSIM.rf_shape import RFShape
from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.spin_operators import Ix, Iy, Iz


PI = np.pi

def _max_diff_for_shape(shape_name, duration=1.0, points=500, flip = PI / 2, axis="x", Gamma=42.577478, offsets=(0.0, 1.0, 3.0, 5.0)):

    offsets = np.array(offsets)

    shape = RFShape.create(shape_name, duration=duration, points=points)
    pulse = Pulse(shape, flip, axis=axis, backend=NumpyBackend(Gamma=Gamma))

    M0 = np.zeros((3, len(offsets)))
    M0[2, :] = 1.0                  # set magnetization to Mz
    M_classical = pulse.apply(M0, offsets)

    RF = pulse.calibrated_rf()
    dt = shape.dt
    Ix_op, Iy_op, Iz_op = Ix(), Iy(), Iz()

    M_density = np.zeros((3, len(offsets)))
    with np.errstate(divide='ignore', invalid='ignore'):
        for f, df in enumerate(offsets):
            sigma = Iz_op.copy()
            for n in range(len(RF)):
                Bx, By, Bz = np.real(RF[n]), np.imag(RF[n]), df /Gamma
                H = 2 * PI * Gamma * (Bx * Ix_op + By * Iy_op + Bz * Iz_op)
                sigma = expm(-1j * H * dt) @ sigma @ expm(1j * H *dt)

            tmp = sigma.real / Ix_op.real; Mx = tmp[np.isfinite(tmp)].mean()
            tmp = sigma.imag / Iy_op.imag; My = tmp[np.isfinite(tmp)].mean()
            tmp = sigma.real / Iz_op.real; Mz = tmp[np.isfinite(tmp)].mean()
            M_density[:, f] = [Mx, My, Mz]

    return np.max(np.abs(M_density - M_classical))

def test_bloch_matches_density_matrix_hard():
    diff = _max_diff_for_shape("hard")
    print(f"hard pulse max diff: {diff:.3e}")
    assert diff < 1e-9, "classical and density-matrix trajectories disagree for a hard pulse"

def test_bloch_matches_density_matrix_shaped():
    diff = _max_diff_for_shape("gausscasQ5")
    print(f"hard pulse max diff: {diff:.3e}")
    assert diff < 1e-9, "classical and density-matrix trajectories disagree for a shaped (complex_envelop) pulse"



if __name__ == "__main__":
    test_bloch_matches_density_matrix_hard()
    test_bloch_matches_density_matrix_shaped()
    print("PASS")