"""
pulse_diagnostics.py -- quantify how much J-coupling evolution a finite
shaped pulse contributes, and find the free-evolution delay that actually
maximize transfer for a real (non-instantaneous-pulse) sequence.

Two tools, two different jobs:

effective_coupling_generator(rf, dt)

"""

import numpy as np
from scipy.optimize import minimize_scalar

def _rotrigues_rotation(v, axis, angle):
    k = axis / np.linalg.norm(axis)
    return (v * np.cos(angle) + np.cross(k, v) * np.sin(angle)
            + k * np.dot(k, v) * (1 - np.cos(angle)))

def effective_coupling_generator(rf, dt):
    """
    rf  : complex array, wx + i*wy in rad/ms (same convention as
          RawShapedPulseSegment/ShapePulseSegment's per-timestep RF --
          already includes 2*pi*Gamma if it cam from Pulse.calibrated_rf())
    dt  : sample spacing, ms.
    """
    M = np.array([0.0, 0.0, 1.0])
    traj = [M.copy()]
    for n in range(len(rf)):
        wx, wy = rf[n].real, rf[n].imag
        w = np.hypot(wx, wy)
        if w > 1e-12:
            M = _rotrigues_rotation(M, np.array([wx, wy, 0.0]), w * dt)
        traj.append(M.copy())
    traj = np.asarray(traj)
    # manual trapezoid, avoids np.trapz/np.trapezoid version churn
    integral = dt * (0.5 * traj[0] + traj[1:-1].sum(axis=0) + 0.5 * traj[-1])
    return tuple(integral)

def optimize_delay(run_sequence, objective, bounds):
    """
    run_sequence(Delta) -> sigma_final   builds and propagates the real
        sequence for a given free-evolution delay Delta, with J-coupling
        active throughout (including during shaped pulses).
    objective(sigma_final) -> float      what to MAXIMIZE, e.g. the
        transfer amplitude of interest.
    bounds : (low, high) search range for Delta, ms.

    Returns the scipy OptimizeResult (`.x` is the optimal Delta).
    """
    def neg_objective(Delta):
        return -objective(run_sequence(Delta))
    return minimize_scalar(neg_objective, bounds=bounds, method='bounded')
