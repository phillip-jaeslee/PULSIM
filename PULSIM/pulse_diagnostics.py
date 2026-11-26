"""
pulse_diagnostics.py -- quantify how much J-coupling evolution a finite
shaped pulse contributes, and find the free-evolution delay that actually
maximizes transfer for a real (non-instantaneous-pulse) sequence.

Two tools, two different jobs:

effective_coupling_generator(rf, dt, side="before")
    First-order (average-Hamiltonian) account of a scalar coupling J*IzSz
    acting DURING a pulse on the I spin. The z component is the free-evolution
    delay the pulse is equivalent to -- placed before the pulse (side="before")
    or after it (side="after") -- and so is what a delay can correct. The
    transverse components are the part no delay on that side can correct.

optimize_delay(run_sequence, objective, bounds)
    The exact answer: propagate the whole sequence and maximize the objective
    over the delay. Use it to confirm the first-order estimate.
"""

import numpy as np
from scipy.optimize import minimize_scalar

def _rodrigues_matrix(axis, angle):
    """Rotation matrix for `angle` (rad) about `axis` (right-handed)."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

def effective_coupling_generator(rf, dt, side="before"):
    """
    rf   : complex array, wx + i*wy in rad/ms (same convention as
           RawShapedPulseSegment/ShapePulseSegment's per-timestep RF --
           already includes 2*pi*Gamma if it came from Pulse.calibrated_rf())
    dt   : sample spacing, ms.
    side : "before" or "after" -- which side of the pulse to express the
           coupling on.

    Returns (cx, cy, cz) in ms: the time integral of the toggling-frame
    operator U(t)^dag Iz U(t) = cx Ix + cy Iy + cz Iz, where U(t) is the RF
    propagator up to time t. To first order in J, the pulse with coupling on is

        side="before":  U_rf(T) * exp(-i*kappa*(cx Ix + cy Iy + cz Iz) Sz)
        side="after":   exp(-i*kappa*(cx Ix + cy Iy + cz Iz) Sz) * U_rf(T)

    with H_J = kappa * IzSz. So cz is an equivalent free-evolution delay on
    that side of an ideal pulse. An excitation pulse (E-BURP-2) carries its
    delay BEFORE itself; its time-reversed flip-back partner carries the same
    delay AFTER itself.

    The rotations are composed in the order the propagator requires,
    (R_k ... R_1)^T e_z. Rotating M forward from +z instead gives the same cz
    at every instant (a scalar equals its transpose) but the wrong transverse
    components -- for a phase-modulated pulse wrong in magnitude, not only
    sign. tests/test_pulse_diagnostics.py holds this to the exact propagator.
    """
    if side not in ("before", "after"):
        raise ValueError(f"side must be 'before' or 'after', not {side!r}")
    ez = np.array([0.0, 0.0, 1.0])
    L = np.eye(3)                       #R_1^T R_2^T ... R_n^T
    traj = [ez.copy()]
    for w in rf:
        axis = np.array([w.real, w.imag, 0.0])
        rate = np.linalg.norm(axis)
        if rate > 1e-12:
            L = L @ _rodrigues_matrix(axis, rate * dt).T
        traj.append(L @ ez)
    traj = np.asarray(traj)
    # manual trapezoid, avoid np.trapz/np.trapezoid version churn
    c = dt * (0.5 * traj[0] + traj[1:-1].sum(axis=0) + 0.5 * traj[-1])
    if side == "after":
        c = L.T @ c         # the full rotation R_n ... R_1 applied to c

    return tuple(c)

def optimize_delay(run_sequence, objective, bounds):
    """
    run_sequence(Delta) -> sigma_final   builds and propagates the real
        sequence for a given free-evolution delay Delta, with J-coupling
        active throughout (including during shaped pulses).
    objective(sigma_final) -> float      what to MAXIMIZE, e.g. the
        transfer amplitude of interest.
    bounds : (low, high) search range for Delta, ms.

    Returns the scipy OptimizeResult (`.x` is the optimal Delta). Raises if
    the optimum lies on a bound: a bounded search that stops on its own
    bound has found the bound, not an optimum.
    """
    def neg_objective(Delta):
        return -objective(run_sequence(Delta))
    result = minimize_scalar(neg_objective, bounds=bounds, method='bounded')
    margin = 1e-3 * (bounds[1] - bounds[0])
    if not (bounds[0] + margin < result.x < bounds[1] - margin):
        raise ValueError (f"optimum {result.x:.4f} ms is on the search bound {tuple(bounds)}; widen the bounds")

    return result
