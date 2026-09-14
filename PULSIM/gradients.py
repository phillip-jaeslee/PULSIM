"""
gradients.py -- gradients as position-dependent resonance offsets.

A linear field gradient adds a position-dependent logitudinal field. In the
rotating frame that is indistinguishable from a resonance offset, and PULSIM's
kernel already propagates a whole batch of offsets in one call. So a gradient
needs no new propagator: it is a map from position to offset on the way in,
and an average over positions on the way out.

    df(r) = df0 + Gamma (G . r)                        [kHz]

with Gamma in kHz/mT, G in mT/m and r in m, so G.r is in mT add the product is
in kHz -- no conversion factor, and consistent with the offset sign convention
of PHYSICS_SPECIFICATION.md section 1.5 (positive df gives positive Omega_z)

Scope: stataic or piecewise-constant gradients. Pulse.appy hold df fixed for
every timestep, so a gradient that varies WITHIN on call is not representable
and is not claimed; piecewise-constant means one call per constatn segment.
Diffusion, flow, concomitant fields, gradient nonlinearity and eddy currents
are all outside the mode.
"""

import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}

def gradient_offsets(G, r, Gamma):
    """Resonance offsets produced by a linear gradient.

    G       : (3,) gradient vector [mT/m]
    r       : (n_positions, 3) positions [m]
    Gamma   : reduced gyromagnetic ratio (kHz/mT)
    returns : (n_positions, ) offsets [kHz]

    Add a base offset yourself if you want one -- df = df0 + gradient_offsets(...)
    Keeping the two separate is what lets a gradient be composed with anything
    else that produces an offset.    
    """
    G = np.asarray(G, dtype=float)
    r = np.asarray(r, dtype=float)

    if G.shape != (3,):
        raise ValueError(f"G must be a 3-vector [mT/m]; got shape {G.shape}")
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r must be (n_positions, 3) [m]; got shape {r.shape}")

    return Gamma * (r @ G)

def uniform_positions(length, n_positions, axis="z", center=0.0):
    """Midpoint-rule sample of a 1D slab.

    length      : slab thickness [m]
    n_positions : number of samples
    axis        : "x", "y" or "z" -- the other two coordinates are zero
    center      : slab center along that axis [m]
    returns     : (n_positions, 3) positions [m]
    
    Samples sit at CELL CENTERS, not at the slab edges:

        z_j = center + length * ((j + 0.5)/n - 0.5),   j = 0 .. n-1

    That choice is deliberate and load-bearing. Over k complete twists of
    dephasing the sampled phases are n-th roots of unity, so their mean is
    EXACTLY zero whenever n does not divide k -- complete dephasing comes out
    at machine precision rather than with an 0(1/n) residue. Endpoint sampling
    double-counts one edge and does not have this property.

    One consequence to the aware of: the sampled offsets span
    Gamma*|G|*length*(1 - 1/n), not Gamma*|G|*length. The slab spans the full
    range; the sample points sit half a cell inside each face.
    """
    if n_positions < 1:
        raise ValueError(f"n_positions must be at least 1; got {n_positions}")
    if length < 0:
        raise ValueError(f"length must be non-negative [m]; got {length}")
    if axis not in _AXES:
        raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}")

    j = np.arange(n_positions)
    coord = center + length * ((j + 0.5) / n_positions - 0.5)

    r = np.zeros((n_positions, 3))
    r[:, _AXES[axis]] = coord

    return r

def ensemble_average(M, weights=None):
    """Average a spatial ensemble down to the bulk observable.

    M       : (3, n_positions) magnetization, one column per position
    weights : (n_positions,) optional, e.g. a slice profile. Normalized
              internally, so they need not sum to one.
    returns : (3,) bulk magnetization vector

    This is where a gradient experiment actually produces a signal: a real
    receiver sees the volume integral, not any isochromat.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != 3:
        raise ValueError(f"M must be (3, n_positions); got shape {M.shape}")

    if weights is None:
        return M.mean(axis=1)

    w = np.asarray(weights, dtype=float)
    if w.shape != (M.shape[1],):
        raise ValueError(f"weights must be ({M.shape[1]},); got shape {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if total <= 0:
        raise ValueError("weigths must not sum to zero")

    return (M * w).sum(axis=1) / total

def ideal_spoil(M):
    """Zero Mx and My exactly, leaving Mz untouched. AN IDEALIZATION.

    THis is NOT what a crusher gradient does. A real gradient dephases the
    ensemble, and the transverse signal vanishes only because the isochromats
    cancel on averaeging -- each one still carries full transverse
    magnetizaton, and a subsequent gradient can bring it back (see the
    gradient-echo test). Use this when you want to assert "no transverse
    coherence survives" without paying for an ensemble, and reach for
    gradient_offsets + ensemble_average when the physics of the dephasing is
    the point.

    Accepts a single (3,) vector or a (3, n) batch. Returns a COPY; the
    argument is not modified.
    """
    M = np.asarray(M, dtype=float)
    if M.shape[0] != 3:
        raise ValueError(f"M must have 3 rows; got shape {M.shape}")

    out = M.copy()
    out[0] = 0.0
    out[1] = 0.0
    return out