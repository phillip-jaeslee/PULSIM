import numpy as np

# NOTE: torch is deliberately NOT imported here. bloch.py is imported by
# backend.py and by PULSIM/__init__.py, so a top-level `import torch` would
# make the whole package require torch. Only torch_bloch_rotate needs it, and
# it imports it lazily at call time (see below).
from .mat_operator import cpu_rot, cpu_rot_batch

def _decay(T, tau):
    """exp(-T/tau), with tau=None meaning infinite -- i.e. no decay.

    A relaxation time of None means "this component does not relax".
    Zero is an erro, not shorthand for instantaneous.
    """
    if tau is None or tau == np.inf:
        return 1.0
    if tau <= 0:
        raise ValueError(f"relaxation time must be positive, or None for infinite; got {tau}")

    return float(np.exp(-T/tau))

def relaxation_matrix(T1=None, T2=None):
    """R = diag(1/T2, 1/T2, 1/T1) in ms^-1. None -> a rate of zero."""
    for name, tau in (("T1", T1), ("T2", T2)):
        if tau is not None and tau != np.inf and tau <= 0:
            raise ValueError(f"{name} must be positive, or None for infinite; got {tau}")

    r2 = 0.0 if (T2 is None or T2 == np.inf) else 1.0 / T2
    r1 = 0.0 if (T1 is None or T1 == np.inf) else 1.0 / T1

    return np.diag([r2, r2, r1])

def bloch_relax(M_init, T, M0=1.0, T1=None, T2=None):
    """Free relaxation over a duration T, with no applied field.

    Closed form of PHYSICS_SPECIFICATION.md section 1.8 item 5:

        Mx(t) = Mx(0) exp(-t/T2)
        My(t) = My(0) exp(-t/T2)
        Mz(t) = M0 + [Mz(0) - M0] exp(-t/T1)

    Accepts a single (3,) vector or a (3, n) batch: the decay factors are
    scalars, so the same expression covers both.

    T, T1, T2 : ms. None means infinite.
    """
    M = np.asarray(M_init, dtype=float)
    e2 = _decay(T, T2)
    e1 = _decay(T, T1)

    out = np.empty_like(M)
    out[0] = e2 * M[0]
    out[1] = e2 * M[1]
    out[2] = e1 * M[2] + M0 * (1.0 - e1)
    return out

def affine_propagate(M_init, dt, Omega, T1=None, T2=None, M0=1.0):
    """EXACT propagation over one constant-field step. Reference implementation.

    Correct but not fast -- a 4x4 matrix exponential per call. The production
    path uses Strang splitting; a convergence test asserts the two agree.

    The spec's equation (section 1.7)

        dM/dt = Omega x M - R (M - M_eq)
    
    is linear inhomogeneous, dM/dt = L M + c, with

        L = Omega_hat - R   Omega_hat M = Omega x M
        c = R M_eq          M_eq = (0, 0, M0)
    
    so the exaact solution over dt is affine: M -> A M + b. Both come from one
    4x4 exponential, which avoids inverting L -- and L is singular in exactly
    the case that matters most, T1 = T2 = None:

        G = [[L, c],        expm(G dt) = [[A, b],
             [0, 0]]                      [0, 1]]
    
    Omega   : (3,) angular frequency vector, rad/ms (Omega = 2*pi*gamma_bar*B)
    dt      : ms
    """
    from scipy.linalg import expm

    wx, wy, wz = np.asarray(Omega, dtype=float)

    # Omega x M, written directly from the compnent equations in section 1.7
    Omega_hat = np.array([[0.0, -wz, wy],
                          [wz, 0.0, -wx],
                          [-wy, wx, 0.0]])

    R = relaxation_matrix(T1, T2)
    L = Omega_hat - R
    c = R @ np.array([0.0, 0.0, M0])

    G = np.zeros((4, 4))
    G[:3, :3] = L
    G[:3, 3] = c

    E = expm(G * dt)

    return E[:3, :3] @ np.asarray(M_init, dtype=float) + E[:3, 3]

def bloch_relax_rotate_batch(M_init, dt, B, angle, Gamma, T1=None, T2=None, M0=1.0):
    """One time step with rotation and relaxation, Strang-split

        relax(dt/2) -> rotate(dt) -> relax(dt/2)
    
    Second-order accurate in dt. Rotation and relaxation do not commute unless
    T1 == T2, so a single step is not exact; affine_propagate is the exact
    reference, and tests/test_relaxation.py asserts convergence to it.

    Relaxation is the cheap half (closed form, no matrix), so it is the part
    that gets done twice.

    With T1 = T2 = None this returns bloch_rotate_batch's result BIT FOR BIT,
    not approximately -- the early return guarantees it rather than relying on
    the identity relaxation being exactly 1.0.
    """
    if T1 is None and T2 is None:
        return bloch_rotate_batch(M_init, dt, B, angle, Gamma)

    M = bloch_relax(M_init, 0.5 * dt, M0, T1, T2)
    M = bloch_rotate_batch(M, dt, B, angle, Gamma)
    return bloch_relax(M, 0.5 * dt, M0, T1, T2)

## Bloch relaxation in batch
# calculation of Bloch equation on a batch of timepoints
"""
Parameters - see Bloch relaxtion
Ts      : time points [ms]
"""
def bloch_relax_batch(M_init, Ts, M0, T1, T2):
    Mx = np.exp(-Ts/T2) * M_init[0]
    My = np.exp(-Ts/T2) * M_init[1]
    Mz = np.exp(-Ts/T1) * M_init[2] + M0*(1-np.exp(-Ts/T1))

    M_final = np.array([Mx, My, Mz])

    return M_final

## Bloch rotation
# calculation of Bloch rotation
"""
Parameters
M_init  : initial magnetization
T       : duration [ms]
B       : [Bx, By, Bz] - magnetic field [mT]
M_final : final magnetization
angle   : flip angle among coordinates (x, y, z)
"""

def bloch_rotate(M_init, T, B, angle, Gamma):
    # Gamma kHz/mT MHz/T

    flip = 2 * np.pi * Gamma * np.linalg.norm(B) * T
    if angle == "x":
        phi = np.arctan2(B[1], B[0])
    elif angle == "y":
        phi = np.arctan2(B[1], B[0]) + np.pi/2
    elif isinstance(angle, (int, float)):
        phi = np.arctan2(B[1], B[0]) + float(angle)
    else:
        raise NotImplementedError(
            f'bloch_rotate("{angle}") has no verified formula -- "z" was never used '
            f'anywhere in the codebase and has no independent ground truth to check '
            f'against. Only "x" and "y" (real NMR pulse phases) are implemented.'
        )
    theta = np.arctan2(np.hypot(B[0], B[1]), B[2])
    M_final = cpu_rot.Rz(phi) @ cpu_rot.Ry(theta) @ cpu_rot.Rz(flip) @ cpu_rot.Ry(-theta) @ cpu_rot.Rz(-phi) @ M_init
    return M_final

def bloch_rotate_batch(M_init, T, B, angle, Gamma):
    """
    Batched version of bloch_rotate: rotates a whole batch of offsets in one
    call instead of looping over bloch_rotate per offset (see NumpyBackend).

    Parameters
    M_init : (3, n_offsets) starting magnetization, one column per offset
    T      : duration [ms] -- shared across the batch (one RF time-step)
    B      : (n_offsets, 3) -- [Bx, By, Bz] per offset [mT]
    angle  : "x", "y", or a float phase [rad] -- same axis for every offset
    M_final: (3, n_offsets)
    """
    B = np.asarray(B, dtype=float)
    M_init = np.asarray(M_init, dtype=float)

    norm = np.linalg.norm(B, axis=1)
    flip = 2 * np.pi * Gamma * norm * T

    if angle == "x":
        phi = np.arctan2(B[:, 1], B[:, 0])
    elif angle == "y":
        phi = np.arctan2(B[:, 1], B[:, 0]) + np.pi / 2
    elif isinstance(angle, (int, float)):
        phi = np.arctan2(B[:, 1], B[:, 0]) + float(angle)
    else:
        raise NotImplementedError(
            f'bloch_rotate("{angle}") has no verified formula -- "z" was never used '
            f'anywhere in the codebase and has no independent ground truth to check '
            f'against. Only "x" and "y" (real NMR pulse phases) are implemented.'
        )
    theta = np.arctan2(np.hypot(B[:, 0], B[:, 1]), B[:, 2])

    R = (cpu_rot_batch.Rz(phi) @ cpu_rot_batch.Ry(theta) @ cpu_rot_batch.Rz(flip) @ cpu_rot_batch.Ry(-theta) @ cpu_rot_batch.Rz(-phi))

    M_final = np.einsum('nij,jn->in', R, M_init)
    return M_final


## Bloch rotation
# calculation of Bloch rotation
"""
Parameters
M_init  : initial magnetization
T       : duration [ms]
B1      : RF amplitude, B1X+iB1Y [mT]
M_final : final magnetization
"""

def torch_bloch_rotate(M_init, T, B, angle, Gamma):
    from .mat_operator import require_torch, torch_rot
    torch = require_torch()

    flip = 2 * torch.pi * Gamma * torch.norm(B, dim=1) * T
    if angle == "x":
        phi = torch.atan2(B[:, 1], B[:, 0])
    elif angle == "y":
        phi = torch.atan2(B[:, 1], B[:, 0]) + torch.pi / 2
    elif isinstance(angle, (int, float)):
        phi = torch.atan2(B[:, 1], B[:, 0]) + float(angle)
    else:
        raise NotImplementedError(
            f'torch_bloch_rotate("{angle}") has no verified formula -- "z" was never used '
            f'anywhere in the codebase and has no independent ground truth to check '
            f'against. Only "x" and "y" (real NMR pulse phases) are implemented.'
        )
    theta = torch.atan2(torch.hypot(B[:, 0], B[:, 1]), B[:, 2])

    # torch.permute (2, 0, 1) = change the order of dimension
    # torch.bmm = matrix multiplication (not available for broadcast)
    R = torch.bmm(torch_rot.Rz(phi).permute(2, 0, 1), torch_rot.Ry(theta).permute(2, 0, 1))
    R = torch.bmm(R, torch_rot.Rz(flip).permute(2, 0, 1))
    R = torch.bmm(R, torch_rot.Ry(-theta).permute(2, 0, 1))
    R = torch.bmm(R, torch_rot.Rz(-phi).permute(2, 0, 1))    

    # Convert M_init to the appropriate data type before the bmm operation
    M_init = M_init.to(torch.float32)

    return torch.bmm(R, M_init.unsqueeze(2)).squeeze(2)

def bloch_rftip(M_init, T, B1):

    M_final = bloch_rotate(M_init, T, [np.real(B1), np.imag(B1), 0])

    return M_final


### Bloch_simulation
# compute Bloch simulation for a pulse sequence
"""
% INPUTS
%	Mstart - initial magnetization
%	dt - time step between points in B1 and G [ms] 
%	B1 - RF vector, B1X + i B1Y [mT], defined at each time point in T
%	G - Gradient field vector [mT/m], defined for Gx,Gy, and Gz at each time point in T
%	M0 - equilibrium magnetization (default = 1)
%	T1 - longitudinal relaxation time [ms]
%	T2 - transverse relaxation time [ms]
%	r - positions at which to evaluate simulation [m]  (JUST POSITION)
%	df - off-resonance frequencies to evaluate simulation [kHz] (JUST one off-resonance)
% OUTPUTS
%   Mall - magnetization
"""

def bloch_simulation(M_init, dt, B1, G, M0, T1, T2, r, df):
    Nt = max(B1.shape)
    M_all = np.zeros(3, Nt)

    for i in range(1, Nt):
        if i == 1:
            M_temp1 = M_init
        else:
            M_temp1 = M_all[:, i-1]
        
        M_temp2 = bloch_rotate(M_temp1, dt, [np.real(B1[i]), np.imag(B1[i]), G[:,i]*r + df])

        M_all[:, i] = bloch_relax(M_temp2, dt, M0, T1, T2)

    return M_all

def spoil_magnetization(M_init):
    M_final = M_init
    M_final[1:-1, :] = 0

    return M_final
