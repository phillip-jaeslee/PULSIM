import numpy as np

# NOTE: torch is deliberately NOT imported here. bloch.py is imported by
# backend.py and by PULSIM/__init__.py, so a top-level `import torch` would
# make the whole package require torch. Only torch_bloch_rotate needs it, and
# it imports it lazily at call time (see below).
from PULSIM.mat_operator import cpu_rot, cpu_rot_batch

## Bloch relaxation
# calculation of Bloch equation for time T
"""
Parameters
M_init  : initial magentization
T       : duraion [ms]
M0      : equilibrium magnetization (defualt = 1)
T1      : logitudianl relaxation time [ms]
T2      : transverse relaxation time [ms]
A_relax : relaxation array
M_final : final magnetization
"""

def bloch_relax(M_init, T, M0, T1, T2):
    A_relax = np.array([[np.exp(-T/T2), 0, 0],
                    [0, np.exp(-T/T2), 0],
                    [0, 0, np.exp(-T/T1)]])
    brecover = np.array([0, 0, M0*(1-np.exp(-T/T1))])
    

    M_final = A_relax * M_init + brecover

    return M_final

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
    from PULSIM.mat_operator import require_torch, torch_rot
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
