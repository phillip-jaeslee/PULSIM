import numpy as np
from rotation import Rx, Ry, Rz

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
"""

def bloch_rotate(M_init, T, B):
    Gamma = 42.58  # kHz/mT MHz/T
    flip = 2*np.pi* Gamma * np.linalg.norm(B) * T
    eta = np.arccos(B[2] / (np.linalg.norm(B) +np.finfo(np.float64).eps))
    theta = np.arctan2(B[1], B[0])
    M_final = Rz(-theta)@Ry(-eta)@Rz(flip)@Ry(eta)@Rz(theta) @ M_init

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
    M_all = np.zeors(3, Nt)

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
