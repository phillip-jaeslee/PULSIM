import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class spin_half:

    # Anglular momentum operators
    def Ix():
        return torch.tensor([[0, 1/2], [1/2, 0]], dtype=torch.complex128)

    def Iy():
        return torch.tensor([[0, -1j/2], [1j/2, 0]], dtype=torch.complex128)


    def Iz():
        return torch.tensor([[1/2, 0], [0, -1/2]], dtype=torch.complex128)


    # Unity operator
    def unity():
        return torch.eye(2, dtype=torch.complex128)

    # Shift operators
    def Ip():
        return torch.stack([
            torch.stack([torch.zeros(1), torch.ones(1)], dim=0),
            torch.stack([torch.zeros(1), torch.zeros(1)], dim=0)
        ], dim=0).to(device) / 2

    def In():
        return torch.stack([
            torch.stack([torch.zeros(1), torch.zeros(1)], dim=0),
            torch.stack([torch.ones(1), torch.zeros(1)], dim=0)
        ], dim=0).to(device) / 2

    # Projection operators
    def Ia():
        return torch.add(spin_half.Iz(), spin_half.unity(), alpha=1/2)

    def Ib():
        return torch.add(-spin_half.Iz(), spin_half.unity(), alpha=1/2)

    # Density matrix
    def thermal_eq(GAMMA, B0, T=293.7):
        return torch.add(spin_half.Iz() * 1/2 * boltzmann_factor(GAMMA, B0, T), spin_half.unity(), alpha=1/2, dtype=torch.complex128).to(device)

def boltzmann_factor(GAMMA, B0, T=293.7):
    h_bar = 1.05457182E-34
    kB =  1.380649E-23
    return h_bar * GAMMA * B0 / (kB * T) 
    
class torch_rot:
    """
    Rotation operators based on PyTorch library
    """

    def Rx(theta):
        return torch.stack([
            torch.stack([torch.ones_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)], dim=0),
            torch.stack([torch.zeros_like(theta), torch.cos(theta), -torch.sin(theta)], dim=0),
            torch.stack([torch.zeros_like(theta), torch.sin(theta), torch.cos(theta)], dim=0)
        ], dim=0).to(device)

    def Ry(theta):
        return torch.stack([
            torch.stack([torch.cos(theta), torch.zeros_like(theta), torch.sin(theta)], dim=0),
            torch.stack([torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta)], dim=0),
            torch.stack([-torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)], dim=0)
        ], dim=0).to(device)

    def Rz(theta):
        return torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta)], dim=0),
            torch.stack([torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)], dim=0),
            torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=0)
        ], dim=0).to(device)

    def Rot(theta):
        return torch.stack([
            torch.stack([torch.cos(theta), torch.sin(theta)], dim=0),
            torch.stack([-torch.sin(theta), torch.cos(theta)], dim=0)
        ], dim=0).to(device)

class cpu_rot:
    """
    Rotation operators based on Numpy library
    """

    def Rx(flip):
        Rx = np.array ([[1, 0, 0],
                    [0, np.cos(flip), -np.sin(flip)],
                    [0, np.sin(flip), np.cos(flip)]])
        return Rx

    def Ry(flip):
        Ry = np.array ([[np.cos(flip), 0, np.sin(flip)],
                    [0, 1, 0],
                    [-np.sin(flip), 0, np.cos(flip)]])
        return Ry

    def Rz(flip):
        Rz = np.array ([[np.cos(flip), -np.sin(flip), 0],
                    [np.sin(flip), np.cos(flip), 0],
                    [0, 0, 1]])
        return Rz

    def Rot(flip):
        Rot = np.array ([[np.cos(flip), np.sin(flip)],
                        [-np.sin(flip), np.cos(flip)]])
        return Rot

class cpu_rot_batch:
    """
    Batched rotation operators based on NumPy, vectorized over a leading
    n_offsets axis. Each Rx/Ry/Rz takes an array of angles, shape (n,),
    and returns a stack of rotation matrices, shape (n, 3, 3), so that
    R1 @ R2 (numpy's batched matmul for (...,3,3) arrays) chains n
    independent rotations in one call instead of a Python loop.
    """

    def Rx(flip):
        flip = np.asarray(flip, dtype=float)
        c, s = np.cos(flip), np.sin(flip)
        zero, one = np.zeros_like(flip), np.ones_like(flip)
        return np.stack([
            np.stack([one,  zero,  zero], axis=-1),
            np.stack([zero,    c,    -s], axis=-1),
            np.stack([zero,    s,     c], axis=-1),
        ], axis=-2)

    def Ry(flip):
        flip = np.asarray(flip, dtype=float)
        c, s = np.cos(flip), np.sin(flip)
        zero, one = np.zeros_like(flip), np.ones_like(flip)
        return np.stack([
            np.stack([   c, zero,    s], axis=-1),
            np.stack([zero,  one, zero], axis=-1),
            np.stack([  -s, zero,    c], axis=-1),
        ], axis=-2)

    def Rz(flip):
        flip = np.asarray(flip, dtype=float)
        c, s = np.cos(flip), np.sin(flip)
        zero, one = np.zeros_like(flip), np.ones_like(flip)
        return np.stack([
            np.stack([   c,   -s, zero], axis=-1),
            np.stack([   s,    c, zero], axis=-1),
            np.stack([zero, zero,  one], axis=-1),
        ], axis=-2)