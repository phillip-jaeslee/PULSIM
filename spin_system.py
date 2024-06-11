from mat_operator import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# density matrix formalism
def thermal_eq(boltzmann_factor):
    
    return 1/2 * spin_half.unity() + 1/2 * boltzmann_factor * spin_half.Iz()

import torch

def spin_system(nspins):

    L = torch.empty((3, nspins, 2**nspins, 2**nspins), dtype=torch.complex128)  # TODO: consider other dtype?
    for n in range(nspins):
        Lx_current = torch.tensor([1], dtype=torch.complex128)
        Ly_current = torch.tensor([1], dtype=torch.complex128)
        Lz_current = torch.tensor([1], dtype=torch.complex128)

        for k in range(nspins):
            if k == n:
                Lx_current = torch.kron(Lx_current, spin_half.Ix())
                Ly_current = torch.kron(Ly_current, spin_half.Iy())
                Lz_current = torch.kron(Lz_current, spin_half.Iz())
            else:
                Lx_current = torch.kron(Lx_current, spin_half.unity())
                Ly_current = torch.kron(Ly_current, spin_half.unity())
                Lz_current = torch.kron(Lz_current, spin_half.unity())

        L[0][n] = Lx_current
        L[1][n] = Ly_current
        L[2][n] = Lz_current

    L_T = L.permute(1, 0, 2, 3)
    Lproduct = torch.tensordot(L_T, L, dims=([1, 3], [0, 2])).permute(0, 2, 1, 3)

    return L[2], Lproduct
