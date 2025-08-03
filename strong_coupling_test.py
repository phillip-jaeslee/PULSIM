from mat_operator import spin_half
import torch
import sparse
import numpy

import sys
import temp
from pathlib import Path
import scipy.sparse

from mat import normalize_peaklist


Lx_current = torch.tensor([1], dtype=torch.complex128)

print(Lx_current)

Lx_current = torch.kron(Lx_current, spin_half.Ix())

print(Lx_current)

Lx_current = torch.kron(Lx_current, spin_half.unity())

print(Lx_current)

Lx_current = torch.kron(Lx_current, spin_half.unity())

print(Lx_current)

Lx_current = torch.kron(Lx_current, spin_half.Ix())

print(Lx_current)

nspins = 2
L = torch.empty((3, nspins, 2**nspins, 2**nspins), dtype=torch.complex128)  # TODO: consider other dtype?

for n in range(nspins):
    Lx_current = torch.tensor([1], dtype=torch.complex128)
    Ly_current = torch.tensor([1], dtype=torch.complex128)
    Lz_current = torch.tensor([1], dtype=torch.complex128)
    print("passing n", n)
    for k in range(nspins):
        if k == n:
            Lx_current = torch.kron(Lx_current, spin_half.Ix())
            Ly_current = torch.kron(Ly_current, spin_half.Iy())
            Lz_current = torch.kron(Lz_current, spin_half.Iz())
            print("passing k inside if", k)
        else:
            Lx_current = torch.kron(Lx_current, spin_half.unity())
            Ly_current = torch.kron(Ly_current, spin_half.unity())
            Lz_current = torch.kron(Lz_current, spin_half.unity())
            print("passing k inside if", k)

        print("passing k", k)
    print(Lx_current)
    L[0][n] = Lx_current
    L[1][n] = Ly_current
    L[2][n] = Lz_current

print(L)

L_T = L.permute(1, 0, 2, 3)

print(L_T)
