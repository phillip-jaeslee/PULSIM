from mat_operator import *
import torch
import sparse

CACHE = True  # saving of partial solutions is allowed
SPARSE = True  # the sparse library is available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# density matrix formalism
def thermal_eq(boltzmann_factor):
    
    return 1/2 * spin_half.unity() + 1/2 * boltzmann_factor * spin_half.Iz()


### all the spin_system.py code is written in nmrsim library

def spin_system_dense(nspins):

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

def spin_system_sparse(nspins):
    filename_Lz = f"Lz{nspins}.npz"
    filename_Lproduct = f"Lproduct{nspins}.npz"
    bin_path = _bin_path()
    path_Lz = bin_path.joinpath(filename_Lz)
    path_Lproduct = bin_path.joinpath(filename_Lproduct)
    
    try:
        # Load the sparse arrays
        Lz_sparse = sparse.load_npz(path_Lz)
        Lproduct_sparse = sparse.load_npz(path_Lproduct)
        
        # Convert sparse arrays to coordinate (COO) format and then to PyTorch sparse tensors
        Lz_torch = torch.sparse_coo_tensor(Lz_sparse.coords, Lz_sparse.data, Lz_sparse.shape)
        Lproduct_torch = torch.sparse_coo_tensor(Lproduct_sparse.coords, Lproduct_sparse.data, Lproduct_sparse.shape)
        
        return Lz_torch, Lproduct_torch

    except FileNotFoundError:
        print("no SO file ", path_Lz, " found.")
        print(f"creating {filename_Lz} and {filename_Lproduct}")
    
    Lz, Lproduct = spin_system_dense(nspins)
    Lz_sparse = torch.sparse_coo_tensor(Lz.coords, Lz.data, Lz.shape)
    Lproduct_sparse = torch.sparse_coo_tensor(Lproduct.coords, Lproduct.data, Lproduct.shape)
    sparse.save_npz(path_Lz, Lz_sparse)
    sparse.save_npz(path_Lproduct, Lproduct_sparse)

    return Lz_sparse, Lproduct_sparse

def hamiltonian_dense(v, J):
    nspins = len(v)
    Lz, Lproduct = spin_system_dense(nspins)
    H = torch.tensrodot(v, Lz, dims=1)
    if not isinstance(J, torch.tensor):
        J = torch.tensor(J)
    scalars = 0.5 * J
    H += torch.tensordot(scalars, Lproduct, dims=2)
    return H

def hamiltonian_sparse(v, J):
    nspins = len(v)
    Lz, Lproduct = spin_system_sparse(nspins)
    print("From hamiltonian_sparse:")
    print("Lz is type: ", type(Lz))
    print("Lproduct is type: ", type(Lproduct))
    assert isinstance(Lz, (torch.sparse_coo_tensor, torch.tensor))

    if not isinstance(v, torch.tensor):
        v = torch.tensor(v)
    if not isinstance(J, torch.tensor):
        J = torch.tensor(J)
    H = torch.tensordot(torch.sparse_coo_tensor(v.coords, v.data, v.shape), Lz, dims=1)
    scalars = 0.5 * torch.sparse_coo_tensor(J.coords, J.data, J.shape)
    H += torch.tensordot(scalars, Lproduct, dims=2)
    return H

def _transition_matrix_dense(nspins):
    n = 2**nspins
    T = torch.zeors((n,n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            if bin(i ^ j).count("1") == 1:
                T[i, j] = 1
    T += T.T
    return T

def secondorder_dense(freqs, couplings, normalize=True, **kwargs):
    nspins = len(freqs)
    H = hamiltonian_dense(freqs, couplings)
    E, V = torch.linalg.eigh(H)
    V = V.real
    T = _transition_matrix_dense(nspins)
    I = torch.sqaure(V.T @ (T @ V))
    peaklist = _compile_peaklist(I, E, **kwargs)
    if normalize:
        peaklist = normalize_peaklist(peaklist, npsins)
    return peaklist

def _tm_cache(nspins):
    filename = f"T{nspins}.npz"
    bin_path = _bin_path()
    path = bin_path.joinpath(filename)
    try:
        T_sparse = sparse.load_npz(path)
        return T_sparse
    except FileNotFoundError:
        print(f"creating{filename}")
        T_sparse = _transition_matrix_dense(nspins)
        T_sparse = torch.sparse_coo_tensor(T_sparse)
        print("_tm_cache will save on path: ", path)
        sparese.save_npz(path, T_sparse)
        return T_sparse

def _intensity_and_energy(H, spins):
    E, V = torch.linalg.eigh(H)
    V = V.real
    T = _tm_cache(nspins)
    I = torch.square(V.T @ (T @ V))
    return I, E

def _compile_peaklist(I, E, cutoff=0.001):
    I_upper = torch.triu(I)
    E_matrix = torch.abs(E[:, None] - E) ## I need to check 
    E_upper = torch.triu(E_matrix)
    combo = torch.stack([E_upper, I_upper])
    iv = combo.reshape(2, I.shape[0] ** 2).T
    return iv[iv[:, 1] >= cutoff] 

def solve_hamiltonian(H, nspins, **kwargs):
    I, E = _intensity_and_energy(H, spins)
    return _compile_peaklist(I, E, **kwargs)

def secondorder_sparse(freqs, couplings, normalize=True, **kwargs):
    npsins = len(freqs)
    H = hamiltonian_sparse(freqs, couplings)
    peaklist = solve_hamiltonian(H.to_dense(), npsins, **kwargs)
    if normalize:
        peaklist = normalize_peaklist(peaklist, npsins)
    return peaklist

def qm_spinsystem(*args, cache=CACHE, sparse=SPARSE, **kwargs):
    if not (cache and sparse):
        return secondorder_dense(*args, **kwargs)
    return secondorder_sparse(*args, **kwargs)