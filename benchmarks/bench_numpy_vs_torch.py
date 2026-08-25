"""
benchmarks/bench_numpy_vs_torch.py

Compares NumpyBackend (bloch_rotate_batch, CPU) against TorchBackend
(torch_bloch_rotate) for a single rotate() call across a range of
n_offsets. Run locally where torch is installed:

    python benchmarks/bench_numpy_vs_torch.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from PULSIM.backend import NumpyBackend, TorchBackend

Gamma = 42.577478
rng = np.random.default_rng(0)


def bench(n_offsets, n_repeats=200, torch_device="cpu"):
    B = rng.normal(size=(n_offsets, 3))
    M = rng.normal(size=(3, n_offsets))
    M /= np.linalg.norm(M, axis=0, keepdims=True)
    T = 0.005

    np_backend = NumpyBackend(Gamma=Gamma)
    torch_backend = TorchBackend(device=torch.device(torch_device), Gamma=Gamma)

    # warm-up -- lets torch do any lazy init/allocation before timing
    np_backend.rotate(M, T, B, "x")
    torch_backend.rotate(M, T, B, "x")

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        np_backend.rotate(M, T, B, "x")
    t_numpy = (time.perf_counter() - t0) / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        torch_backend.rotate(M, T, B, "x")
    t_torch = (time.perf_counter() - t0) / n_repeats

    return t_numpy, t_torch


def run_sweep(torch_device="cpu"):
    print(f"\ntorch device: {torch_device}")
    print(f"{'n_offsets':>10} | {'numpy (ms)':>12} | {'torch (ms)':>12} | {'torch/numpy':>12}")
    print("-" * 55)
    for n in (1, 10, 50, 100, 500, 2000, 10000):
        reps = 200 if n <= 500 else 20
        t_numpy, t_torch = bench(n, n_repeats=reps, torch_device=torch_device)
        ratio = t_torch / t_numpy
        faster = "numpy" if ratio > 1 else "torch"
        print(f"{n:>10} | {t_numpy*1000:>12.4f} | {t_torch*1000:>12.4f} | {ratio:>10.2f}x ({faster} faster)")


if __name__ == "__main__":
    run_sweep("cpu")
    if torch.cuda.is_available():
        run_sweep("cuda")
    else:
        print("\nNo CUDA device visible -- only CPU-vs-CPU measured. Torch's advantage "
              "is expected mainly on GPU, where per-offset kernel-launch overhead disappears "
              "into one batched call across thousands of offsets; on CPU it often loses to "
              "numpy for small/medium batches due to torch's own per-call dispatch overhead.")