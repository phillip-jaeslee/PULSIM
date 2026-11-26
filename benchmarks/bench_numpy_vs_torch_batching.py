"""
benchmarks/bench_bloch_rotate_batching.py

Compares the old per-offset Python loop over bloch_rotate (what
NumpyBackend.rotate used to do) against bloch_rotate_batch (what it does
now) for a single rotate() call across a range of n_offsets.

    python benchmarks/bench_bloch_rotate_batching.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from PULSIM.bloch import bloch_rotate, bloch_rotate_batch

Gamma = 42.577478
rng = np.random.default_rng(0)


def loop_version(M, T, B, angle, Gamma):
    n = M.shape[1]
    out = np.zeros_like(M)
    for f in range(n):
        out[:, f] = bloch_rotate(M[:, f], T, B[f, :], angle, Gamma)
    return out


def bench(n_offsets, n_repeats=200):
    B = rng.normal(size=(n_offsets, 3))
    M = rng.normal(size=(3, n_offsets))
    M /= np.linalg.norm(M, axis=0, keepdims=True)
    T = 0.005

    # warm-up
    loop_version(M, T, B, "x", Gamma)
    bloch_rotate_batch(M, T, B, "x", Gamma)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        loop_version(M, T, B, "x", Gamma)
    t_loop = (time.perf_counter() - t0) / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        bloch_rotate_batch(M, T, B, "x", Gamma)
    t_batch = (time.perf_counter() - t0) / n_repeats

    return t_loop, t_batch


if __name__ == "__main__":
    print(f"{'n_offsets':>10} | {'loop (ms)':>12} | {'batched (ms)':>13} | {'speedup':>8}")
    print("-" * 52)
    for n in (1, 10, 50, 100, 500, 2000, 10000):
        reps = 200 if n <= 500 else 20
        t_loop, t_batch = bench(n, n_repeats=reps)
        speedup = t_loop / t_batch
        print(f"{n:>10} | {t_loop*1000:>12.4f} | {t_batch*1000:>13.4f} | {speedup:>7.2f}x")