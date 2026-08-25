"""
benchmarks/bench_parallel_arrows.py

Verifies parallel_map(simulate_one_arrow, ...) matches a plain serial loop
over the same arrows, then benchmarks wall-clock time for both, for a couple
of num_arrows sizes. simulate_one_arrow is copied from 3D_simulation_test.py
(not imported from it -- that script runs a full simulation + opens a plot
window as a side effect of import, which we don't want here).

    python benchmarks/bench_parallel_arrows.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from PULSIM.simulate import sim_hard_pulse, sim_import_shaped_pulse
from PULSIM.parallel import parallel_map

Gamma = 42.58
M0 = 1
M_equilibrium = np.array([0, 0, M0])

t_max_1 = 0.6
t_max_2 = 0.012
t_max_3 = 0.6
N = int((t_max_1 + t_max_2 + t_max_3) * 1000)
N_0 = 0
angle = "y"


def simulate_one_arrow(j, N, t_max_1, t_max_2, t_max_3, angle, Gamma, N_0):
    M = np.tile(M_equilibrium, (N, 1)).T.astype(float)
    file_path = 'wave/sine.jhl'
    M, temp_1, angle_temp_1, N_1 = sim_import_shaped_pulse(M, np.pi/2, angle, t_max_1, file_path, N_0, j, Gamma)
    M, temp_2, angle_temp_2, N_2 = sim_hard_pulse(M, -np.pi, angle, t_max_2, N_1, int(t_max_2 * 1000), j, Gamma)
    M, temp_3, angle_temp_3, N_3 = sim_import_shaped_pulse(M, np.pi/2, angle, t_max_3, file_path, N_2, j, Gamma)
    RF = np.append(np.append(temp_1, temp_2), temp_3)
    RF_angle = np.append(np.append(angle_temp_1, angle_temp_2), angle_temp_3)
    return M, RF, RF_angle


def param_list_for(num_arrows):
    return [
        ((i - num_arrows / 2) / num_arrows * np.pi * 4, N, t_max_1, t_max_2, t_max_3, angle, Gamma, N_0)
        for i in range(num_arrows)
    ]


def run_serial(param_list):
    return [simulate_one_arrow(*p) for p in param_list]


def run_parallel(param_list, n_jobs=-1):
    return parallel_map(simulate_one_arrow, param_list, n_jobs=n_jobs)


def verify(num_arrows=10):
    params = param_list_for(num_arrows)
    serial = run_serial(params)
    parallel = run_parallel(params)
    max_diff = max(np.max(np.abs(s[0] - p[0])) for s, p in zip(serial, parallel))
    print(f"verify (num_arrows={num_arrows}): max |M diff| serial vs parallel = {max_diff:.3e}")
    assert max_diff < 1e-12, "parallel_map result doesn't match serial loop"
    print("PASS")


def bench(num_arrows):
    params = param_list_for(num_arrows)

    t0 = time.perf_counter()
    run_serial(params)
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    run_parallel(params)
    t_parallel = time.perf_counter() - t0

    speedup = t_serial / t_parallel
    print(f"num_arrows={num_arrows:>4} | serial: {t_serial:6.3f}s | parallel: {t_parallel:6.3f}s | speedup: {speedup:5.2f}x")


if __name__ == "__main__":
    verify(num_arrows=10)
    print()
    for n in (10, 40, 100):
        bench(n)