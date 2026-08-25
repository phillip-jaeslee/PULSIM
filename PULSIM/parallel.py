"""
parallel.py -- run independent simulations across multiple CPU cores.

For sweeps where each call is fully self-contained (a whole Pulse or 
PulseSequence run with a different offset/phase/parameter, not a single RF
time-step) and takes long enough that joblib's per-task overhaed (spawning
worker processes, pickling arguments and results between them) is
negligible next to the work itself.

This is deliberately NOT wired into Backend/Pulse: Backend.rotate() is
already vectorized over the offset axis (see bloch_rotate_batch), and
parallelizing that per-RF-time-step call would pay joblib's dispatch cost
hundreds of times per pulse, for microseconds of actual work each time --
almost certainly a net loss. parallel_map is for the coarser, embarrassingly
parallel case on level up: many independent whole simulations, e.g. the
num_arrows loop in 3D_simulation_test.py, or a parameter sweep.
"""

from joblib import Parallel, delayed

def parallel_map(fn, paramlist, n_jobs=-1, **joblib_kwargs):
    """
    Run fn(*params) once per entry in param_list, across n_jobs processes.

    fn          : a pickable top-level callable (not a lambda or bound
                  method -- joblib's default backend pickles fn to ship it
                  to worker processes).
    param_list  : list of argument tuples, one per independent call.
    n_jobs      : passed straight to joblib.Parallel (-1 = use all cores).
    returns     : list of fn(*params) results, in the same order as param_list.
    """
    return Parallel(n_jobs=n_jobs, **joblib_kwargs)(delayed(fn)(*params) for params in paramlist)