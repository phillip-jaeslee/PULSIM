"""
parallel.py -- run independent simulations across multiple CPU cores.

For sweeps where each call is fully self-contained (a whole Pulse or
PulseSequence run with a different offset/phase/parameter, not a single RF
time-step) and takes long enough that joblib's per-task overhead (spawning
worker processes, pickling arguments and results between them) is
negligible next to the work itself.

This is deliberately NOT wired into Backend/Pulse: Backend.rotate() is
already vectorized over the offset axis (see bloch_rotate_batch), and
parallelizing that per-RF-time-step call would pay joblib's dispatch cost
hundreds of times per pulse, for microseconds of actual work each time --
almost certainly a net loss. parallel_map is for the coarser, embarrassingly
parallel case one level up: many independent whole simulations, e.g. a
parameter sweep. (For several offsets of the same pulse, pass them as
columns of one M instead -- see tutorial_3d_bloch_animation.py.)

joblib is an optional dependency:  pip install ".[parallel] from a PULSIM clone"
"""

def require_joblib():
    """Entry gate for the joblib-only code path:
    returns the module or raises a message naming the extra to install
    """
    try:
        import joblib
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "parallel_map requires joblib, an optional dependency.\n"
            '    pip install ".[parallel]"   (from a PULSIM clone)\n'
            "Not `pip install pulsim`: that name on PyPI is an unrelated package.\n"
            "Everything else in PULSIM runs without it."           
        ) from None
    return joblib

def parallel_map(fn, param_list, n_jobs=-1, **joblib_kwargs):
    """
    Run fn(*params) once per entry in param_list, across n_jobs processes.

    fn            : a picklable top-level callable (not a lambda or bound
                    method -- joblib's default backend pickles fn to ship it
                    to worker processes).
    param_list    : list of argument tuples, one per independent call.
    n_jobs        : passed straight to joblib.Parallel (-1 = use all cores).
    joblib_kwargs : passed straight to joblib.Parallel.
    returns       : list of fn(*params) results, in the same order as param_list.
    """
    joblib = require_joblib()
    return joblib.Parallel(n_jobs=n_jobs, **joblib_kwargs)(joblib.delayed(fn)(*params) for params in param_list)
