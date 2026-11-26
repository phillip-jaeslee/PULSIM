"""
simulate.py -- classical Bloch-equation pulse simulation, sim_* functions.

Split out of bloch_pulse_simulation.py during the nmr_core package
restructure: this half is genuine library code (build a shape, calibrate
it, step magnetization through it via Backend.rotate); the plotting/
animation half stayed behind in bloch_pulse_simulation.py at the top
level, since it's driver-side matplotlib code, not physics.
"""
import numpy as np

from .rf_shape import RFShape
from .backend import NumpyBackend
from .pulse_oo import Pulse

def sim_own_shaped_pulse(M, flip, angle, t_max, file_path, N_init, phi, Gamma):
    shape = RFShape.create("composite", path=file_path, duration=t_max, resample_to=int(t_max * 1000))
    pulse = Pulse(shape, flip=flip, axis=angle, backend=NumpyBackend(Gamma=Gamma))
    RF = pulse.calibrated_rf()
    dt = shape.dt
    N = shape.points
    RF_angle = np.where(shape.xy[:, 0] < 0, shape.xy[:, 1] + 180, shape.xy[:, 1])

    N_final = N_init + N
    for n in range(N_init, N_final):
        if n == 0:
            M[:, n] = M[:, n]
        else:
            B = np.array([[np.real(RF[n - N_init]), np.imag(RF[n - N_init]), phi / Gamma]])
            M[:, n] = pulse.backend.rotate(M[:, n-1:n], dt, B, angle)[:, 0]

    return M, RF, RF_angle, N_final


def sim_import_shaped_pulse(M, flip, angle, t_max, file_path, N_init, phi, Gamma):
    """See pulse_oo.Pulse + rf_shpae.FileShape for actual physics"""
    shape = RFShape.create("file", path=file_path, duration=t_max, resample_to=int(t_max * 1000))
    pulse = Pulse(shape, flip=flip, axis=angle, backend=NumpyBackend(Gamma=Gamma))
    RF = pulse.calibrated_rf()
    dt = shape.dt
    N = shape.points
    RF_angle = shape.xy[:, 1]

    for n in range(N_init, N_init + N):
        if n == 0:
            M[:, n] = M[:, n]
        else:
            B = np.array([[np.real(RF[n - N_init]), np.imag(RF[n - N_init]), phi / Gamma]])
            M[:, n] = pulse.backend.rotate(M[:, n-1:n], dt, B, angle)[:, 0]

    N_final = N_init + N
    RF = np.abs(RF)
    return M, RF, RF_angle, N_final

def sim_shaped_pulse(M, flip, angle, t_max, shape, N_init, N, phi, Gamma):
    """See pulse_oo.Pulse + rf_shape.py's shape registry for actual physics"""
    rf_shape = RFShape.create(shape, duration=t_max, points=N)
    pulse = Pulse(rf_shape, flip, axis=angle, backend=NumpyBackend(Gamma=Gamma))
    RF = pulse.calibrated_rf()
    dt = rf_shape.dt
    RF_angle = rf_shape.phase_profile

    for n in range(N_init, N_init + N):
        B = np.array([[np.real(RF[n - N_init]), np.imag(RF[n - N_init]), phi / Gamma]])
        M[:, n] = pulse.backend.rotate(M[:, n-1:n], dt, B, angle)[:, 0]

    N_final = N_init + N
    RF = np.abs(RF)
    return M, RF, RF_angle, N_final


def sim_hard_pulse(M, flip, angle, t_max, N_init, N, phi, Gamma):
    shape = RFShape.create("hard", duration=t_max, points=N)
    pulse = Pulse(shape, flip, axis=angle, backend=NumpyBackend(Gamma=Gamma))
    RF = pulse.calibrated_rf()
    dt = shape.dt

    RF_angle = np.ones((1, int(N))) * (0.0 if flip > 0 else 180.0 if flip < 0 else 0.0)

    for n in range(N_init, N_init + N):
        B = np.array([[np.real(RF[n - N_init]), np.imag(RF[n - N_init]), phi / Gamma]])
        M[:, n] = pulse.backend.rotate(M[:, n-1:n], dt, B, angle)[:, 0]

    N_final = N_init + N
    RF = np.abs(RF).reshape(1, -1)
    return M, RF, RF_angle, N_final

