"""
color_map.py -- sweep pulse duration, plot a time-vs-frequency colormap of
the resulting magnetization component.

Migrated to the PULSIM package API during the Task #33 restructure: the old
sc_import_shaped_pulse/sc_hard_pulse functions this script called no longer
exist anywhere in pulse.py -- this script could not run at all before this
fix. Rebuilt using RFShape + Pulse + PulseSequence, the same pattern as
RF_pulse_9018090_GauscascadeQ5_forrev_oo.py.
"""
import numpy as np
import matplotlib.pyplot as plt

from PULSIM.rf_shape import RFShape
from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.parallel import parallel_map

Gamma = 42.58  # kHz/mT
BW = 8         # kHz
N = 1000
file_path = 'wave/GaussCascadeQ5'


def run_simulation(time_temp, Gamma, BW, N, file_path):
    df = np.linspace(-BW / 2, BW / 2, num=N)
    M0 = np.tile(np.array([0., 0., 1.]), (N, 1)).T.astype(float)

    backend = NumpyBackend(Gamma=Gamma)
    shaped_shape = RFShape.create("file", path=file_path, duration=time_temp / 10)
    hard_shape = RFShape.create("hard", duration=0.02, points=N)

    seq = PulseSequence([
        Pulse(shaped_shape, flip=np.pi / 2, axis="x", backend=backend),
        Pulse(hard_shape, flip=-np.pi, axis="x", backend=backend),
        Pulse(shaped_shape, flip=np.pi / 2, axis="x", backend=backend),
    ])
    return seq.run(M0, df)


init_tp = 3
final_tp = 50
direction = "X"

param_list = [(time_temp, Gamma, BW, N, file_path) for time_temp in range(init_tp, final_tp)]
results = parallel_map(run_simulation, param_list, n_jobs=-1)

M = np.array(results)

init_tp_ms = init_tp / 10
final_tp_ms = final_tp / 10

plt.figure()
if direction == "X":
    plt.imshow(M[:, 0], aspect='auto', extent=[-BW/2, BW/2, init_tp_ms, final_tp_ms], cmap='viridis', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(label='Mx')
elif direction == "Y":
    plt.imshow(M[:, 1], aspect='auto', extent=[-BW/2, BW/2, init_tp_ms, final_tp_ms], cmap='viridis', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(label='My')
elif direction == "Z":
    plt.imshow(M[:, 2], aspect='auto', extent=[-BW/2, BW/2, init_tp_ms, final_tp_ms], cmap='viridis', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(label='MZ')
else:
    raise ValueError(f'{direction} is not the proper axis')

plt.xlabel('Frequency (kHz)')
plt.ylabel('Time (ms)')
plt.title('Time-dependent ' + direction + '-direction Magnetization')

plt.show()