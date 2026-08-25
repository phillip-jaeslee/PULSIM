"""
automation_input.py -- interactively build and run a multi-pulse sequence.

Migrated to the PULSIM package API during the Task #33 restructure: the old
shaped_pulse/hard_pulse/import_shaped_pulse functions this script called no
longer exist as bare module-level names in pulse.py (only namespaced under
cpu_pulse/torch_pulse) -- this script could not run at all before this fix.
Rebuild using RFShape + Pulse + PulseSeqeunce, replacing the by-hand
RF_temp/t_max_temp/N bookkeepking with PulseSequence.rf/.time directly.
"""


import numpy as np
import matplotlib.pyplot as plt

from PULSIM.rf_shape import RFShape
from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from input_parameter import gyro_ratio, get_spin_parameters, get_pulse_parameters, number_to_words

Gamma, num_pulse, BW, M_equilibrium = get_spin_parameters()

df = np.linspace(-BW/2, BW/2, num=1000)
N_t = 1000

M0 = np.tile(M_equilibrium, (len(df), 1)).T.astype(float)

backend = NumpyBackend(Gamma=Gamma)

pulses = []

for i in range(num_pulse):
    pulse_type = int(input('Choose pulse type \n [1] composite [2] hard [3] shaped : '))
    if pulse_type == 1:
        file_path, flip_val, angle_val, t_max_val = get_pulse_parameters(pulse_type)
        print(f'{number_to_words(i+1)} pulse "{file_path}" running...')
        shape = RFShape.create("file", path=file_path, duration=t_max_val)
    elif pulse_type == 2:
        flip_val, angle_val, t_max_val, N_val = get_pulse_parameters(pulse_type)    
        print(f'{number_to_words(i+1)} pulse "hard" running...')
        shape = RFShape.create("hard", duration=t_max_val, points=N_val)
    elif pulse_type == 3:
        flip_val, angle_val, shape_val, t_max_val = get_pulse_parameters(pulse_type)
        print(f'{number_to_words(i+1)} pulse "{shape_val}" running...')
        shape = RFShape.create(shape_val, duration=t_max_val, points=N_t)
    else:
        raise ValueError(f'Error of pulse type')
    
    pulses.append(Pulse(shape, flip=flip_val, axis=angle_val, backend=backend))

seq = PulseSequence(pulses)
M = seq.run(M0, df)

for n in range(len(df)):
    if M[2, n] > 0.9:
        print(df[n])
        break

fig, axs = plt.subplots(2, 1)
axs[0].plot(seq.time, np.abs(seq.rf))
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
df_hz = df * 1000
axs[1].plot(df_hz, M[2, :], label="Mz")
axs[1].plot(df_hz, M[1, :], label="My")
axs[1].plot(df_hz, M[0, :], label="Mx")
axs[1].set(xlabel='frequency (Hz)', ylabel='flip')
axs[1].legend()
plt.show()