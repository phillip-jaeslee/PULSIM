import numpy as np
import numpy.matlib 
import pandas as pd
from bloch import torch_bloch_rotate
import matplotlib.pyplot as plt
from pulse import *


global Gamma, BW
Gamma = 42.577478461 # kHz/mT

MULTI = True

M0 = 1
M_equilibrium = np.array([M0, 0, 0])
BW = 30 # kHz
N = 1000
df = np.linspace(-BW/2, BW/2, num=N)

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

num_pulse = 4

M_record = np.ndarray(shape=(num_pulse, 3, N))
df_temp = np.ndarray(shape=(num_pulse, 1, N))
RF_temp = np.ndarray(shape=(num_pulse, 1, N))
RF_angle_temp = np.ndarray(shape=(num_pulse, 1, N))
t_max_temp = np.ndarray(shape=(num_pulse, 1, N))
Ns = np.ndarray(shape=(num_pulse, 1))
#file_path = 'wave/GaussCascadeQ5'
#file_path = 'wave/sine.jhl'


# shaped Pulse (sine)
i = 0
file_path = 'wave/Q5_sebop.1'
#print(f'first pulse "{file_path}" running...')
#M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] = torch_pulse.torch_shaped_pulse(M, np.pi / 2, "x", 0.6, "cos", N, BW, Gamma)
M_record[i], df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi/2 , "x", 0.35, file_path, BW, Gamma)
#M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] = torch_pulse.torch_hard_pulse(M, np.pi/2, "x", 0.02, N, BW, Gamma)

i = 1
file_path = 'wave/Q5_sebop.1'
M_record[i], df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi/2 , "x", 0.455, file_path, BW, Gamma)

i = 2
file_path = 'wave/Q5.1000'
M_record[i], df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi/2, "x", 0.35, file_path, BW, Gamma)

i = 3
file_path = 'wave/Q5.1000'
M_record[i], df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi/2, "x", 0.400, file_path, BW, Gamma)



title_font_size = 18
label_font_size = 15
ticks_font_size = 12
font_name = "Arial"

fig, axs = plt.subplots(1, 1, figsize=(6, 4.5))  # Single subplot, axs is a single Axes object

df = df * 1000
axs.plot(df, M_record[0, 0, :], label="Q5_sebop-350us", color='red', alpha=1)
axs.plot(df, M_record[1, 0, :], label="Q5_sebop-455us", color='magenta')
axs.plot(df, M_record[2, 0, :], label="Q5_1000-350us", color='blue')
axs.plot(df, M_record[3, 0, :], label="Q5_1000-400us", color='green')


# Set titles and labels directly on axs
axs.set_title('Excitation Profile Calculated by PULSIM', fontsize=title_font_size, fontname=font_name)
axs.set_xlabel('frequency (Hz)', fontsize=label_font_size, fontname=font_name)
axs.set_ylabel('flip', fontsize=label_font_size, fontname=font_name)
axs.legend()



fig.savefig('arial_benchmark_600MHz_Q5_x.svg', dpi=600)
#axs[1].legend(loc="lower right", bbox_to_anchor=(1.1, 0))
plt.show()

