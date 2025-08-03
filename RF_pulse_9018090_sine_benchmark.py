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
M_equilibrium = np.array([0, 0, M0])
BW = 60 # kHz
N = 1000
df = np.linspace(-BW/2, BW/2, num=N)

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

num_pulse = 1

df_temp = np.ndarray(shape=(num_pulse, 1, N))
RF_temp = np.ndarray(shape=(num_pulse, 1, N))
RF_angle_temp = np.ndarray(shape=(num_pulse, 1, N))
t_max_temp = np.ndarray(shape=(num_pulse, 1, N))
Ns = np.ndarray(shape=(num_pulse, 1))
file_path = 'wave/HypSec'
#file_path = 'wave/GaussCascadeQ5'
#file_path = 'wave/sine.jhl'

shape = "hypsec"

# shaped Pulse (sine)
i = 0
#print(f'first pulse "{file_path}" running...')
#M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] = torch_pulse.torch_shaped_pulse(M, np.pi / 2, "x", 0.6, "cos", N, BW, Gamma)
#M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi , "x", 1, file_path, BW, Gamma)
M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_shaped_pulse(M, np.pi, "x", 1, shape, N, BW, Gamma)
#M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] = torch_pulse.torch_hard_pulse(M, np.pi/2, "x", 0.02, N, BW, Gamma)
"""
# hard Pulse
i += 1
print("second pulse running...")
M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] = torch_pulse.torch_hard_pulse(M, -np.pi, "x", 0.02, N, BW, Gamma)


#file_path = 'wave/sin600_rev.jfy'
#file_path = 'wave/GaussCascadeQ5_rev'
# shaped Pulse (sine)
i += 1
#print(f'thrid pulse "{file_path}" running...')
M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] = torch_pulse.torch_shaped_pulse(M, np.pi / 2, "x", 0.6, "cos", N, BW, Gamma)
#M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi / 2, "x", 0.6, file_path, BW, Gamma)
"""

RF_t = RF_temp[0, :, :]
RF_angle = RF_angle_temp[0, :, :]

#RF_t = np.append(RF_temp[0, :, :], RF_temp[1, :, :])
#RF_t = np.append(RF_t, RF_temp[2, :, :])

#RF_angle = np.append(RF_angle_temp[0, :, :], RF_angle_temp[1, :, :])
#RF_angle = np.append(RF_angle, RF_angle_temp[2, :, :])

t_1 = np.arange(0, Ns[0], 1) * t_max_temp[0] / Ns[0]
#t_2 = np.arange(0, Ns[1], 1) * t_max_temp[1] / Ns[1] + t_max_temp[0]
#t_3 = np.arange(0, Ns[2], 1) * t_max_temp[2] / Ns[2] + t_max_temp[0] + t_max_temp[1]

#t = np.append(t_1, t_2)
#t = np.append(t, t_3)

title_font_size = 18
label_font_size = 15
ticks_font_size = 12
font_name = "Arial"

t = t_1.T

RF_t = np.abs(RF_t) * Gamma
fig, axs = plt.subplots(4, 1, figsize=(6,12))
fig.subplots_adjust(hspace=0.5)
#axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[0].plot(t, RF_t.T)
axs[0].set_ylim(bottom=-0.5)
#axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[0].set_xlabel('time (ms)', fontsize=label_font_size, fontname=font_name)
axs[0].set_ylabel('RF (kHz)', fontsize=label_font_size, fontname=font_name)
axs[0].set_title('Waveform', fontsize=title_font_size, fontname=font_name)
axs[0].tick_params(axis='both', labelsize=ticks_font_size)


#axs[1].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[1].plot(t, RF_angle.T)
#axs[1].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[1].set_ylim(top=370, bottom=-10)
axs[1].set_yticks(np.arange(0, 370, 60))
axs[1].set_xlabel('time (ms)', fontsize=label_font_size, fontname=font_name)
axs[1].set_ylabel('degree(˚)', fontsize=label_font_size, fontname=font_name)
axs[1].set_title('Phase', fontsize=title_font_size, fontname=font_name)
axs[1].tick_params(axis='both', labelsize=ticks_font_size)


df = df * 1000
axs[2].plot(df, M[2,:], label="Mz", color='red', alpha=1)
axs[2].plot(df, M[1,:], label="My", color='blue')
axs[2].plot(df, M[0,:], label="Mx", color='green')
axs[2].set_title('Excitation Profile Calculated by PULSIM', fontsize=title_font_size, fontname=font_name)
axs[2].set_xlabel('frequency (Hz)', fontsize=label_font_size, fontname=font_name)
axs[2].set_ylabel('flip', fontsize=label_font_size, fontname=font_name)
axs[2].tick_params(axis='both', labelsize=ticks_font_size)
axs[2].legend(loc="upper right")

file_path = 'Excitation_profiles/Hypesec_1000_60.xlsx'
#file_path = 'hard20_100.xlsx'
#file_path = 'sin600_hard20_sin600_jsl.xlsx'
#file_path = 'Excitation_profiles/GaQ5_3m_20u_GaQ5rev_3m.xlsx'
#file_path = 'hard20_100.xlsx'
#file_path = 'sine600.xlsx'
#file_path = 'sin600_180_sin600_jfy.xlsx'
#file_path = 'Excitation_profiles/Eburp.xlsx'

data = pd.read_excel(file_path, skiprows=5)
data[';Offset[Hz]'] = data[';Offset[Hz]'].str.replace(':', '', regex=False)

data = data.astype(float)


axs[3].plot(data[';Offset[Hz]'], data['Mz'], label="Mz", color='red', alpha=1)
axs[3].plot(data[';Offset[Hz]'], data['My'], label="My", color='blue')
axs[3].plot(data[';Offset[Hz]'], data['Mx'], label="Mx", color='green')
axs[3].set_title('Excitation Profile Calculated by Topspin', fontsize=title_font_size, fontname=font_name)
axs[3].set_xlabel('frequency (Hz)', fontsize=label_font_size, fontname=font_name)
axs[3].set_ylabel('flip', fontsize=label_font_size, fontname=font_name)
axs[3].tick_params(axis='both', labelsize=ticks_font_size)
axs[3].legend(loc="upper right")


fig.savefig('arial_benchmark_Hypsec_adiabatic.svg', dpi=600)
#axs[1].legend(loc="lower right", bbox_to_anchor=(1.1, 0))
plt.show()

