import numpy as np
import numpy.matlib 
from bloch import torch_bloch_rotate
import matplotlib.pyplot as plt
from pulse import *
from visualization import plot_pulse, save_figure


global Gamma, BW
Gamma = 42.58 # kHz/mT

MULTI = True

M0 = 1
M_equilibrium = np.array([0, 0, M0])
BW = 6 # kHz
N = 1000
df = np.linspace(-BW/2, BW/2, num=N)

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

df_temp = np.ndarray(shape=(3, 1, N))
RF_temp = np.ndarray(shape=(3, 1, N))
RF_angle_temp = np.ndarray(shape=(3, 1, N))
t_max_temp = np.ndarray(shape=(3, 1, N))
Ns = np.ndarray(shape=(3, 1))
file_path = 'wave/sine.jhl'


# shaped Pulse (sine)
i = 0
print(f'first pulse "{file_path}" running...')
M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi / 2, "x", 0.6, file_path, BW, Gamma)

# hard Pulse
i += 1
print("second pulse running...")
M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] = torch_pulse.torch_hard_pulse(M, np.pi, "x", 0.02, N, BW, Gamma)


file_path = 'wave/sine.jhl'
# shaped Pulse (sine)
i += 1
print(f'thrid pulse "{file_path}" running...')
M, df_temp[i], RF_temp[i], RF_angle_temp[i], t_max_temp[i], Ns[i] =torch_pulse.torch_import_shaped_pulse(M, np.pi / 2, "x", 0.6, file_path, BW, Gamma)


RF_t = np.append(RF_temp[0, :, :], RF_temp[1, :, :])
RF_t = np.append(RF_t, RF_temp[2, :, :])

t_1 = np.arange(0, Ns[0].item(), 1) * t_max_temp[0, :] / Ns[0, :] #NOTE: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
t_2 = np.arange(0, Ns[1].item(), 1) * t_max_temp[1] / Ns[1, :] + t_max_temp[0]
t_3 = np.arange(0, Ns[2].item(), 1) * t_max_temp[2] / Ns[2, :] + t_max_temp[0] + t_max_temp[1]

t = np.append(t_1, t_2)
t = np.append(t, t_3)

RF_angle = np.append(RF_angle_temp[0, :, :], RF_angle_temp[1, :, :])
RF_angle = np.append(RF_angle, RF_angle_temp[2, :, :])

fig = plot_pulse(M, RF_t, RF_angle, df, t, label_Mx=False, label_My=False, label_Mxy=True)

figure_file_path = "test"

save = True

save_figure(fig, save=save, file_path=figure_file_path)