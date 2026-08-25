import numpy as np
import numpy.matlib 
from PULSIM.bloch import torch_bloch_rotate
import matplotlib.pyplot as plt

from PULSIM.rf_shape import RFShape
from PULSIM.backend import TorchBackend
from PULSIM.pulse_oo import Pulse

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

i = 0
print(f"first pulse {file_path} running...")
shape = RFShape.create("file", path=file_path, duration=0.6)
pulse = Pulse(shape, np.pi / 2, axis="x", backend=TorchBackend(Gamma=Gamma))
df_temp[i] = np.linspace(-BW / 2, BW / 2, num=shape.points)
M = pulse.apply(M, df_temp[i])
RF_temp[i] = np.abs(pulse.calibrated_rf())
RF_angle_temp[i] = shape.xy[:, 1]
t_max_temp[i] = 0.6
Ns[i] = shape.points

i += 1
print("second pulse running...")
shape = RFShape.create("hard", duration=0.02, points=N)
pulse = Pulse(shape, np.pi, axis="x", backend=TorchBackend(Gamma=Gamma))
df_temp[i] = np.linspace(-BW / 2, BW / 2, num=N)
M = pulse.apply(M, df_temp[i])
RF_temp[i] = np.abs(pulse.calibrated_rf().reshape(1, -1))
RF_angle_temp[i] = 0.0 if np.pi > 0 else (180.0 if np.pi < 0 else 0.0)
t_max_temp[i] = 0.02
Ns[i] = N


file_path = 'wave/sine.jhl'
# shaped Pulse (sine)
i += 1
print(f"third pulse {file_path} running...")
shape = RFShape.create("file", path=file_path, duration=0.6)
pulse = Pulse(shape, np.pi / 2, axis="x", backend=TorchBackend(Gamma=Gamma))
df_temp[i] = np.linspace(-BW / 2, BW / 2, num=shape.points)
M = pulse.apply(M, df_temp[i])
RF_temp[i] = np.abs(pulse.calibrated_rf())
RF_angle_temp[i] = shape.xy[:, 1]
t_max_temp[i] = 0.6
Ns[i] = shape.points


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