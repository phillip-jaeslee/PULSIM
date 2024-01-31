import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from bloch import bloch_rotate
from bloch_pulse_simulation import sim_hard_pulse, sim_shaped_pulse, plot_3D_arrow_figure


global Gamma, BW, N
Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])

t_max_1 = 0.6
t_max_2 = 0.020
t_max_3 = 0.6
N = int((t_max_1 + t_max_2 + t_max_3) * 1000)

N_0 = 0

M = np.tile(M_equilibrium, (N, 1)).T
M = M.astype(float)

flip = np.pi /2
angle = "x"

M, N_1 = sim_shaped_pulse(M, flip, angle, t_max_1, "cos", N_0, int(t_max_1 * 1000), Gamma)


flip = np.pi
M, N_2 = sim_hard_pulse(M, flip, angle, t_max_2, N_1, int(t_max_2 * 1000), Gamma)


flip = np.pi /2
angle = "x"

M, N_3 = sim_shaped_pulse(M, flip, angle, t_max_3, "cos", N_2, int(t_max_3 * 1000), Gamma)
plot_3D_arrow_figure(M, N)

