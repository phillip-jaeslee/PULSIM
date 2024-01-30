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

N = 3000
N_0 = 0

M = np.tile(M_equilibrium, (N, 1)).T
M = M.astype(float)

flip = np.pi /2
t_max = 0.6
angle = "x"


M, N_1 = sim_shaped_pulse(M, flip, angle, t_max, "cos", N_0, 1000, Gamma)

flip = np.pi
t_max = 0.0192
M, N_2 = sim_hard_pulse(M, flip, angle, t_max, N_1, 1000, Gamma)


flip = np.pi /2
t_max = 0.6
angle = "x"

M, N_3 = sim_shaped_pulse(M, flip, angle, t_max, "cos", N_2, 1000, Gamma)


plot_3D_arrow_figure(M, N)

