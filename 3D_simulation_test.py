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

N = 1000

M = np.tile(M_equilibrium, (N, 1)).T
M = M.astype(float)

flip = np.pi /2
t_max = 10
angle = "x"

M = sim_shaped_pulse(M, flip, angle, t_max, "cos", N, Gamma)

plot_3D_arrow_figure(M, N)

