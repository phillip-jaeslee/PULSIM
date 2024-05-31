import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PULSIM.bloch import bloch_rotate
from PULSIM.bloch_pulse_simulation import sim_hard_pulse, sim_shaped_pulse, plot_3D_arrow_figure, sim_import_shaped_pulse, save_animation_to_gif


global Gamma, BW
Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])

t_max_1 = 1
t_max_2 = 0.020
t_max_3 = 1
N = int((t_max_1 + t_max_2 + t_max_3) * 1000)


N_0 = 0

num_arrows = 30
Ms = np.ndarray((num_arrows, 3, N))


M = np.tile(M_equilibrium, (2020, 1)).T
M = M.astype(float)

flip = np.pi
angle = "x"

j = 0

for i in range(num_arrows):
    j = (i - num_arrows/2) / num_arrows * np.pi
    file_path = '../wave/GaussCascadeQ5'
    Ms[i], N_1 = sim_import_shaped_pulse(M, np.pi/2, angle, t_max_1, file_path, N_0, j , Gamma)
    Ms[i], N_2 = sim_hard_pulse(Ms[i], np.pi, angle, t_max_2, N_1, int(t_max_2 * 1000), j, Gamma)
    #file_path = 'wave/GaussCascadeQ5_rev'    
    Ms[i], N_3 = sim_import_shaped_pulse(Ms[i], np.pi/2, angle, t_max_3, file_path, N_2, j , Gamma)

color = 'viridis' # Default: viridis (color blue to yellow)

ani = plot_3D_arrow_figure(Ms, num_arrows, N, color)


#save_animation_to_gif(ani, 'animation_5.gif') # save the animation to gif file 