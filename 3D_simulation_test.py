import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from bloch import bloch_rotate
from bloch_pulse_simulation import sim_hard_pulse, sim_shaped_pulse, plot_3D_arrow_figure, sim_import_shaped_pulse, save_animation_to_gif
from IPython.display import Video, HTML, display

plt.rcParams['animation.writer'] = 'ffmpeg'


global Gamma, BW
Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])

t_max_1 = 0.6
t_max_2 = 0.012
t_max_3 = 0.6
N = int((t_max_1 + t_max_2 + t_max_3) * 1000)


N_0 = 0

num_arrows = 50
Ms = np.ndarray((num_arrows, 3, N))


M = np.tile(M_equilibrium, (N, 1)).T
M = M.astype(float)

flip = np.pi
angle = "x"


j = 0

for i in range(num_arrows):
    j = (i - num_arrows/2) / num_arrows * np.pi * 2
    #print(N_0)
    file_path = 'wave/sine.jhl'
    Ms[i], N_1 = sim_import_shaped_pulse(M, np.pi/2, angle, t_max_1, file_path, N_0, j , Gamma)
    #print(N_1)
    Ms[i], N_2 = sim_hard_pulse(Ms[i], -np.pi, angle, t_max_2, N_1, int(t_max_2 * 1000), j, Gamma)
    #print(N_2)
    #file_path = 'wave/GaussCascadeQ5_rev'    
    Ms[i], N_3 = sim_import_shaped_pulse(Ms[i], np.pi/2, angle, t_max_3, file_path, N_2, j , Gamma)

color = 'viridis' # Default: viridis (color blue to yellow)

ani = plot_3D_arrow_figure(Ms, num_arrows, N, color, interval=10)

#save_animation_to_gif(ani, 'animation_tab.gif', 1000) # save the animation to gif file 

# Save the animation directly as an MP4 file
#ani.save("animation.mp4", writer='ffmpeg', fps=80)

# Embed and display the saved video
#display(Video("animation.mp4"))
