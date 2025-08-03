import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from bloch import bloch_rotate
from bloch_pulse_simulation import sim_hard_pulse, sim_shaped_pulse, plot_3D_arrow_figure, sim_import_shaped_pulse, save_animation_to_gif, plot_3D_arrow_figure_old, plot_3D_arrow_with_pulse
from IPython.display import Video, HTML, display
from matplotlib.animation import FFMpegWriter

plt.rcParams['animation.writer'] = 'ffmpeg'


global Gamma, BW
Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])

t_max_1 = 1
t_max_2 = 0.012
t_max_3 = 1
N = int((t_max_1) * 1000)


N_0 = 0

num_arrows = 10
Ms = np.ndarray((num_arrows, 3, N))
RFs = np.ndarray((1, N))
RF_angles = np.ndarray((1, N))


M = np.tile(M_equilibrium, (N, 1)).T
M = M.astype(float)

flip = np.pi
angle = "x"


j = 0


for i in range(num_arrows):
    j = (i - num_arrows/2) / num_arrows * np.pi /2
    #print(N_0)
    shape = "iburp2"
    #file_path = 'wave/HypSec'
    Ms[i], temp_1, angle_temp_1, N_1 = sim_shaped_pulse(M, np.pi, angle, t_max_1, shape, N_0, int(t_max_1 * 1000), j , Gamma)
    #print(N_1)
    #Ms[i], temp_2, angle_temp_2, N_2 = sim_hard_pulse(Ms[i], -np.pi, angle, t_max_2, N_1, int(t_max_2 * 1000), j, Gamma)
    #print(N_2)
    #file_path = 'wave/GaussCascadeQ5_rev'    
    #Ms[i], temp_3, angle_temp_3, N_3 = sim_import_shaped_pulse(Ms[i], np.pi/2, angle, t_max_3, file_path, N_2, j , Gamma)

RFs = temp_1.T

#RFs = np.append(temp_1, temp_2)
#RFs = np.append(RFs, temp_3)

RF_angles = angle_temp_1.T
#RF_angles = np.append(angle_temp_1, angle_temp_2)
#RF_angles = np.append(RF_angles, angle_temp_3)

N_time = np.linspace(0, N, N)

color = 'viridis' # Default: viridis (color blue to yellow)

#ani = plot_3D_arrow_figure(Ms, num_arrows, N, color, interval=1)

ani = plot_3D_arrow_with_pulse(Ms, N_time, RFs, RF_angles, num_arrows, N, color, interval=1)

#save_animation_to_gif(ani, 'animation_tab.gif', 1000) # save the animation to gif file 

# Save the animation directly as an MP4 file
#ani.save("animation.mp4", writer='ffmpeg', fps=80)

#writer = FFMpegWriter(fps=80, metadata=dict(artist='PULSIM'), bitrate=1800)

#ani.save("animation.mp4", writer=writer)

# Embed and display the saved video
#display(Video("animation.mp4"))
