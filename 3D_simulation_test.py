import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from PULSIM.bloch import bloch_rotate
from PULSIM.simulate import sim_hard_pulse, sim_shaped_pulse, sim_import_shaped_pulse
from bloch_pulse_simulation import plot_3D_arrow_figure, save_animation_to_gif, plot_3D_arrow_figure_old, plot_3D_arrow_with_pulse
from IPython.display import Video, HTML, display
from matplotlib.animation import FFMpegWriter
from PULSIM.parallel import parallel_map

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

num_arrows = 10
Ms = np.ndarray((num_arrows, 3, N))
RFs = np.ndarray((1, N))
RF_angles = np.ndarray((1, N))

def simulate_one_arrow(j, N, t_max_1, t_max_2, t_max_3, angle, Gamma, N_0):
    M = np.tile(M_equilibrium, (N, 1)).T.astype(float)
    file_path = 'wave/sine.jhl'
    M, temp_1, angle_temp_1, N_1 = sim_import_shaped_pulse(M, np.pi/2, angle, t_max_1, file_path, N_0, j , Gamma)
    M, temp_2, angle_temp_2, N_2 = sim_hard_pulse(M, -np.pi, angle, t_max_2, N_1, int(t_max_2 * 1000), j, Gamma)
    M, temp_3, angle_temp_3, N_3 = sim_import_shaped_pulse(M, np.pi/2, angle, t_max_3, file_path, N_2, j , Gamma)
    RF = np.append(np.append(temp_1, temp_2), temp_3)
    RF_angle = np.append(np.append(angle_temp_1, angle_temp_2), angle_temp_3)
    return M, RF, RF_angle

flip = np.pi
angle = "y"

param_list = [
    ((i - num_arrows / 2) / num_arrows * np.pi * 4, N, t_max_1, t_max_2, t_max_3, angle, Gamma, N_0)
    for i in range(num_arrows)
]

results = parallel_map(simulate_one_arrow, param_list, n_jobs=-1)

Ms = np.array([r[0] for r in results])
RFs = results[0][1]
RF_angles = results[0][2]
print(Ms)

N_time = np.linspace(0, N, N)

color = 'viridis' # Default: viridis (color blue to yellow)

#ani = plot_3D_arrow_figure_old(Ms, num_arrows, N, color, interval=1)

ani = plot_3D_arrow_with_pulse(Ms, N_time, RFs, RF_angles, num_arrows, N, color, interval=1)

#save_animation_to_gif(ani, 'animation_tab.gif', 1000) # save the animation to gif file 

# Save the animation directly as an MP4 file
#ani.save("animation.mp4", writer='ffmpeg', fps=80)

#writer = FFMpegWriter(fps=80, metadata=dict(artist='PULSIM'), bitrate=1800)

#ani.save("animation.mp4", writer=writer)

# Embed and display the saved video
#display(Video("animation.mp4"))