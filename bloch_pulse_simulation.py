import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from bloch import bloch_rotate
from file_import import import_file
from mat_operator import *
from scipy.interpolate import CubicSpline


def sim_import_shaped_pulse(M, flip, angle, t_max, file_path, N_init, phi, Gamma):

    ## shaped pulse calculator
    """
    M, df, RF, t_max = shaped_pulse(M, flip, angle, t_max, file_path, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    file_path       - file path for composite pulse
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    output:
    M               - final magnetization vector
    df              - bandwith array of the pulse
    RF              - pulse shape array
    t               - time array of the pulse
    t_max           - duration of pulse (need to be stored to plot the pulse diagram)
    """
    # Import shaped pulse data
    xy_array = import_file(file_path)

    # Original number of points and time scaling
    N = len(xy_array)
    dt = t_max / N
    t = np.linspace(-N / 2 * dt, N / 2 * dt, N)

    # Target number of points
    target_N = int(t_max * 1000)
    t_resampled = np.linspace(-t_max / 2, t_max / 2, target_N)

    # Resample the xy_array to the t_max * 1000 points
    xy_resampled = np.zeros((target_N, 2), dtype=np.float64)
    xy_resampled[:, 0] = CubicSpline(t, xy_array[:, 0])(t_resampled)
    xy_resampled[:, 1] = CubicSpline(t, xy_array[:, 1])(t_resampled)
    
    # Prepare RF array
    RF_array = np.zeros(np.shape(xy_resampled), dtype=np.complex128)

    for k in range(target_N):
        xy_temp = np.zeros((2, 1), dtype=float)
        xy_temp = cpu_rot.Rot(xy_resampled[k, 1] * np.pi / 180) @ np.array([1, 0]).T
        RF_array[k, 1] = complex(xy_temp[0], xy_temp[1])

    # Check whether adiabatic pulse or not
    pul_type = ""
    if (max(xy_resampled[:,1])>=350):
        pul_type = "adiabatic"
    RF = xy_resampled[:, 0] * RF_array[:, 1]
    if (pul_type == "adiabatic"):
        RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt) * 2
    else:
        RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt)
    
    # Add pulse to the magntization
    for n in range(N_init, N_init + target_N):
        if n == 0:
            M[:, n] = M[:, n]
        else:
            M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[n-N_init]), np.imag(RF[n-N_init]), phi/Gamma], angle)

    # Update the number of points
    N_final = N_init + target_N

    return M, N_final


def sim_shaped_pulse(M, flip, angle, t_max, shape, N_init, N, phi, Gamma):

    ## shaped pulse simulator
    """
    M, df, RF, t_max = sim_shaped_pulse(M, flip, angle, t_max, shape, N, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    N               - the number of points of the pulse
    dt              - size of each step
    shape           - shape of pulse
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    output:
    M               - final magnetization vector in time
    """
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final, 1) * dt
    if shape == "sinc":
        RF = np.hamming(N).T  * np.sinc(t)
    elif shape == "cos":
        RF = np.hamming(N).T * np.cos(t)
    elif shape == "sinc2p":
        RF = np.sinc(2*np.pi*t)
    else:
        raise ValueError(f'Failed to run the proper bloch rotation with "{shape}".')
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

    for n in range(N_init, N_init + N):
        M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[n-N_init]), np.imag(RF[n-N_init]), phi/Gamma], angle)

    N_final = N + N_init

    return M, N_final

def sim_hard_pulse(M, flip, angle, t_max, N_init, N, phi, Gamma):

    ## hard pulse simulator
    """
    M, df, RF, t_max = sim_hard_pulse(M, flip, angle, t_max, N, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    N               - the number of points of the pulse
    dt              - size of each step
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    output:
    M               - final magnetization vector in time
    """
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final-1, 1) * dt
    RF = np.ones((1, int(N)))
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

    for n in range(N_init, N_init + N):
        M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[0, n-N_init]), np.imag(RF[0, n-N_init]), phi/Gamma], angle)

    N_final = N + N_init

    return M, N_final



def plot_3D_arrow_figure(Ms, num_arrows, N, color, interval):
    

    ## 3D arrow motion plot simulator
    """
    ani = plot_3D_arrow_figure(M, N)
    parameters  
    input:
    M               - magnetization vector by time
    N               - the number of points of the pulse
    output:
    ani             - 3D plotted animation
    """

    global fig, ax
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate base colors from a colormap
    cmap = cm.get_cmap(color, num_arrows)
    base_colors = [cmap(i) for i in range(num_arrows)]

    def get_arrow(Ms, frame):
        # drawing the arrow of vector M for each time point
            x = 0
            y = 0
            z = 0
            u, v, w= Ms[:, frame]
            return x, y, z, u, v, w

    global num_phi, num_theta
    
    quivers = [ax.quiver(*get_arrow(Ms[i], 0), color=base_colors[i]) for i in range(num_arrows)]


    def update(frame):
        # updating each quiver for time point
        nonlocal quivers 

        for quiver in quivers:
            quiver.remove()

        quivers = [ax.quiver(*get_arrow(Ms[i], frame), pivot='tail', color=base_colors[i]) for i in range(num_arrows)]

        ax.set_title(f'Time: {frame} microseconds')

    # Plotting radius 1 sphere surface
    radius = 1
    num_phi = 21
    num_theta = 21

    phi = np.linspace(0, 2 * np.pi, num_phi)
    theta = np.linspace(0, np.pi, num_theta)

    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    ax.plot_surface(x, y, z, color='k', alpha=0.05, edgecolors='k', linewidth=0.1)

    # axis condition
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set aspect ratio to make all axes equal
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for x, y, z

    ani = FuncAnimation(fig, update, frames=range(N), interval=interval)
    plt.show()

    return ani

def save_animation_to_gif(ani, file_name, fps):
    """
    ani         : animation returned by FuncAnimation
    file_name   : save file names with file index
    """
    ani.save(file_name, writer='pillow', fps=fps, dpi=150) # pip install pillow or conda install pillow
