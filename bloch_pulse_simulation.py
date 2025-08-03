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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import gridspec
from pulse_shape_list import *



def sim_import_shaped_pulse(M, flip, angle, t_max, file_path, N_init, phi, Gamma):

    ## composite pulse simulator
    """
    M, N_final = sim_import_shaped_pulse(M, flip, angle, t_max, file_path, N_init, phi, Gamma)
    parameters 
    input:
    M               - magnetization vector
    flip            - flip angle (rad)
    angle           - flip angle position (x, y, z)
    t_max           - duration of pulse
    file_path       - file path for composite pulse
    N_init          - initial number of points
    phi             - deviation from the on-resonance
    Gamma           - Gyromagnetic ratio of the nucleus
    output:
    M               - final magnetization vector in time
    RF              - pulse amplitude in time
    RF_angle        - pulse phase in time
    N_final         - final number of points
    """
    # Import shaped pulse data (Bruker pulse reader)
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
    
    RF_angle = np.array(xy_resampled[:, 1])

    # Add pulse to the magntization
    for n in range(N_init, N_init + target_N):
        if n == 0:
            M[:, n] = M[:, n]
        else:
            M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[n-N_init]), np.imag(RF[n-N_init]), phi/Gamma], angle)

    # Update the number of points
    N_final = N_init + target_N
    RF = abs(RF)
    return M, RF, RF_angle, N_final


def sim_shaped_pulse(M, flip, angle, t_max, shape, N_init, N, phi, Gamma):

    ## shaped pulse simulator
    """
    M, N_final = sim_shaped_pulse(M, flip, angle, t_max, shape, N_init, N, phi, Gamma)
    parameters 
    input:
    M               - magnetization vector
    flip            - flip angle (rad)
    angle           - flip angle position (x, y, z)
    t_max           - duration of pulse
    shape           - shape of pulse ("sinc", "cos", "sinc2p") #TODO: expand pulse type (Hermite, Gaussian, G4, Q3, Eburp, Seduce, etc.)
    N_init          - initial number of points
    N               - number of points for the shaped pulse
    phi             - deviation from the on-resonance
    Gamma           - Gyromagnetic ratio of the nucleus
    output:
    M               - final magnetization vector in time
    RF              - pulse amplitude in time
    RF_angle        - pulse phase in time
    N_final         - final number of points
    """
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final, 1) * dt

    shape_funcs = {
        "sinc":         lambda: np.hamming(N).T * np.sinc(t),
        "cos":          lambda: np.hamming(N).T * np.cos(t),
        "sinc2p":       lambda: np.sinc(2 * np.pi * t),
        "eburp1":       lambda: E_BURP_1_pulse(duration=t_max, points=N),
        "eburp2":       lambda: E_BURP_2_pulse(duration=t_max, points=N),
        "iburp1":       lambda: I_BURP_1_pulse(duration=t_max, points=N),
        "iburp2":       lambda: I_BURP_2_pulse(duration=t_max, points=N),
        "uburp":        lambda: U_BURP_pulse(duration=t_max, points=N),
        "reburp":       lambda: RE_BURP_pulse(duration=t_max, points=N),
        "gausscasG3":   lambda: GAUSSCASCADE_G3_pulse(duration=t_max, points=N),
        "gausscasG4":   lambda: GAUSSCASCADE_G4_pulse(duration=t_max, points=N),
        "gausscasQ3":   lambda: GAUSSCASCADE_Q3_pulse(duration=t_max, points=N),
        "gausscasQ5":   lambda: GAUSSCASCADE_Q5_pulse(duration=t_max, points=N),
        "hermite":      lambda: HERMITE_pulse(duration=t_max, points=N),
        "seduce1":      lambda: SEDUCE_1_pulse(duration=t_max, points=N),
        "sneeze":       lambda: SNEEZE_pulse(duration=t_max, points=N),
        "qsneeze":      lambda: QSNEEZE_pulse(duration=t_max, points=N),
        "esnob":        lambda: eSNOB_pulse(duration=t_max, points=N),
        "i2snob":       lambda: i2SNOB_pulse(duration=t_max, points=N),
        "i3snob":       lambda: i3SNOB_pulse(duration=t_max, points=N),
        "rsnob":        lambda: rSNOB_pulse(duration=t_max, points=N),
        "dsnob":        lambda: dSNOB_pulse(duration=t_max, points=N),
        "hypsec":       lambda: HYPSEC_pulse(duration=t_max, points=N),
        "swrl11":       lambda: SWIRL11_pulse(duration=t_max, points=N),
        "swrl12":       lambda: SWIRL12_pulse(duration=t_max, points=N),
        "swrl17":       lambda: SWIRL17_pulse(duration=t_max, points=N)
    }

    try:
        RF_org = shape_funcs[shape]()
    except KeyError:
        raise ValueError(f"Unknown shape '{shape}'. Available shapes: {list(shape_funcs)}")
    
    RF_angle = np.ones((1, int(N)))
    # If pulse is real-only
    if np.isrealobj(RF_org):
        RF_angle = np.where(RF_org >= 0, 0.0, 180.0)
        RF = (flip) * RF_org/np.sum(RF_org) / (2*np.pi*Gamma*dt)

    # If pulse has complex components
    else:
        phase_rad = np.angle(RF_org)            # returns −π to π
        RF_angle = (-np.degrees(phase_rad)) % 360      # convert to degrees
        RF = (flip) * RF_org/np.sum(RF_org) / (2*np.pi*Gamma*dt) *2

    for n in range(N_init, N_init + N):
        M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[n-N_init]), np.imag(RF[n-N_init]), phi/Gamma], angle)

    N_final = N + N_init
    RF = abs(RF)
    return M, RF, RF_angle, N_final

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
    RF              - pulse amplitude in time
    RF_angle        - pulse phase in time
    N_final         - final number of points
    """
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final-1, 1) * dt
    RF = np.ones((1, int(N)))
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)
    RF_angle = np.ones((1, int(N)))

    if flip > 0:
        RF_angle = RF_angle * 0
    elif flip < 0:
        RF_angle = RF_angle * 180
    for n in range(N_init, N_init + N):
        M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[0, n-N_init]), np.imag(RF[0, n-N_init]), phi/Gamma], angle)

    N_final = N + N_init
    RF = abs(RF)
    return M, RF, RF_angle, N_final


def plot_3D_arrow_figure_old(Ms, num_arrows, N, color, interval):
    

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


def draw_cone_arrow(ax, origin, direction, color='r', length=1.0, cone_length=0.2, cone_radius=0.05, resolution=20):

    ## Draw cone arrow
    """
    draw_cone_arrow(ax, origin, direction, color='r', length=1.0, cone_length=0.2, cone_radius=0.05, resolution=20)
    parameters  
    input:
    ax              - axes (matplotlib.Axes)
    origin          - origin of arrow
    direction       - direction of arrow
    optional:
    color           - color of cone arrow (default='r')
    length          - length of arrow (default=1.0)
    cone_length     - length of cone (default=0.2)
    cone_radius     - largest radius of cone (default=0.05)
    resolution      - resolution of cone (default=20)
    output:
    """

    direction = direction / np.linalg.norm(direction)
    shaft_end = origin + direction * (length - cone_length)

    ax.plot([origin[0], shaft_end[0]], [origin[1], shaft_end[1]], [origin[2], shaft_end[2]], color=color)

    # Cone tip
    z = direction
    not_z = np.array([1, 0, 0]) if not np.allclose(z, [1, 0, 0]) else np.array([0, 1, 0])
    x = np.cross(not_z, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)

    theta = np.linspace(0, 2 * np.pi, resolution)
    circle = np.array([
        shaft_end + cone_radius * (np.cos(t) * x + np.sin(t) * y)
        for t in theta
    ])
    tip = shaft_end + direction * cone_length
    verts = [[tip, circle[i], circle[(i + 1) % resolution]] for i in range(resolution)]
    cone = Poly3DCollection(verts, color=color)
    ax.add_collection3d(cone)

def plot_3D_arrow_figure(Ms, num_arrows, N, color, interval):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate color map
    cmap = cm.get_cmap(color, num_arrows)
    base_colors = [cmap(i) for i in range(num_arrows)]

    # Define unit sphere mesh
    def draw_sphere(ax, radius=1, num_phi=21, num_theta=21):
        phi, theta = np.meshgrid(
            np.linspace(0, 2 * np.pi, num_phi),
            np.linspace(0, np.pi, num_theta)
        )
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        ax.plot_surface(x, y, z, color='k', alpha=0.05, edgecolors='k', linewidth=0.1)

    def get_arrow(Ms, frame):
        return 0, 0, 0, *Ms[:, frame]

    def update(frame):
        ax.cla()
        draw_sphere(ax)

        for i in range(num_arrows):
            x, y, z, u, v, w = get_arrow(Ms[i], frame)
            draw_cone_arrow(ax, origin=np.array([x, y, z]), direction=np.array([u, v, w]), color=base_colors[i])

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f'Time: {frame} microseconds')

    draw_sphere(ax)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])

    ani = FuncAnimation(fig, update, frames=range(N), interval=interval)
    plt.show()
    return ani

def save_animation_to_gif(ani, file_name, fps):
    """
    ani         : animation returned by FuncAnimation
    file_name   : save file names with file index
    fps         : frame per second
    """
    ani.save(file_name, writer='pillow', fps=fps, dpi=150) # pip install pillow or conda install pillow



def plot_3D_arrow_with_pulse(Ms, pulse_time, pulse_amplitude, pulse_phase, num_arrows, N, color="viridis", interval=1):

    ## plot 3D bloch sphere animation with pulse
    """
    plot_3D_arrow_with_pulse(Ms, pulse_time, pulse_amplitude, phase, num_arrows, N, color, interval=1)
    parameters  
    input:
    Ms              - magnetization in time series
    pulse_time      - numpy array of time that pulse irradiated
    pulse_amplitude - amplitude of pulse
    pulse_phase     - phase of pulse
    num_arrows      - the number of arrows
    N               - the number of time points
    optional:
    color           - color series of arrow (default = "viridis")
    interval        - interval between frame (default = 1)
    output:
    ani             - animation of 3D bloch sphere with pulse
    """


    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.3)

    # 3D Arrow Plot
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    # Pulse amplitude Timeline Plot
    ax_pulse = fig.add_subplot(gs[0, 1])
    # Pulse phase timeline Plot
    ax_phase = fig.add_subplot(gs[1, 1])
    
    cmap = plt.cm.get_cmap(color, num_arrows)
    base_colors = [cmap(i) for i in range(num_arrows)]

    def draw_sphere(ax, radius=1, num_phi=21, num_theta=21):
        phi, theta = np.meshgrid(
            np.linspace(0, 2 * np.pi, num_phi),
            np.linspace(0, np.pi, num_theta)
        )
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        ax.plot_surface(x, y, z, color='k', alpha=0.05, edgecolors='k', linewidth=0.1)


    def get_arrow(Ms, frame):
        x, y, z = 0, 0, 0
        u, v, w = Ms[:, frame]
        return x, y, z, u, v, w

    arrow_objs = []

    # Initial 3D Arrows
    for i in range(num_arrows):
        x, y, z, u, v, w = get_arrow(Ms[i], 0)
        print(u,v, w)
        direction = np.array([u, v, w])
        arrow = draw_cone_arrow(ax3d, origin=np.array([x, y, z]), direction=direction, color=base_colors[i])
        arrow_objs.append(arrow)

    # Pulse amplitude Plot (static gray waveform)
    ax_pulse.plot(pulse_time, pulse_amplitude, color='gray', linewidth=1)
    red_line, = ax_pulse.plot([], [], color='red', linewidth=2)

    # Pulse phase Plot (static gray phase)
    ax_phase.plot(pulse_time, pulse_phase, color='gray', linewidth=1)
    blue_line, = ax_phase.plot([], [], color='blue', linewidth=2)

    def update(frame):
        ax3d.cla()
        ax3d.set_xlim(-1.5, 1.5)
        ax3d.set_ylim(-1.5, 1.5)
        ax3d.set_zlim(-1.5, 1.5)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_title(f"Time: {frame} µs")
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')

        draw_sphere(ax3d)

        for i in range(num_arrows):
            x0, y0, z0, u, v, w = get_arrow(Ms[i], frame)
            direction = np.array([u, v, w])
            draw_cone_arrow(ax3d, origin=np.array([x0, y0, z0]), direction=direction, color=base_colors[i])

        # Pulse timeline update
        red_line.set_data(pulse_time[:frame], pulse_amplitude[:frame])
        blue_line.set_data(pulse_time[:frame], pulse_phase[:frame])
        ax_pulse.set_xlim(pulse_time[0], pulse_time[-1])
        ax_phase.set_xlim(pulse_time[0], pulse_time[-1])
        if min(pulse_amplitude) == 0:
            ax_pulse.set_ylim(min(pulse_amplitude) - 0.1, max(pulse_amplitude) * 1.4)
        elif max(pulse_amplitude) == 0:
            ax_pulse.set_ylim(min(pulse_amplitude) * 1.4, max(pulse_amplitude) + 0.1)
        else:
            ax_pulse.set_ylim(min(pulse_amplitude) * 1.4, max(pulse_amplitude) * 1.4)
        ax_phase.set_yticks(np.arange(0, 370, 60))
        ax_pulse.set_ylabel("Amplitutde")
        ax_pulse.set_xlabel("Time (µs)")
        ax_phase.set_ylabel("Phase")
        ax_phase.set_xlabel("Time (µs)")

    ani = FuncAnimation(fig, update, frames=range(N), interval=interval)
    #plt.tight_layout()
    plt.show()
    return ani


