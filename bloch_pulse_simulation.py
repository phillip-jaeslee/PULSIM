import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from bloch import bloch_rotate
from file_import import import_file
from rotation import Rot

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xy_array = import_file(file_path)
    N = len(xy_array)
    dt = t_max / N
    init = -N/2
    final = N/2
    angles = torch.tensor(xy_array[:, 1], dtype=torch.float32, device=device) * torch.pi / 180
    magnitudes = torch.tensor(xy_array[:, 0], dtype=torch.float32, device=device)
    
    RF_real = torch.cos(angles)
    RF_imag = torch.sin(angles)
    RF_array = torch.complex(magnitudes * RF_real, magnitudes * RF_imag)
    pul_type = ""
    pul_type = "adiabatic" if (max(xy_array[:,1])>=350) else ""
    RF = RF_array
    RF = (flip) * RF/ torch.sum(RF) / (2*torch.pi*Gamma*dt)
    if pul_type == "adiabatic":
        RF *= 2
    g_expanded = torch.ones(N, dtype=torch.float32, device=device) * phi / Gamma 

    M = torch.tensor(M, dtype=torch.float32, device=device)
    B = torch.stack([RF.real, RF.imag, g_expanded], dim=1)

    for n in range(N):
            M[:, N_init + n] = bloch_rotate(M[:, N_init + n - 1].unsqueeze(0), dt, B[n].unsqueeze(0), angle).squeeze(0)

    N_final = N_init + N

    return M.cpu().numpy(), N_final


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    RF = torch.ones(N, dtype=torch.float32, device=device)
    RF = (flip) * RF/torch.sum(RF) / (2*torch.pi*Gamma*dt)
    g_expanded = torch.ones(N, dtype=torch.float32, device=device) * phi / Gamma
    zero_expanded = torch.zeros_like(RF, dtype=torch.float32, device=device)

    B = torch.stack([RF, zero_expanded, g_expanded], dim=1)
    M = torch.tensor(M, dtype=torch.float32, device=device)

    for n in range(N):
            M[:, N_init + n] = bloch_rotate(M[:, N_init + n - 1].unsqueeze(0), dt, B[n].unsqueeze(0), angle).squeeze(0)

    N_final = N + N_init

    return M, N_final



def plot_3D_arrow_figure(Ms, num_arrows, N):
    

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
    cmap = cm.get_cmap('viridis', num_arrows)
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

        ax.set_title(f'Time: {frame} milliseconds')

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

    ani = FuncAnimation(fig, update, frames=range(N), interval=1)
    plt.show()

    return ani

def save_animation_to_gif(ani, file_name):
    """
    ani         : animation returned by FuncAnimation
    file_name   : save file names with file index
    """
    ani.save(file_name, writer='pillow', fps=10000, dpi=300) # pip install pillow or conda install pillow
