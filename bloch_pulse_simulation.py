import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from bloch import bloch_rotate

def sim_shaped_pulse(M, flip, angle, t_max, shape, N_init, N, Gamma):

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
    else:
        print("There is no such type of the envelop function")
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

    for n in range(N_init, N_init + N):
        M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[n-N_init]), np.imag(RF[n-N_init]), 0], angle)

    N_final = N + N_init

    return M, N_final

def sim_hard_pulse(M, flip, angle, t_max, N_init, N, Gamma):

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
        M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[0, n-N_init]), np.imag(RF[0, n-N_init]), 0], angle)

    N_final = N + N_init

    return M, N_final



def plot_3D_arrow_figure(M, N):

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def get_arrow(i):
        # drawing the arrow of vector M for each time point
            x = 0
            y = 0
            z = 0
            u = M[0, i]
            v = M[1, i]
            w = M[2, i]
            return x, y, z, u, v, w

    global quiver, num_phi, num_theta

    quiver = ax.quiver(*get_arrow(0))

    def update(frame):
        # updating the each quiver for time point
        global quiver
        quiver.remove()
        quiver = ax.quiver(*get_arrow(frame), pivot='tail', color='r')
        ax.set_title(f'Time: {frame * 1} frame')

    # Plotting radius 1 sphere surface
    radius = 1
    num_phi=20
    num_theta=20

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