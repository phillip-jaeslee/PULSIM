import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from bloch import bloch_rotate

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


global Gamma, BW
Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])
BW = 6 # kHz
df = np.linspace(-BW/2, BW/2, num=1000)
N = 1000

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

flip = np.pi /3
t_max = 10
angle = "x"

dt = t_max / N
init = -N/2
final = N/2
t = np.arange(init, final-1, 1) * dt
RF = np.ones((1, int(N)))
RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)
print(RF)

for n in range(N):
    M[:, n]  = bloch_rotate(M[:, n-1], dt, [np.real(RF[0, n]), np.imag(RF[0, n]), 0], angle)

print(np.shape(M))

for i in range(1000):
    u = M[0, i]
    v = M[1, i]
    w = M[2, i]


def get_arrow(i):
    x = 0
    y = 0
    z = 0
    u = M[0, i]
    v = M[1, i]
    w = M[2, i]
    return x, y, z, u, v, w

quiver = ax.quiver(*get_arrow(0))

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)


def update(frame):
    global quiver
    quiver.remove()
    quiver = ax.quiver(*get_arrow(frame))

ani = FuncAnimation(fig, update, frames=range(N), interval=1)
plt.show()