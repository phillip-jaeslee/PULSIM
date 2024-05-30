import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt

Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])

flip = np.pi / 2

# shaped Pulse (sinc)
t_max = 1.5
N_1 = 100
dt = t_max / N_1
init = -N_1/2
final = N_1/2
t_1 = np.arange(init, final, 1) * dt
RF_1 = np.hamming(N_1).T * np.sinc(t_1)
RF_1 = (flip) * RF_1/np.sum(RF_1) / (2*np.pi*Gamma*dt)

BW = 8 # kHz
df = np.linspace(-BW/2, BW/2, num=100)

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

for n in range(len(t_1)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF_1[n]), np.imag(RF_1[n]), df[f]/Gamma])

flip = np.pi

# hard pulse
t_max = 0.0192
N_2 = 100
dt = t_max / N_2
init = -N_2/2
final = N_2/2
t_2 = np.arange(init, final-1, 1) * dt
RF_2 = np.ones((1, int(N_2)))
RF_2 = (flip) * RF_2/np.sum(RF_2) / (2*np.pi*Gamma*dt)
print(np.shape(RF_2))
BW = 8 # kHz
df = np.linspace(-BW/2, BW/2, num=100)

for n in range(len(t_2)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF_2[0, n]), np.imag(RF_2[0, n]), df[f]/Gamma])



flip = np.pi / 2

# shaped Pulse (sinc)
t_max = 1.5
N_3 = 100
dt = t_max / N_3
init = -N_3/2
final = N_3/2
t_3 = np.arange(init, final, 1) * dt
RF_3 = np.hamming(N_3).T * np.sinc(t_3)
RF_3 = (flip) * RF_3/np.sum(RF_3) / (2*np.pi*Gamma*dt)

BW = 8 # kHz
df = np.linspace(-BW/2, BW/2, num=100)
for n in range(len(t_3)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF_3[n]), np.imag(RF_3[n]), df[f]/Gamma])

RF = np.append(RF_1, RF_2)
RF = np.append(RF, RF_3)

init = 0
final = N_1
t_1 = np.arange(init, final, 1) * dt
init = N_1
final = N_1 + N_2 
t_2 = np.arange(init, final, 1) * dt
init = N_1 + N_2
final = N_1 + N_2 + N_3
t_3 = np.arange(init, final, 1) * dt

t = np.append(t_1, t_2)
t = np.append(t, t_3)

fig, axs = plt.subplots(2, 1)
axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[0].plot(t, RF.T)
axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
df = df * 1000 / 600
axs[1].plot(df, M[2,:], label="Mz")
axs[1].plot(df, M[1,:], label="My")
axs[1].plot(df, M[0,:], label="Mx")
axs[1].set(xlabel='frequency (ppm)', ylabel='flip')
axs[1].legend()
plt.show()