import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt

Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])
dt = 0.1

flip = np.pi / 2

# shaped Pulse (sinc)
t_max = 8
N = t_max / dt
init = -N/2
final = N/2 - 1
t = np.arange(init, final+1, 1) * dt
RF = np.hamming(N).T * np.sinc(t)
RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

BW = 2 # kHz
df = np.linspace(-BW, BW, num=100)

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)
print(np.shape(RF))
for n in range(len(t)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma])


flip = np.pi

# hard pulse
N = t_max / dt
init = -N/2
final = N/2 - 1
t = np.arange(init, final+1, 1) * dt
RF = np.ones((1, int(N)))
RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)
BW = 2 # kHz
df = np.linspace(-BW, BW, num=100)

for n in range(len(t)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[0, n]), np.imag(RF[0, n]), df[f]/Gamma])



flip = np.pi / 2

# shaped Pulse (sinc)
N = t_max / dt
init = -N/2
final = N/2 - 1
t = np.arange(init, final+1, 1) * dt
RF = np.hamming(N).T * np.sinc(t)
RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

BW = 2 # kHz
df = np.linspace(-BW, BW, num=100)
print(np.shape(RF))
for n in range(len(t)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma])

fig, axs = plt.subplots(2, 1)
axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[0].plot(t, RF.T)
axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')

axs[1].plot(df, M[2,:], label="Mz")
axs[1].plot(df, np.sqrt(M[0,:]**2+M[1,:]**2), label="|Mxy|")
axs[1].set(xlabel='frequency (kHz)', ylabel='flip')
axs[1].legend()
plt.show()