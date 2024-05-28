import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt

## Hard pulse calculator
"""
parameters 
M0              - start vector size of magnetization
M_equilibrium   - location of initial magnetization vector
dt              - size of each step
flip            - flip angle (rad)
t_max           - maximum time of T
BW              - bandwith (kHz)
"""

Gamma = 42.58 # kHz/mT proton gyromagnetic ratio

M0 = 1
M_equilibrium = np.array([0, 0, M0])
dt = 0.1            # time step (ms)

flip = np.pi    # 90 degree (pi/2 rad)

# hard pulse
"""
t_max = 1.5       # duration (ms)
N = t_max / dt      # Number of steps
"""
t_max = 2
N = 100
dt = t_max / N
init = -N/2       
final = N/2 + 1
t = np.arange(init, final-1, 1) * dt

RF = np.ones((1, int(N)))
print(RF)
print(np.shape(RF))
RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)
BW = 4 # kHz
df = np.linspace(-BW/2, BW/2, num=100)
M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)
print(len(t))
for n in range(len(t)):
    for f in range(len(df)):
        M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[0, n]), np.imag(RF[0, n]), df[f]/Gamma], "x")
    
t_ext_neg = [t[0] - 1e-5, 0]
t_ext_pos = [t[-1] + 1e-5, 0]
print(t_ext_pos)

fig, axs = plt.subplots(2, 1)
axs[0].plot(t_ext_neg[0], t_ext_neg[1])
axs[0].plot(t, RF.T)
axs[0].plot(t_ext_pos[0], t_ext_pos[1])
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
axs[1].plot(df, M[2,:], label="Mz")
axs[1].plot(df, np.sqrt(M[0,:]**2+M[1,:]**2), label="|Mxy|")
axs[1].set(xlabel='frequency (kHz)', ylabel='flip')
axs[1].legend()
plt.show()

