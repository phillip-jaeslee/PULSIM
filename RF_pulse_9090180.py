import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt
from pulse import shaped_pulse, hard_pulse


global Gamma, BW
Gamma = 42.58 # kHz/mT

M0 = 1
M_equilibrium = np.array([0, 0, M0])
BW = 6 # kHz
df = np.linspace(-BW/2, BW/2, num=1000)
N = 100

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

df_temp = np.ndarray(shape=(3, 1, 1000))
RF_temp = np.ndarray(shape=(3, 1, N))
t_max_temp = np.ndarray(shape=(3, 1, N))


# shaped Pulse (sinc)
i = 0
print("first pulse running...")
M, df_temp[i], RF_temp[i], t_max_temp[i] = shaped_pulse(M, np.pi / 2, "x", 0.6, "cos" , N, BW, Gamma)

# shaped Pulse (sinc)
i += 1
print("second pulse running...")
M, df_temp[i], RF_temp[i], t_max_temp[i] = shaped_pulse(M, np.pi / 2, "x", 0.6, "cos", N, BW, Gamma)

# hard pulse
i += 1
print("third pulse running...")
M, df_temp[i], RF_temp[i], t_max_temp[i] = hard_pulse(M, -np.pi, "x", 0.0192, N, BW, Gamma)

RF_t = np.append(RF_temp[0, :, :], RF_temp[1, :, :])
RF_t = np.append(RF_t, RF_temp[2, :, :])

t_1 = np.arange(0, N, 1) * t_max_temp[0] / N
t_2 = np.arange(0, N, 1) * t_max_temp[1] / N + t_max_temp[0]
t_3 = np.arange(0, N, 1) * t_max_temp[2] / N + t_max_temp[0] + t_max_temp[1]

t = np.append(t_1, t_2)
t = np.append(t, t_3)

fig, axs = plt.subplots(2, 1)
axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[0].plot(t, RF_t.T)
axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
df = df * 1000
axs[1].plot(df, M[2,:], label="Mz")
axs[1].plot(df, M[1,:], label="My")
axs[1].plot(df, M[0,:], label="Mx")
axs[1].set(xlabel='frequency (Hz)', ylabel='flip')
axs[1].legend()
plt.show()