import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt
from pulse import shaped_pulse, hard_pulse, import_shaped_pulse
from input_parameter import gyro_ratio, get_spin_parameters, get_pulse_parameters, number_to_words


global Gamma

Gamma, num_pulse, BW, M_equilibrium = get_spin_parameters()

df = np.linspace(-BW/2, BW/2, num=1000)
N = 1000

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

df_temp = np.ndarray(shape=(num_pulse, 1, 1000))
RF_temp = np.ndarray(shape=(num_pulse, 1, N))
t_max_temp = np.ndarray(shape=(num_pulse, 1, N))

RF_t = np.ndarray(shape=(num_pulse, 1, N))
t_temp = np.ndarray(shape=(num_pulse, 1, N))
t = np.ndarray(shape=(num_pulse, 1, N))

for i in range(num_pulse):
    pulse_type = get_pulse_parameters()[0]
    if pulse_type == 1:
        pulse_type, file_path, flip, angle, t_max = get_pulse_parameters()
        print(f'{number_to_words(i+1)} pulse "{file_path}" running...')
        M, df_temp[i], RF_temp[i], t_max_temp[i] =import_shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma)

    elif pulse_type == 2:
        pulse_type, flip, angle, t_max, N = get_pulse_parameters()
        print(f'{number_to_words(i+1)} pulse "hard" running...')
        M, df_temp[i], RF_temp[i], t_max_temp[i] = hard_pulse(M, flip, angle, t_max, N, BW, Gamma)

    elif pulse_type == 3:
        pulse_type, flip, angle, shape, t_max = get_pulse_parameters()
        print(f'{number_to_words(i+1)} pulse "{shape}" running...')
        M, df_temp[i], RF_temp[i], t_max_temp[i] = shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma)

    else:
        raise ValueError(f'Error of pulse type')

for n in range(num_pulse):
    RF_t = np.append(RF_t, RF_temp[i, :, :])

for j in range(num_pulse):
    t_temp[j] = np.arange(0, N, 1) * t_max_temp[j] / N 
    for k in range(j):
        t_temp[j] = t_temp[j] + t_max_temp[k]

for r in range(num_pulse):
    t = np.append(t, t_temp[r])

for n in range(len(df)):
    if M[2, n] > 0.9:
        print (df[n])
        break

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