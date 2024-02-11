import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt
from pulse import shaped_pulse, hard_pulse, import_shaped_pulse
from input_parameter import gyro_ratio, get_spin_parameters, get_pulse_parameters, number_to_words


global Gamma

Gamma, num_pulse, BW, M_equilibrium = get_spin_parameters()

df = np.linspace(-BW/2, BW/2, num=1000)
N_t = 1000

M = np.tile(M_equilibrium, (len(df), 1)).T
M = M.astype(float)

df_temp = np.ndarray(shape=(num_pulse, 1, 1000))
RF_temp = np.ndarray(shape=(num_pulse, 1, N_t))
t_max_temp = np.ndarray(shape=(num_pulse, 1))


pulse_type = []
file_paths = []
flip = []
angles = []
t_max = []
N = []
shape = []


for i in range(num_pulse):
    pulse_type.append(int(input('Choose pulse type \n [1] composite [2] hard [3] shaped : ')))
    file_path, angle_val, shape_val, N_val = '', '', '', ''
    flip_val, t_max_val = None, None
    if pulse_type[i] == 1:
        file_path, flip_val, angle_val, t_max_val = get_pulse_parameters(pulse_type[i])
    elif pulse_type[i] == 2:
        flip_val, angle_val, t_max_val, N_val = get_pulse_parameters(pulse_type[i])    
    elif pulse_type[i] == 3:
        flip_val, angle_val, shape_val, t_max_val = get_pulse_parameters(pulse_type[i])
    else:
        raise ValueError(f'Error of pulse type')
    file_paths.append(file_path)
    flip.append(flip_val)
    angles.append(angle_val)
    t_max.append(t_max_val)
    shape.append(shape_val)
    N.append(N_val)

for i in range(num_pulse):
    if pulse_type[i] == 1:
        print(f'{number_to_words(i+1)} pulse "{file_paths[i]}" running...')
        M, df_temp[i], RF_temp[i], t_max_temp[i], N[i] =import_shaped_pulse(M, flip[i], angles[i], t_max[i], file_paths[i], BW, Gamma)
    elif pulse_type[i] == 2:
        print(f'{number_to_words(i+1)} pulse "hard" running...')
        M, df_temp[i], RF_temp[i], t_max_temp[i], N[i] = hard_pulse(M, flip[i], angles[i], t_max[i], N[i], BW, Gamma)
    elif pulse_type[i] == 3:
        print(f'{number_to_words(i+1)} pulse "{shape[i]}" running...')
        M, df_temp[i], RF_temp[i], t_max_temp[i], N[i] = shaped_pulse(M, flip[i], angles[i], t_max[i], shape[i], N[i], BW, Gamma)
    else:
        raise ValueError(f'Error of pulse type')

RF_t = np.empty(shape=(1, np.sum(N)), dtype='float')

for n in range(num_pulse):
    RF_t = np.append(RF_t, RF_temp[n, :, :])

t_temp = np.empty(shape=(num_pulse, N_t), dtype='float')
t = np.empty(shape=(np.sum(N)), dtype='float')

for j in range(num_pulse):
    t_temp[j] = np.arange(0, N[j], 1) * t_max_temp[j] / N[j]
    k = 0
    for k in range(j):
        t_temp[j] = t_temp[j] + t_max_temp[k]
    print(t_temp[j])

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