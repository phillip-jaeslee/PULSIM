import numpy as np
import numpy.matlib 
from bloch import bloch_rotate
import matplotlib.pyplot as plt
from pulse import shaped_pulse, hard_pulse, import_shaped_pulse
from joblib import Parallel, delayed

def run_simulation(time_temp):
    global Gamma, BW
    Gamma = 42.58  # kHz/mT

    M0 = 1
    M_equilibrium = np.array([0, 0, M0])
    BW = 8  # kHz
    df = np.linspace(-BW/2, BW/2, num=1000)
    N = 1000

    M = np.tile(M_equilibrium, (len(df), 1)).T
    M = M.astype(float)

    df_temp = np.ndarray(shape=(3, 1, 1000))
    RF_temp = np.ndarray(shape=(3, 1, N))
    t_max_temp = np.ndarray(shape=(3, 1, N))
    file_path = 'wave/GaussCascadeQ5'

    # shaped Pulse (sine)
    i = 0
    print(f'first pulse "{file_path}" running...')
    M, df_temp[i], RF_temp[i], t_max_temp[i], N = sc_import_shaped_pulse(M, np.pi / 2, "x", time_temp / 10, file_path, BW, Gamma)
    
    # hard Pulse
    i += 1
    print("second pulse running...")
    M, df_temp[i], RF_temp[i], t_max_temp[i], N = sc_hard_pulse(M, -np.pi, "x", 0.02, N, BW, Gamma)

    # shaped Pulse (sine)
    i += 1
    print(f'third pulse "{file_path}" running...')
    M, df_temp[i], RF_temp[i], t_max_temp[i], N = sc_import_shaped_pulse(M, np.pi / 2, "x", time_temp / 10, file_path, BW, Gamma)
    
    # Return the z-component of the magnetization (Mz)
    return M

init_tp = 3
final_tp = 50
direction = "X"
BW = 8

# Run parallel jobs and collect Mz results
results = Parallel(n_jobs=-1)(delayed(run_simulation)(time_temp) for time_temp in range(init_tp, final_tp))

# Convert results to a 2D array for colormap
M = np.array(results)

init_tp = init_tp / 10
final_tp = final_tp / 10


# Create a colormap plot using pcolormesh
plt.figure()
if direction == "X":
    plt.imshow(M[:, 0], aspect='auto', extent=[-BW/2, BW/2, init_tp, final_tp], cmap='viridis', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(label='Mx')
elif direction == "Y":
    plt.imshow(M[:, 1], aspect='auto', extent=[-BW/2, BW/2, init_tp, final_tp], cmap='viridis', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(label='My')
elif direction == "Z":
    plt.imshow(M[:, 2], aspect='auto', extent=[-BW/2, BW/2, init_tp, final_tp], cmap='viridis', origin='lower', vmin=-1, vmax=1)
    plt.colorbar(label='MZ')
else:
    raise ValueError(f'{direction} is not the proper axis')
# Add labels and title
plt.xlabel('Frequency (kHz)')
plt.ylabel('Time (ms)')
plt.title('Time-dependent '+direction+'-direction Magnetization')

# Display the plot
plt.show()
