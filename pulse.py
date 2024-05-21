import time
import numpy as np
from rotation import Rot
from bloch import bloch_rotate
from file_import import import_file, read_xy_points
from joblib import Parallel, delayed


def import_shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma) :

    start = time.time()

    ## shaped pulse calculator
    """
    M, df, RF, t_max = shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    file_path       - file path for composite pulse
    BW              - bandwith (kHz)
    output:
    M               - final magnetization vector
    df              - bandwith array of the pulse
    RF              - pulse shape array
    t               - time array of the pulse
    t_max           - duration of pulse (need to be stored to plot the pulse diagram)
    """
    xy_array = import_file(file_path)
    RF_array = np.zeros(np.shape(xy_array), dtype=np.complex128)
    N = len(xy_array)
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final, 1) * dt
    pul_type = ""
    for k in range(len(xy_array)):
        xy_temp = np.zeros((2, 1), dtype=float)
        xy_temp = Rot(xy_array[k, 1] * np.pi / 180) @ np.array([1, 0]).T
        RF_array[k, 1] = complex(xy_temp[0], xy_temp[1])
    if (max(xy_array[:,1])>=350):
        pul_type = "adiabatic"
    RF = xy_array[:, 0] * RF_array[:, 1]
    if (pul_type == "adiabatic"):
        RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt) * 2
    else:
        RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt)

    df = np.linspace(-BW/2, BW/2, num=1000)

    M = Parallel(n_jobs=-1)(delayed(parallel_rotate_shape)(M[:, f], t, RF, Gamma, angle, dt, df[f]) for f in range(len(df)))
    M = np.array(M).T

    end = time.time()

    print('elapsed time: {} sec'.format(end-start) )

    return M, df, RF, t_max, N

def shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma) :

    start = time.time()
    
    ## shaped pulse calculator
    """
    M, df, RF, t_max = shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    N               - the number of points of the pulse
    dt              - size of each step
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    shape           - shape of the pulse (options: sinc, cos)
    BW              - bandwith (kHz)
    output:
    M               - final magnetization vector
    df              - bandwith array of the pulse
    RF              - pulse shape array
    t               - time array of the pulse
    t_max           - duration of pulse (need to be stored to plot the pulse diagram)
    """
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final, 1) * dt
    if shape == "sinc":
        RF = np.hamming(N).T  * np.sinc(t)
    elif shape == "cos":
        RF = np.cos(np.pi / t_max * t)
    elif shape == "gauss":
        RF = gaussian(t, 0, 1/7)
    elif shape == "sinc2p":
        RF = np.sinc(np.pi*t)
    else:
        raise ValueError(f'Failed to run the proper bloch rotation with "{shape}".')
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

    df = np.linspace(-BW/2, BW/2, num=1000)

    M = Parallel(n_jobs=-1)(delayed(parallel_rotate_shape)(M[:, f], t, RF, Gamma, angle, dt, df[f]) for f in range(len(df)))
    M = np.array(M).T


    end = time.time()

    print('elapsed time: {} sec'.format(end-start) )

    return M, df, RF, t_max, N

def hard_pulse(M, flip, angle, t_max, N, BW, Gamma):

    start = time.time()
    
    ## hard pulse calculator
    """
    M, df, RF, t_max = hard_pulse(M, flip, angle, t_max, N, BW, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    N               - the number of points of the pulse
    dt              - size of each step
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    BW              - bandwith (kHz)
    output:
    M               - final magnetization vector
    df              - bandwith array of the pulse
    RF              - pulse shape array
    t               - time array of the pulse
    t_max           - duration of pulse (need to be stored to plot the pulse diagram)
    """

    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final-1, 1) * dt
    RF = np.ones((1, int(N)))
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)
    df = np.linspace(-BW/2, BW/2, num=1000)

    M = Parallel(n_jobs=-1)(delayed(parallel_rotate_hard)(M[:, f], t, RF, Gamma, angle, dt, df[f]) for f in range(len(df)))
    M = np.array(M).T

    end = time.time()

    print('elapsed time: {} sec'.format(end-start) )

    return M, df, RF, t_max, N


def sc_import_shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma) :

    start = time.time()

    ## shaped pulse calculator (single core)
    """
    M, df, RF, t_max = shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    file_path       - file path for composite pulse
    BW              - bandwith (kHz)
    output:
    M               - final magnetization vector
    df              - bandwith array of the pulse
    RF              - pulse shape array
    t               - time array of the pulse
    t_max           - duration of pulse (need to be stored to plot the pulse diagram)
    """
    xy_array = import_file(file_path)
    RF_array = np.zeros(np.shape(xy_array), dtype=np.complex128)
    N = len(xy_array)
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final, 1) * dt
    pul_type = ""
    for k in range(len(xy_array)):
        xy_temp = np.zeros((2, 1), dtype=float)
        xy_temp = Rot(xy_array[k, 1] * np.pi / 180) @ np.array([1, 0]).T
        RF_array[k, 1] = complex(xy_temp[0], xy_temp[1])
    if (max(xy_array[:,1])>=350):
        pul_type = "adiabatic"
    RF = xy_array[:, 0] * RF_array[:, 1]
    if (pul_type == "adiabatic"):
        RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt) * 2
    else:
        RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt)

    df = np.linspace(-BW/2, BW/2, num=1000)

    for n in range(len(t)):
        for f in range(len(df)):
            M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma], angle)

    end = time.time()

    print('elapsed time: {} sec'.format(end-start) )

    return M, df, RF, t_max, N

def sc_shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma) :

    start = time.time()
    
    ## shaped pulse calculator (single core)
    """
    M, df, RF, t_max = shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    N               - the number of points of the pulse
    dt              - size of each step
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    shape           - shape of the pulse (options: sinc, cos)
    BW              - bandwith (kHz)
    output:
    M               - final magnetization vector
    df              - bandwith array of the pulse
    RF              - pulse shape array
    t               - time array of the pulse
    t_max           - duration of pulse (need to be stored to plot the pulse diagram)
    """
    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final, 1) * dt
    if shape == "sinc":
        RF = np.hamming(N).T  * np.sinc(t)
    elif shape == "cos":
        RF = np.cos(np.pi / t_max * t)
    elif shape == "gauss":
        RF = gaussian(t, 0, 1/7)
    elif shape == "sinc2p":
        RF = np.sinc(np.pi*t)
    else:
        raise ValueError(f'Failed to run the proper bloch rotation with "{shape}".')
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

    df = np.linspace(-BW/2, BW/2, num=1000)


    for n in range(len(t)):
        for f in range(len(df)):
            M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma], angle)

    end = time.time()

    print('elapsed time: {} sec'.format(end-start) )

    return M, df, RF, t_max, N

def sc_hard_pulse(M, flip, angle, t_max, N, BW, Gamma):

    start = time.time()
    
    ## hard pulse calculator (single core)
    """
    M, df, RF, t_max = hard_pulse(M, flip, angle, t_max, N, BW, Gamma)
    parameters 
    input:
    M               - magnetization vector 
    N               - the number of points of the pulse
    dt              - size of each step
    angle           - flip angle position (x, y, z)
    flip            - flip angle (rad)
    t_max           - duration of pulse
    BW              - bandwith (kHz)
    output:
    M               - final magnetization vector
    df              - bandwith array of the pulse
    RF              - pulse shape array
    t               - time array of the pulse
    t_max           - duration of pulse (need to be stored to plot the pulse diagram)
    """

    dt = t_max / N
    init = -N/2
    final = N/2
    t = np.arange(init, final-1, 1) * dt
    RF = np.ones((1, int(N)))
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)
    df = np.linspace(-BW/2, BW/2, num=1000)


    for n in range(len(t)):
        for f in range(len(df)):
            M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[0, n]), np.imag(RF[0, n]), df[f]/Gamma], angle)

    end = time.time()

    print('elapsed time: {} sec'.format(end-start) )

    return M, df, RF, t_max, N



def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def parallel_rotate_hard(M_f, t, RF, Gamma, angle, dt, df_f):
    for n in range(len(t)):
            M_f  = bloch_rotate(M_f, dt, [np.real(RF[0, n]), np.imag(RF[0, n]), df_f/Gamma], angle)

    return M_f

def parallel_rotate_shape(M_f, t, RF, Gamma, angle, dt, df_f):
    for n in range(len(t)):
            M_f  = bloch_rotate(M_f, dt, [np.real(RF[n]), np.imag(RF[n]), df_f/Gamma], angle)

    return M_f
    