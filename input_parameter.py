import numpy as np

def gyro_ratio(nucleus):
    global Gamma

    if nucleus == 'H':
        Gamma = 42.577478518
    elif nucleus == 'D':
        Gamma = 6.536
    elif nucleus == 'T':
        Gamma = 45.415
    elif nucleus == '13C':
        Gamma = 10.7084
    elif nucleus == '15N':
        Gamma = -4.316
    else:
        raise ValueError(f'{nucleus} may not support for the system.')
    
    return Gamma

def get_spin_parameters():

    nucleus = input('Enter the nucleus : ')

    Gamma = gyro_ratio(nucleus)

    num_pulse = int(input('Enter the number of pulse : '))

    BW = int(input('Enter the bandwith (kHz) : '))

    init_vect = input('Enter the inital magnetization vector coordinates (x, y, z) : ')

    M0 = 1

    if (init_vect == 'x'):
        M_equilibrium = np.array([M0, 0, 0])
    elif (init_vect == 'y'):
        M_equilibrium = np.array([0, M0, 0])
    elif (init_vect == 'z'):
        M_equilibrium = np.array([0, 0, M0])
    else:
        raise ValueError(f'There is no such "{init_vect}" coordinates. Please choose again.')

    return Gamma, num_pulse, BW, M_equilibrium

def get_pulse_parameters():
    pulse_type = int(input('Choose pulse type \n [1] composite [2] hard [3] shaped'))
    if pulse_type == 1:
        file_path = input('file path : ')
        flip = int(input('flip angle (degree) : ')) * np.pi / 180
        angle = input('flip angle direction : ')
        t_max = float(input('time duration : '))
        return file_path, flip, angle, t_max

    elif pulse_type == 2:
        flip = int(input('flip angle (degree) : ')) * np.pi / 180
        angle = input('flip angle direction : ')
        t_max = float(input('time duration : '))
        N = int(input('the number of point of the hard pulse : '))
        return flip, angle, t_max

    elif pulse_type == 3:
        flip = int(input('flip angle (degree) : ')) * np.pi / 180
        angle = input('flip angle direction : ')
        shape = input('shape (sinc, cos, gauss) : ')
        t_max = float(input('time duration :v'))
        return flip, angle, shape, t_max