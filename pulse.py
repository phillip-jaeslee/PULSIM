import numpy as np
from bloch import bloch_rotate


def shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma) :

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
        RF = np.hamming(N).T * np.cos(t)
    else:
        print("There is no such type of the envelop function")
    RF = (flip) * RF/np.sum(RF) / (2*np.pi*Gamma*dt)

    df = np.linspace(-BW/2, BW/2, num=1000)

    for n in range(len(t)):
        for f in range(len(df)):
            M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma], angle)

    return M, df, RF, t_max

def hard_pulse(M, flip, angle, t_max, N, BW, Gamma):

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

    for n in range(len(t)):
        for f in range(len(df)):
            M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[0, n]), np.imag(RF[0, n]), df[f]/Gamma], angle)

    return M, df, RF, t_max
