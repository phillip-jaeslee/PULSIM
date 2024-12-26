import numpy as np
import matplotlib.pyplot as plt

def plot_pulse(M, RF, RF_angle, df, t, power=True, phase=True, label_Mx=True, label_My=True, label_Mz=True, label_Mxy=False):
    ## exciation profile plot module
    """
    plot_pulse(M, RF, RF_angle, df, t)
    input:
    M           - magnetization vector
    RF          - pulse shape array
    RF_angle    - pulse phases
    df          - bandwith array of the pulse
    t           - time array of the pulse
    (optional)
    power       - plot the power vs time (default = True)
    phase       - plot the phase vs time (default = True)
    label_      - plot the excitation profile of Mx, My, Mz, Mxy
                  (default Mx, My, Mz = True, Mxy = False)
    """
    if power==True and phase==True: # draw both power, phase plot, and exciation profile
        fig, axs = plt.subplots(3, 1, figsize=(6, 9))
        axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
        axs[0].plot(t, RF.T)
        axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
        axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
        df = df * 1000
        axs[1].plot(t, RF_angle.T)
        axs[1].set(xlabel='time (ms)', ylabel='Phase(˚)')
        axs[1].set_yticks(np.arange(0, 370, 60))

        if label_Mz: axs[2].plot(df, M[2,:], label="Mz")
        if label_My: axs[2].plot(df, M[1,:], label="My")
        if label_Mx: axs[2].plot(df, M[0,:], label="Mx")
        if label_Mxy: axs[2].plot(df, np.sqrt(M[1, :]**2 + M[0, :]**2), label="|Mxy|")
        axs[2].set(xlabel='frequency (Hz)', ylabel='flip')
        axs[2].legend()
    elif power==True and phase==False: # draw power plot and exciation profile
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))
        axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
        axs[0].plot(t, RF.T)
        axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
        axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
        df = df * 1000
        if label_Mz: axs[1].plot(df, M[2,:], label="Mz")
        if label_My: axs[1].plot(df, M[1,:], label="My")
        if label_Mx: axs[1].plot(df, M[0,:], label="Mx")
        if label_Mxy: axs[1].plot(df, np.sqrt(M[1, :]**2 + M[0, :]**2), label="|Mxy|")
        axs[1].set(xlabel='frequency (Hz)', ylabel='flip')
        axs[1].legend()
    else: # draw only the exication profile
        fig, axs = plt.subplots(1, 1, figsize=(6, 3))
        df = df * 1000
        if label_Mz: axs.plot(df, M[2,:], label="Mz")
        if label_My: axs.plot(df, M[1,:], label="My")
        if label_Mx: axs.plot(df, M[0,:], label="Mx")
        if label_Mxy: axs.plot(df, np.sqrt(M[1, :]**2 + M[0, :]**2), label="|Mxy|")
        axs.set(xlabel='frequency (Hz)', ylabel='flip')
        axs.legend()
    plt.show()
    return fig

def save_figure(fig, save=False,file_path=""):
    if save: fig.savefig(file_path + ".png")

    print("Saving figure:" + file_path + ".png")
    
