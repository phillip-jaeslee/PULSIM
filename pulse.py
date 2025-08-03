import time
import numpy as np
import torch
from mat_operator import cpu_rot, torch_rot
from bloch import bloch_rotate, torch_bloch_rotate
from file_import import import_file, read_xy_points
from joblib import Parallel, delayed
from pulse_shape_list import *

class cpu_pulse:
    
    global MULTI

    def import_shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma=42.577478, MULTI=False):

        start = time.time()

        ## shaped pulse calculator
        """
        M, df, RF, t_max = shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma, MULTI)
        parameters 
        input:
        M               - magnetization vector 
        angle           - flip angle position (x, y, z)
        flip            - flip angle (rad)
        t_max           - duration of pulse
        file_path       - file path for composite pulse
        BW              - bandwith (kHz)
        Gamma           - gyomagnetic ratio (default = 42.577478)
        MULTI           - boolean of Multiple process calculation (default = False)
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
            xy_temp = cpu_rot.Rot(xy_array[k, 1] * np.pi / 180) @ np.array([1, 0]).T
            RF_array[k, 1] = complex(xy_temp[0], xy_temp[1])
        if (max(xy_array[:,1])>=350):
            pul_type = "adiabatic"
        RF = xy_array[:, 0] * RF_array[:, 1]
        if (pul_type == "adiabatic"):
            RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt) * 2
        else:
            RF = (flip) * RF/ np.sum(RF) / (2*np.pi*Gamma*dt)

        df = np.linspace(-BW/2, BW/2, num=1000)

        if MULTI == True:
            M = Parallel(n_jobs=-1)(delayed(parallel_rotate_shape)(M[:, f], t, RF, Gamma, angle, dt, df[f]) for f in range(len(df)))
            M = np.array(M).T
        elif MULTI == False:
            for n in range(len(t)):
                for f in range(len(df)):
                    M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma], angle)
        else:
            raise TypeError(f'MULTI must be a BOOLEAN type, not {type(MULTI).__name__}')
        

        end = time.time()

        print('elapsed time: {} sec'.format(end-start) )

        return M, df, RF, t_max, N

    def shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma=42.577478, MULTI=False) :

        start = time.time()
        
        ## shaped pulse calculator
        """
        M, df, RF, t_max = shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma, MULTI)
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
        Gamma           - gyomagnetic ratio (default = 42.577478)
        MULTI           - boolean of Multiple process calculation (default = False)
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
        shape_funcs = {
            "sinc":         lambda: np.hamming(N).T * np.sinc(t),
            "cos":          lambda: np.hamming(N).T * np.cos(t),
            "sinc2p":       lambda: np.sinc(2 * np.pi * t),
            "eburp1":       lambda: E_BURP_1_pulse(duration=t_max, points=N),
            "eburp2":       lambda: E_BURP_2_pulse(duration=t_max, points=N),
            "iburp1":       lambda: I_BURP_1_pulse(duration=t_max, points=N),
            "iburp2":       lambda: I_BURP_2_pulse(duration=t_max, points=N),
            "uburp":        lambda: U_BURP_pulse(duration=t_max, points=N),
            "reburp":       lambda: RE_BURP_pulse(duration=t_max, points=N),
            "gausscasG3":   lambda: GAUSSCASCADE_G3_pulse(duration=t_max, points=N),
            "gausscasG4":   lambda: GAUSSCASCADE_G4_pulse(duration=t_max, points=N),
            "gausscasQ3":   lambda: GAUSSCASCADE_Q3_pulse(duration=t_max, points=N),
            "gausscasQ5":   lambda: GAUSSCASCADE_Q5_pulse(duration=t_max, points=N),
            "hermite":      lambda: HERMITE_pulse(duration=t_max, points=N),
            "seduce1":      lambda: SEDUCE_1_pulse(duration=t_max, points=N),
            "sneeze":       lambda: SNEEZE_pulse(duration=t_max, points=N),
            "qsneeze":      lambda: QSNEEZE_pulse(duration=t_max, points=N),
            "esnob":        lambda: eSNOB_pulse(duration=t_max, points=N),
            "i2snob":       lambda: i2SNOB_pulse(duration=t_max, points=N),
            "i3snob":       lambda: i3SNOB_pulse(duration=t_max, points=N),
            "rsnob":        lambda: rSNOB_pulse(duration=t_max, points=N),
            "dsnob":        lambda: dSNOB_pulse(duration=t_max, points=N),
            "hypsec":       lambda: HYPSEC_pulse(duration=t_max, points=N),
            "swrl11":       lambda: SWIRL11_pulse(duration=t_max, points=N),
            "swrl12":       lambda: SWIRL12_pulse(duration=t_max, points=N),
            "swrl17":       lambda: SWIRL17_pulse(duration=t_max, points=N)
        }

        try:
            RF_org = shape_funcs[shape]()
        except KeyError:
            raise ValueError(f"Unknown shape '{shape}'. Available shapes: {list(shape_funcs)}")
        
        RF = (flip) * RF_org/np.sum(RF_org) / (2*np.pi*Gamma*dt)

        df = np.linspace(-BW/2, BW/2, num=1000)

        if MULTI == True:
            M = Parallel(n_jobs=-1)(delayed(parallel_rotate_shape)(M[:, f], t, RF, Gamma, angle, dt, df[f]) for f in range(len(df)))
            M = np.array(M).T
        elif MULTI == False:
            for n in range(len(t)):
                for f in range(len(df)):
                    M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[n]), np.imag(RF[n]), df[f]/Gamma], angle)
        else:
            raise TypeError(f'MULTI must be a BOOLEAN type, not {type(MULTI).__name__}')


        end = time.time()

        print('elapsed time: {} sec'.format(end-start) )

        return M, df, RF, t_max, N

    def hard_pulse(M, flip, angle, t_max, N, BW, Gamma=42.577478, MULTI=False):

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
        Gamma           - gyomagnetic ratio (default = 42.577478)
        MULTI           - boolean of Multiple process calculation (default = False)        
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

        if MULTI == True:
            M = Parallel(n_jobs=-1)(delayed(parallel_rotate_hard)(M[:, f], t, RF, Gamma, angle, dt, df[f]) for f in range(len(df)))
            M = np.array(M).T
        elif MULTI == False:
            for n in range(len(t)):
                for f in range(len(df)):
                    M[:, f]  = bloch_rotate(M[:, f], dt, [np.real(RF[0, n]), np.imag(RF[0, n]), df[f]/Gamma], angle)            
        else:
            raise TypeError(f'MULTI must be a BOOLEAN type, not {type(MULTI).__name__}')                    
        
        end = time.time()

        print('elapsed time: {} sec'.format(end-start) )

        return M, df, RF, t_max, N


class torch_pulse:

    def torch_import_shaped_pulse(M, flip, angle, t_max, file_path, BW, Gamma=42.577478) :

        start = time.time()

        ## shaped pulse calculator (torch)
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
        Gamma           - gyomagnetic ratio (default = 42.577478)
        output:
        M               - final magnetization vector
        df              - bandwith array of the pulse
        RF              - pulse shape array
        RF              - pulse phase
        t               - time array of the pulse
        t_max           - duration of pulse (need to be stored to plot the pulse diagram)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        xy_array = import_file(file_path)
        N = len(xy_array)
        dt = t_max / N
        init = -N/2
        final = N/2
        t = torch.arange(init, final, dtype=torch.float32, device=device) * dt
        pul_type = ""
        angles = torch.tensor(xy_array[:, 1], dtype=torch.float32, device=device) * torch.pi / 180
        magnitudes = torch.tensor(xy_array[:, 0], dtype=torch.float32, device=device)
        RF_real = torch.cos(-angles)
        RF_imag = torch.sin(-angles)
        RF_array = torch.complex(magnitudes * RF_real, magnitudes * RF_imag)

        pul_type = "adiabatic" if (max(xy_array[:,1])>=350) else ""
        RF = RF_array
        RF = (flip) * RF/ torch.sum(RF) / (2*torch.pi*Gamma*dt)
        if pul_type == "adiabatic":
            RF *= 2

        df = torch.linspace(-BW/2, BW/2, steps=N, dtype=torch.float32, device=device)

        M = torch.tensor(M, dtype=torch.float32, device=device)

        RF_real_expanded = RF.real.expand(len(df), -1)
        RF_imag_expanded = RF.imag.expand(len(df), -1)
        df_expanded = df[:, None].expand(-1, N) / Gamma

        RF_angle = np.array(xy_array[:, 1])
        
        B = torch.stack([RF_real_expanded, RF_imag_expanded, df_expanded], dim=2)
        for n in range(len(t)):
            M = torch_bloch_rotate(M.T, dt, B[:, n, :], angle, Gamma).T

        end = time.time()
        RF = abs(RF)
        print('elapsed time: {} sec'.format(end-start) )

        return M.cpu().numpy(), df.cpu().numpy(), RF.cpu().numpy(), RF_angle, t_max, N

    def torch_shaped_pulse(M, flip, angle, t_max, shape, N, BW, Gamma=42.577478) :

        start = time.time()
        
        ## shaped pulse calculator (torch)
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
        Gamma           - gyomagnetic ratio (default = 42.577478)
        output:
        M               - final magnetization vector
        df              - bandwith array of the pulse
        RF              - pulse shape array
        t               - time array of the pulse
        t_max           - duration of pulse (need to be stored to plot the pulse diagram)
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        M = torch.tensor(M, dtype=torch.float32, device=device)

        dt = t_max / N
        init = -N/2
        final = N/2
        t = torch.arange(init, final, 1) * dt

        shape_funcs = {
            "sinc":         lambda: np.hamming(N).T * np.sinc(t),
            "cos":          lambda: np.hamming(N).T * np.cos(t),
            "sinc2p":       lambda: np.sinc(2 * np.pi * t),
            "eburp1":       lambda: E_BURP_1_pulse(duration=t_max, points=N),
            "eburp2":       lambda: E_BURP_2_pulse(duration=t_max, points=N),
            "iburp1":       lambda: I_BURP_1_pulse(duration=t_max, points=N),
            "iburp2":       lambda: I_BURP_2_pulse(duration=t_max, points=N),
            "uburp":        lambda: U_BURP_pulse(duration=t_max, points=N),
            "reburp":       lambda: RE_BURP_pulse(duration=t_max, points=N),
            "gausscasG3":   lambda: GAUSSCASCADE_G3_pulse(duration=t_max, points=N),
            "gausscasG4":   lambda: GAUSSCASCADE_G4_pulse(duration=t_max, points=N),
            "gausscasQ3":   lambda: GAUSSCASCADE_Q3_pulse(duration=t_max, points=N),
            "gausscasQ5":   lambda: GAUSSCASCADE_Q5_pulse(duration=t_max, points=N),
            "hermite":      lambda: HERMITE_pulse(duration=t_max, points=N),
            "seduce1":      lambda: SEDUCE_1_pulse(duration=t_max, points=N),
            "sneeze":       lambda: SNEEZE_pulse(duration=t_max, points=N),
            "qsneeze":      lambda: QSNEEZE_pulse(duration=t_max, points=N),
            "esnob":        lambda: eSNOB_pulse(duration=t_max, points=N),
            "i2snob":       lambda: i2SNOB_pulse(duration=t_max, points=N),
            "i3snob":       lambda: i3SNOB_pulse(duration=t_max, points=N),
            "rsnob":        lambda: rSNOB_pulse(duration=t_max, points=N),
            "dsnob":        lambda: dSNOB_pulse(duration=t_max, points=N),
            "hypsec":       lambda: HYPSEC_pulse(duration=t_max, points=N),
            "swrl11":       lambda: SWIRL11_pulse(duration=t_max, points=N),
            "swrl12":       lambda: SWIRL12_pulse(duration=t_max, points=N),
            "swrl17":       lambda: SWIRL17_pulse(duration=t_max, points=N)
        }

        try:
            RF_org = shape_funcs[shape]()
        except KeyError:
            raise ValueError(f"Unknown shape '{shape}'. Available shapes: {list(shape_funcs)}")
        
        RF_angle = np.ones((1, int(N)))
        # If pulse is real-only
        if np.isrealobj(RF_org):
            RF_angle = np.where(RF_org >= 0, 0.0, 180.0)
            RF_tensor = torch.Tensor(RF_org, dtype=torch.complex128, device=device)
            RF = (flip) * RF_tensor/torch.sum(RF_tensor) / (2*torch.pi*Gamma*dt)

        # If pulse has complex components
        else:
            phase_rad = np.angle(RF_org)            # returns −π to π
            RF_angle = (-np.degrees(phase_rad)) % 360      # convert to degrees
            RF_tensor = torch.complex(torch.Tensor(RF_org.real), torch.Tensor(RF_org.imag))
            RF = (flip) * RF_tensor/torch.sum(RF_tensor) / (2*torch.pi*Gamma*dt) * 2

        df = torch.linspace(-BW/2, BW/2, steps=N, dtype=torch.float32, device=device)

        # Expand dimensions to align properly for stacking
        # tensor.expand = repeating the tensor (-1 without changing dimension)
        RF_real_expanded = RF.real.expand(len(df), -1)
        RF_imag_expanded = RF.imag.expand(len(df), -1)
        df_expanded = df[:, None].expand(-1, N) / Gamma

        # B = [RF.real, RF.imag, df]
        B = torch.stack([RF_real_expanded, RF_imag_expanded, df_expanded], dim=2)

        for n in range(len(t)):
            M = torch_bloch_rotate(M.T, dt, B[:, n, :], angle, Gamma).T

        end = time.time()
        RF = abs(RF)
        print('elapsed time: {} sec'.format(end-start) )

        return M.cpu().numpy(), df.cpu().numpy(), RF.cpu().numpy(), RF_angle, t_max, N


    def torch_hard_pulse(M, flip, angle, t_max, N, BW, Gamma=42.577478):
        start = time.time()

        ## hard pulse calculator (torch)


        """
        M, df, RF, t_max, N = sc_hard_pulse(M, flip, angle, t_max, N, BW, Gamma)
        parameters 
        input:
        M               - magnetization vector 
        N               - the number of points of the pulse
        dt              - size of each step
        angle           - flip angle position (x, y, z)
        flip            - flip angle (rad)
        t_max           - duration of pulse
        BW              - bandwith (kHz)
        Gamma           - gyomagnetic ratio (default = 42.577478)
        output:
        M               - final magnetization vector
        df              - bandwith array of the pulse
        RF              - pulse shape array
        t_max           - duration of pulse (need to be stored to plot the pulse diagram)
        N               - the number of points of the pulse
        """    

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert inputs to torch tensors and move to device
        M = torch.tensor(M, dtype=torch.float32, device=device)
        dt = t_max / N
        init = -N / 2
        final = N / 2
        t = torch.arange(init, final, device=device) * dt
        RF = torch.ones((1, int(N)), dtype=torch.float32, device=device)
        RF = (flip * RF) / torch.sum(RF) / (2 * torch.pi * Gamma * dt)
        df = torch.linspace(-BW / 2, BW / 2, steps= N, device=device)

        RF_angle = np.ones((1, int(N)))

        if flip > 0:
            RF_angle = RF_angle * 0
        elif flip < 0:
            RF_angle = RF_angle * 180

        # Expand dimensions to align properly for stacking
        # tensor.expand = repeating the tensor (-1 without changing dimension)
        RF_expanded = RF.expand(len(df), -1)
        zeros_expanded = torch.zeros(len(df), N, device=device)
        df_expanded = df[:, None].expand(-1, N) / Gamma

        # B = [RF, 0, df]
        B = torch.stack([RF_expanded, zeros_expanded, df_expanded], dim=2)

        for n in range(len(t)):
            M = torch_bloch_rotate(M.T, dt, B[:, n, :], angle, Gamma).T

        end = time.time()
        print('elapsed time: {} sec'.format(end - start))

        return M.cpu().numpy(), df.cpu().numpy(), RF.cpu().numpy(), RF_angle, t_max, N
    

    def torch_gaussian(x, mu, sig):
        return (
            1.0 / (torch.sqrt(2.0 * torch.pi) * sig) * torch.exp(-torch.pow((x - mu) / sig, 2.0) / 2)
        )

def parallel_rotate_hard(M_f, t, RF, Gamma, angle, dt, df_f):
    for n in range(len(t)):
            M_f  = bloch_rotate(M_f, dt, [np.real(RF[0, n]), np.imag(RF[0, n]), df_f/Gamma], angle)

    return M_f

def parallel_rotate_shape(M_f, t, RF, Gamma, angle, dt, df_f):
    for n in range(len(t)):
            M_f  = bloch_rotate(M_f, dt, [np.real(RF[n]), np.imag(RF[n]), df_f/Gamma], angle)

    return M_f
    