import numpy as np
import matplotlib.pyplot as plt


##############################
#######CLASSIC PULSES#########
##############################


#### BURP PULSE ####

def E_BURP_1_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
    A = np.array([0.88, -1.04, -0.24, 0.14, 0.03, 0.04, -0.03, 0.00])
    B = np.array([-0.40, -1.42, 0.77, 0.06, 0.03, -0.04, -0.02, 0.01])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.23
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude

    return amp

def E_BURP_2_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
    A = np.array([0.91, 0.29, -1.28, -0.05, 0.04, 0.02, 0.06, 0.00, -0.02, 0.00])
    B = np.array([-0.16, -1.82, 0.18, 0.42, 0.07, 0.07, -0.01, -0.04, 0.00, 0.00])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.26
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

def I_BURP_1_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
    A = np.array([0.70, -0.15, -0.94, 0.11, -0.02, -0.04, 0.01, -0.02, -0.01])
    B = np.array([-1.54, 1.01, -0.24, -0.04, 0.08, -0.04, -0.01, 0.01, -0.01])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.50
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp


def I_BURP_2_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
    A = np.array([0.81, 0.07, -1.25, -0.24, 0.07, 0.11, 0.05, -0.02, -0.03, -0.02, 0.00])
    B = np.array([-0.68, -1.38, 0.20, 0.45, 0.23, 0.05, -0.04, -0.04, 0.00, 0.01, 0.01])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.50
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

def U_BURP_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
    A = np.array([-1.42, -0.37, -1.84, 4.40, -1.19, 0.00, -0.37, 0.50, -0.31, 0.18, -0.21, 0.23, -0.12, 0.07, -0.06, 0.06, -0.04, 0.03, -0.02, 0.02])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.27
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

def RE_BURP_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
    A = np.array([-1.02, 1.11, -1.57, 0.83, -0.42, 0.26, -0.16, 0.10, -0.07, 0.04, -0.03, 0.01, -0.02, 0.00, -0.01])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.49
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

#### GAUSSIAN CASCADE PULSE ####


def GAUSSCASCADE_G3_pulse(duration=1.0, points=100, amplitude=1.0):
    t = np.linspace(0, duration, points)

    # G(-270)G(270)G(180) from L. Emsley & G. Bodenhausen, Chem. Phys. Lett. 165, 469 (1990).
    t_half = np.array([18.9, 18.3, 24.3]) / 100 * duration / 2
    t_max = np.array([28.7, 50.8, 79.5]) / 100 * duration
    omega_max = np.array([-1.00, 1.37, 0.49])

    a = np.log(2) / (t_half ** 2)

    amp = np.zeros_like(t)
    for n in range(len(t_half)):
        amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

    amp /= np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Apply desired scaling
    return amp

def GAUSSCASCADE_G4_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)

    # G(-270)G(270)G(180)G(90) from L. Emsley & G. Bodenhausen, Chem. Phys. Lett. 165, 469 (1990).
    t_half = np.array([17.2, 12.9, 11.9, 13.9]) / 100 * duration / 2
    t_max = np.array([17.7, 49.2, 65.3, 89.2]) / 100 * duration
    omega_max = np.array([0.62, 0.72, -0.91, -0.33])

    a = np.log(2) / (t_half ** 2)

    amp = np.zeros_like(t)
    for n in range(len(t_half)):
        amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

    amp /= np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Apply desired scaling
    return amp

def GAUSSCASCADE_Q3_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)

    # Parameters from literature (L. Emsley & G. Bodenhausen, J. Magn. Reson. 97, 135-148 (1992).)
    t_half = np.array([18.0, 18.3, 24.5]) / 100 * duration / 2
    t_max = np.array([30.6, 54.5, 80.4]) / 100 * duration
    omega_max = np.array([-4.39, 4.57, 2.60])

    a = np.log(2) / (t_half ** 2)

    amp = np.zeros_like(t)
    for n in range(len(t_half)):
        amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

    amp /= np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Apply desired scaling
    return amp

def GAUSSCASCADE_Q5_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)

    # Parameters from literature (L. Emsley & G. Bodenhausen, J. Magn. Reson. 97, 135-148 (1992).)
    t_half = np.array([18.6, 13.9, 14.3, 29.0, 13.7]) / 100 * duration / 2
    t_max = np.array([16.2, 30.7, 49.7, 52.5, 80.3]) / 100 * duration
    omega_max = np.array([-1.48, -4.34, 7.33, -2.30, 5.66])

    a = np.log(2) / (t_half ** 2)

    amp = np.zeros_like(t)
    for n in range(len(t_half)):
        amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

    amp /= np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Apply desired scaling
    return amp


#### HERMITE PULSE ####

def HERMITE_pulse(duration=1.0, points=1000, amplitude=1.0, coefficient=1.0, truncate=1):
    t_original = np.linspace(-3, 3, 1000)  # High-res for accurate thresholding
    truncate /= 100 # change threshold into percentage
    # Pulse definition from literature
    T = 1 / coefficient
    amp = (1 - (1.782 * (t_original / T) ** 2)) * np.exp(-(t_original / T) ** 2)
    
    amp /= np.max(np.abs(amp))  # Normalize
    amp *= amplitude

    # Find where amp ≥ threshold
    mask = abs(amp) >= truncate
    t_start = t_original[mask][0]
    t_end = t_original[mask][-1]

    # Regenerate t from t_start to t_end
    t = np.linspace(t_start, t_end, points)
    amp = (1 - (0.956 * (t / T) ** 2)) * np.exp(-(t / T) ** 2)
    amp /= np.max(np.abs(amp))
    amp *= amplitude
    return amp

#### SEDUCE PULSE ####

def SEDUCE_1_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(-0.5, 0.5, points)

    # Parameters from literature (M.A. McCoy & L. Mueller, J. Magn. Reson. A 101, 122-130 (1993).)
    c = 10 * np.tanh(0.08 * abs(t))**2
    amp = np.sin(np.pi* (t+0.5))**2 / np.cosh(500 * t * c)
    return amp

#### SNEEZE PULSE ####

def SNEEZE_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (J. M. Nuzillard & R. Freeman, J. Magn. Reson. A 110, 252-256 (1994).)
    A = np.array([0.730, 1.091, -0.975, -1.038, -0.047, 0.083, -0.001, 0.036, 0.061, 0.005, -0.026, -0.013])
    B = np.array([0.001, -0.927, -1.706, 0.399, 0.454, 0.089, 0.036, 0.052, -0.017, -0.052, -0.020, 0.001])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.248
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp


def QSNEEZE_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (Kupce, E. & Freeman, R. (1995) J. Magn. Reson., Ser. A112, 134−137.)
    A = np.array([0.934, 0.180, -1.527, 0.003, 0.143, 0.050, 0.072, -0.015, -0.040, -0.005])
    B = np.array([-0.197, -1.772, 0.204, 0.619, 0.076, 0.039, -0.025, -0.060, 0.005, 0.017])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.250
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp


#### SNOB PULSE ####

def eSNOB_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
    A = np.array([-0.6176, -0.0373, -0.0005, -0.0182, -0.0058, -0.0036, -0.0051, -0.0031, -0.0017])
    B = np.array([-0.4855, 0.1260, -0.0191, -0.0005, -0.0003, 0.0017, -0.0013, 0.0001, -0.0025])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.7500
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

def i2SNOB_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
    A = np.array([-0.2687, -0.2972, 0.0989, -0.0010, -0.0168, 0.0009, -0.0017, -0.0013, -0.0014])
    B = np.array([-1.1461, 0.4016, 0.0736, -0.0307, 0.0079, 0.0062, 0.0003, -0.0002, 0.0009])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.5000
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

def i3SNOB_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
    A = np.array([0.2801, -0.9995, 0.1928, 0.0967, -0.0480, -0.0148, 0.0088, -0.0002, -0.0030])
    B = np.array([-1.1990, 0.4893, 0.2439, -0.0816, -0.0409, 0.0234, 0.0036, -0.0042, 0.0001])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.5000
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

def rSNOB_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
    A = np.array([-1.1472, 0.5572, -0.0829, 0.0525])
    B = np.array([0.0000, 0.0000, 0.0000, 0.0000])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.5000
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

def dSNOB_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
    A = np.array([-1.0662, 0.4466, 0.1673, -0.0049, -0.0753, 0.0001, 0.0144, 0.0041])
    B = np.array([-2.4513, -0.2442, 0.6025, 0.1362, -0.0521, -0.0210, 0.0070, 0.0079])
    
    omega = 2 * np.pi / duration
    amp = np.zeros_like(t)
    amp += 0.5000
    for n in range(len(A)):
        amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp

##############################
#######ADIABATIC PULSE########
##############################

#### HYPERBOLIC SECANT PULSE ####


def HYPSEC_pulse(duration=4.0, points=1000, truncate=1, sweepwidth=20, amplitude=1.0, beta=5.29829, mu=5.92944, low_to_high=True):

    # Adiabatic Hyperbolic Secant pulse from literature (M.S. Silver, R.I. Joseph & D.I. Hoult, J. Magn. Reson. 59, 347 (1984).)
    t_original = np.linspace(-3, 3, 10_000)
    truncate /= 100 # change threshold into percentage
    if low_to_high == False:
        amp_original = (np.cosh(beta * t_original))**(1 + 1j * mu)  # Complex amplitude
    elif low_to_high == True:
        amp_original = (np.cosh(beta * t_original))**(-1 - 1j * mu)
    
    real = np.real(amp_original)
    imag = np.imag(amp_original)
    amplitude = np.sqrt(real**2 + imag**2)
    mask = abs(amplitude) > truncate
    t_start = t_original[mask][0]
    t_end = t_original[mask][-1]
    
    t = np.linspace(t_start, t_end, points)
    if low_to_high == False:
        amp = (np.cosh(beta * t))**(1 + 1j * mu)  # Complex amplitude
    elif low_to_high == True:
        amp = (np.cosh(beta * t))**(-1 - 1j * mu)
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1

    amplitude = np.sqrt(real**2 + imag**2)
    phase = np.angle(amp)  # Instantaneous phase in radians
    phase = (-np.degrees(phase)) % 360
    return amp


#### SIN/COS PULSE ####
## TODO: ADD MORE ADIABATIC PULSES

def SINCOS_pulse(duration=1.0, points=1000, amplitude=1.0, low_to_high=True):

    # Adiabatic Hyperbolic Secant pulse from literature (M.S. Silver, R.I. Joseph & D.I. Hoult, J. Magn. Reson. 59, 347 (1984).)
    t = np.linspace(0, duration, points)
    amp = np.sin(2 * np.pi * duration * t / 2) ** (1 + 1j)
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude

    real = np.real(amp)
    imag = np.imag(amp)
    amplitude = np.sqrt(real**2 + imag**2)
    phase = np.angle(amp)  # Instantaneous phase in radians
    phase = (-np.degrees(phase)) % 360
    return phase


def sincospulse(duration=1.0, points=1000, phasefac=4.0, fullpass=True, swdir=1, amplitude=1.0):
    """
    Generate a SinCos adiabatic pulse shape.

    Parameters:
        npoints  : int     — number of points
        phasefac : float   — phase sweep factor (0 to 16)
        fullpass : bool    — full passage (True) or half passage (False)
        swdir    : int     — sweep direction: -1 = low-to-high, +1 = high-to-low
        amplitude: float   — peak amplitude scaling

    Returns:
        t        : np.array — time vector
        pulse    : np.array — complex RF pulse (real + 1j * imag)
    """
    t = np.linspace(0, duration, points)

    # Amplitude modulation: sin envelope
    amp = np.sin(np.pi * t / duration / 2) if fullpass else np.sin((t + np.pi / 2) / 2)

    # Frequency modulation: dφ/dt = PHASEFAC * cos(t), integrate to get phase
    dphi_dt = phasefac * np.sin(np.pi * t / duration / 2) * swdir
    phase = np.cumsum(dphi_dt) * (2 * np.pi / points)  # phase in radians

    # Combine amplitude and phase
    amp /= np.max(np.abs(amp))  # normalize
    amp *= amplitude

    real = amp * np.cos(phase)
    imag = amp * np.sin(phase)

    phase = (np.degrees(phase)) % 360

    return phase

#######################################
#######SPECIAL DECOUPLING PULSE########
#######################################

#### SWIRL PULSE ####

def SWIRL11_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (H. Geen & J.-M. Boehlen, J. Magn. Reson. 125, 376-382 (1997)..)
    A = np.array([1.71, 1.87, 2.02, 2.19, 1.99, 1.41])
    B = np.array([2.66, -2.84, -3.22, 3.00, -3.03, -2.06])
    omega_n = np.array([1.58, 1.70, 1.90, 1.86, 1.81, 1.25])
    phi = np.array([-1.00, 0.99, 1.01, -0.94, 0.99, 0.97])
    omega = 2 * np.pi / duration

    A_n = 2 * np.abs(omega_n) * np.cos(phi)
    B_n = -2 * np.abs(omega_n) * np.sin(phi)

    amp = np.zeros_like(t)
    amp += 0.000
    for n in range(len(omega_n)):
        amp += A_n[n] * np.cos((2*n+1) * omega * t) + B_n[n] * np.sin((2*n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp


def SWIRL12_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (H. Geen & J.-M. Boehlen, J. Magn. Reson. 125, 376-382 (1997)..)
    A = np.array([1.71, -0.01, 1.84, -0.01, 1.97, 0.01, 1.81, 0.05, 1.99, -0.01, 1.43, 0.11])
    B = np.array([2.73, 0.02, -2.81, 0.02, -3.36, -0.04, 3.02, -0.09, -3.56, 0.02, -2.43, -0.43])
    omega_n = np.array([1.61, 0.01, 1.68, 0.01, 1.95, 0.02, 1.76, 0.05, 2.04, 0.01, 1.41, 0.22])
    phi = np.array([-1.01, -1.86, 0.99, -2.16, 1.04, 1.30, -1.03, 1.10, 1.06, -2.11, 1.04, 1.32])
    omega = 2 * np.pi / duration

    A_n = 2 * np.abs(omega_n) * np.cos(phi)
    B_n = -2 * np.abs(omega_n) * np.sin(phi)

    amp = np.zeros_like(t)
    amp += 0.000
    for n in range(len(omega_n)):
        amp += A_n[n] * np.cos((n+1) * omega * t) + B_n[n] * np.sin((n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp


def SWIRL17_pulse(duration=1.0, points=1000, amplitude=1.0):
    t = np.linspace(0, duration, points)
    
    # Fourier coefficients from literature (H. Geen & J.-M. Boehlen, J. Magn. Reson. 125, 376-382 (1997)..)
    A = np.array([1.70, 1.77, 1.75, 1.61, 1.71, 1.87, 2.45, 2.27])
    B = np.array([2.42, -2.64, -2.73, 2.51, -2.66, -3.12, 3.29, -2.98, -2.49])
    omega_n = np.array([1.48, 1.59, 1.62, 1.49, 1.58, 1.82, 2.05, 1.87, 1.36])
    phi = np.array([-0.96, 0.98, 1.00, -1.00, 1.00, 1.03, -0.93, 0.92, 1.16])
    omega = 2 * np.pi / duration

    A_n = 2 * np.abs(omega_n) * np.cos(phi)
    B_n = -2 * np.abs(omega_n) * np.sin(phi)

    amp = np.zeros_like(t)
    amp += 0.000
    for n in range(len(omega_n)):
        amp += A_n[n] * np.cos((2*n+1) * omega * t) + B_n[n] * np.sin((2*n+1) * omega * t)
    
    amp = amp / np.max(np.abs(amp))  # Normalize to max 1
    amp *= amplitude  # Scale to desired amplitude
    return amp


def plot_pulse(pulse, title="Pulse Shape", xlabel="Points", ylabel="Amplitude"):
    plt.figure()
    plt.plot(pulse.real, label="Real")
    if np.iscomplexobj(pulse):
        plt.plot(pulse.imag, label="Imag", linestyle="--")
    plt.title(title, fontname="Arial")
    plt.xlabel(xlabel, fontname="Arial")
    plt.ylabel(ylabel, fontname="Arial")
    plt.grid(True)
    plt.legend()
    plt.show()

#pulse = sincospulse()
#pulse = HYPSEC_pulse(duration=2, points=1000)
#plot_pulse(pulse, title="Hyperbolic Secant Pulse")

