"""
calibration.py -- how a normalized waveform becomes a physical RF amplitude.

A shape is a dimesionless envelope. Turning it into a field in mT requires a
calibration strategy, and different kinds of pulse use different ones. This
module holds those strategies; the propagation engine is identical for all of 
them (see docs/PHYSICS_SPECIFICATION.md section 1)

Units follow the sepcification: nu1 in kHz, B1 in mT, duration in ms,
Gamma = gamma / (2 * pi) in kHz/mT, flip angles in rad
"""

from dataclasses import dataclass

import numpy as np

@dataclass
class AreaCalibration:
    """Classical fixed-axis amplitude-modulated pulse.

    The flip angle is a literal rotation angle, and the RF amplitude follows
    from the pulse area:  beta = 2*pi * nu1_max * integral(a(t) dt).

    signed_integral is the normalized SIGNED integral,
        sum(envelope.real) / N / max|envelope|
    which is the same quantity Bruker stores as SHAPE_INTEGFAC. Signed, not
    absolute: a 0/180-degree phase step means those samples rotate about -x,
    and that cancellation is physical.
    """

    signed_integral: float

    def nu1_for(self, flip_rad, duration_ms):
        """Peak nutation frequency in kHz"""
        return flip_rad / (2 * np.pi * duration_ms * self.signed_integral)

def signed_integral_of(envelope):
    """The normalized signed integral of a waveform (Bruker INTEGFAC)"""
    env = np.asarray(envelope)
    return float(np.real(env.sum()) / len(env) / np.abs(env).max())


def beta_from_truncation(truncation_percent):
    """Hyperbolic-secant beta from the amplitude truncation level.

    The envelope is sech(beta*u) on u in [-1, 1], truncated where it falls to
    `truncation_percent` of its peak:  sech(beta) = truncation  =>
    beta = arccosh(1/truncation).

    Reproduces Bruker SHL_BETA: 1% -> 5.298292366 (file: 5.298292).
    """
    return float(np.arccosh(100.0 / truncation_percent))

def mu_from_sweep_width(sweep_width_1s_hz, beta):
    """HypSec mu from the sweep width quoted for a 1-second pulse.

    The stored phase gives a sweep of  SW = 2*mu*beta/pi  (Hz for T = 1 s),
    so  mu = pi*SW/(2*beta).

    Reproduces Bruker SHL_MU: SW=20 Hz, 1% truncation -> 5.929443747
    (file: 5.929443). Note there is NO tanh(beta) factor -- including one
    shifts the result by 3e-4 and no longer matches the vendor file.
    """
    return float(np.pi * sweep_width_1s_hz / (2.0 * beta))

@dataclass
class AdiabaticCalibration:
    """Bruker-style adiabatic pulse: RF strength from Q and the sweep rate.

    Flip angle plays no part. With the adiabaticity factor
    Q0 = omega1_max^2 / |d(delta_omega)/dt| at the resonance crossing, and the
    HypSec on-resonance sweep rate 4*a^2*mu*beta^2/T^2:

        nu1_max = (a*beta/(pi*T)) * sqrt(mu*Q)        [kHz, T in ms]

    Note nu1_max/sqrt(Q) = a*beta/(pi*T) is independent of Q -- which is why
    Bruker's integradia reports gamma*B1max/2pi/sqrt(Q) rather than
    gamma*B1max/2pi.

    FULL PASSAGE ONLY. For a half passage the resonance crossing sits at the
    edge of the sweep rather than its centre, so the sweep rate differs and
    this prefactor does not apply.
    """

    q_mid: float
    mu: float
    beta: float
    half_width: float = 1.0
    convention: str = "bruker_hs_full"

    def nu1_for(self, duration_ms):
        """Peak nutation frequency in kHz"""
        return (self.half_width * self.beta / (np.pi * duration_ms)) * np.sqrt(self.mu * self.q_mid)

    def nu1_over_sqrt_q(self, duration_ms):
        """Bruker integradia's Q-independent output."""
        return self.half_width * self.beta / (np.pi * duration_ms)
    
