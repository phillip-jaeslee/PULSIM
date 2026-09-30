"""
calibration.py -- how a normalized waveform becomes a physical RF amplitude.

A shape is a dimensionless envelope. Turning it into a field in mT requires a
calibration strategy, and different kinds of pulse use different ones. This
module holds those strategies; the propagation engine is identical for all of 
them (see docs/PHYSICS_SPECIFICATION.md section 1)

Units follow the specification: nu1 in kHz, B1 in mT, duration in ms,
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

    Note nu1_max/sqrt(Q) = a*beta*sqrt(mu)/(pi*T) is independent of Q -- which
    is why Bruker's integradia reports gamma*B1max/2pi/sqrt(Q) rather than
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
        """nu1_max / sqrt(Q), kHz -- independent of Q."""
        return self.half_width * self.beta * np.sqrt(self.mu) / (np.pi * duration_ms)
    def q_for(self, nu1_max, duration_ms):
        """The adiabaticity actually achieved at a given RF amplitude.

        Inverse of nu1_for. Diagnostic only -- never an input to propagation.
        Meaningful when the caller supplies nu1_max explicitly, where the
        realised Q may differ from the design q_mid.
        """
        root = nu1_max * np.pi * duration_ms / (self.half_width * self.beta)
        return float(root * root / self.mu)

@dataclass(frozen=True)
class ResonanceCrossing:
    """Where an adiabatic sweep passes through resonance, in normalized time.
    u           : crossing position, u = t/T in [0, 1]
    rate_norm   : |d(delta_omega)/du| there, rad per unit u^2. The physical
                  sweep rate is rate_norm / T^2 in rad/ms^2 (T in ms).
    amp_rel     : RF amplitude at the crossing relative to the peak, 0...1
    at_edge     : True for a half passage, which ends on resonance
    """

    u: float
    rate_norm: float
    amp_rel: float
    at_edge: bool

def resonance_crossing(envelope, window=0.01, amp_floor=1e-3, edge_tol=0.01):
    """Locate the resonance crossing of a frequency-swept envelope.

    Measured from the waveform itself, so it works for any adiabatic family
    without a per-family sweep-rate formula. The instantaneous offset is the 
    phase step between neighboring samples (wrap-safe), kept only where the
    amplitude is above amp_floor -- the phase is undefined at a zero edge.

    Full passage: the offset changes sign exactly once, inside the pulse.
    Half passage: no sign change, and the offset extrapolates to zero at one
    end. Anything else (no crossing, or several, as in a composite chirp) has
    no single Q, and this raises rather than picking one.

    The slope comes from a cubic fit centered on the crossing -- its linear
    coefficient is the slope AT u_c. A straight line would average over the
    tanh-like bend of a HypSec sweep and read 0.2% low.
    """
    env = np.asarray(envelope, dtype=complex)
    n = len(env)
    amp = np.abs(env)
    peak = amp.max()

    step = np.angle(env[1:] * np.conj(env[:-1]))        # rad per sample
    offset = step * (n - 1)                             # rad per unit u
    u_mid = (np.arange(n - 1) + 0.5) / (n - 1)          # where each step sits
    keep = np.minimum(amp[1:], amp[:-1]) > amp_floor * peak
    offset, u_mid = offset[keep], u_mid[keep]

    flips = np.flatnonzero(np.diff(np.sign(offset)) != 0)
    sweep = np.abs(offset).max()
    half_win = max(4, int(window * n))

    if len(flips) == 1:
        k = flips[0]
        f0, f1 = offset[k], offset[k + 1]       # bracket the zero
        u_c = u_mid[k] + (u_mid[k + 1] - u_mid[k]) * f0 / (f0 - f1)
        sel = slice(max(0, k - half_win), k + half_win + 2)
        at_edge = False
    elif len(flips) == 0:
        at_start = abs(offset[0]) < abs(offset[-1])
        u_c = 0.0 if at_start else 1.0
        sel = slice(0, 2 * half_win) if at_start else slice(-2 * half_win, None)
        at_edge = True
    else:
        raise ValueError(
            f"found {len(flips)} resonance crossings; this waveform has no single Q. Supply nu1_max explicitly."
        )
    coeffs = np.polyfit(u_mid[sel] - u_c, offset[sel], 3)
    if at_edge and abs(coeffs[-1]) > edge_tol * sweep:
        raise ValueError(
            "the sweep never reaches resonance (no sign change, and neither "
            "end extrapolates to zero offset); no Q can be defined. "
            "Supply nu1_max explicitly."
        )

    amp_rel = float(np.interp(u_c, np.linspace(0.0, 1.0, n), amp) / peak)
    return ResonanceCrossing(u=float(u_c), rate_norm=float(abs(coeffs[-2])), amp_rel=amp_rel, at_edge=at_edge)

@dataclass(frozen=True)
class NumericAdiabaticCalibration:
    """Adiabatic calibration for any single-sweep shape, from its waveform

    Same definition of Q as AdiabaticCalibration, evaluated where the sweep crosses resonance:

        Q = omega1(t_c)^2 / |d(delta_omega)/dt|(t_c)
          = (2*pi * nu1_max * amp_rel * T)^2 / rate_norm
    
    so nu1_max = sqrt(Q * rate_norm) / (2*pi * amp_rel * T) [kHz, T in ms]

    rate_norm and amp_rel come frm resonance_crossing(), so no per-family
    sweep-rate formula is needed. For HypSec this reproduces the closed form
    in AdiabaticCalibration; the tests hold it to that.
    """
    q_mid: float
    crossing: ResonanceCrossing

    @classmethod
    def from_envelope(cls, envelope, q_mid):
        return cls(q_mid=float(q_mid), crossing=resonance_crossing(envelope))

    def nu1_over_sqrt_q(self, duration_ms):
        """Peak nutation frequency per sqrt(Q), kHz -- Q-independent."""
        c = self.crossing
        return np.sqrt(c.rate_norm) / (2 * np.pi * c.amp_rel * duration_ms)

    def nu1_for(self, duration_ms):
        """Peak nutation frequency in kHz."""
        return self.nu1_over_sqrt_q(duration_ms) * np.sqrt(self.q_mid)

    def q_for(self, nu1_max, duration_ms):
        """The adiabaticity actually achieved at a given RF amplitude.

        Inverse of nu1_for. Diagnostic only -- never an input to propagation.
        """
        return float((nu1_max / self.nu1_over_sqrt_q(duration_ms)) ** 2)