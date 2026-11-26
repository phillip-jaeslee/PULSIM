"""
Bloch-Siegert phase of frequency-shifted shaped pulses.

A selective pulse applied away from the carrier (a frequency-shifted shaped
pulse) also turns the spins it does not excite. PULSIM measures this extra
phase, phi_BS, as the transverse phase after the sequence minus the phase after
a delay of the same total length, so free precession at the offset cancels and
only the rotation caused by the RF field remains.

Used by module 2 of PULSIM_extension.ipynb and by
tutorials/tutorial_bloch_siegert.py. Offsets are in kHz, times in ms and phases
in degrees, as everywhere in PULSIM.
"""

import numpy as np

from .pulse_oo import Pulse
from .pulse_sequence import PulseSequence
from .liouville import Delay
from .spin_system import gyro_ratio

# -- units
def ppm_to_khz(ppm, ref_ppm, mhz):
    """Offset (kHz) of a shift 'ppm' from 'ref_ppm', for a nucleus observed at 'mhz' MHz."""
    return (np.asarray(ppm, dtype=float) - ref_ppm) * mhz / 1000.0

def nucleus_mhz(b0_mhz, nucleus):
    """Frequency (MHz) of 'nucleus' in the magnet whose 1H frequency is 'b0_mhz'."""
    return b0_mhz * abs(gyro_ratio(nucleus) / gyro_ratio("H"))


# -- a pulse played away from the carrier
class OffsetPulse(Pulse):
    """A Pulse played `offset_khz` away from the carrier (a frequency-shifted
    shaped pulse). Its RF carries the phase ramp exp(+i 2 pi nu t), zero at
    `align` of the pulse (0 start, 0.5 center, 1 end -- the meaning of Bruker's
    SPOAL). The amplitude is specified as for the same pulse on resonance; only
    the phase changes. The pulse acts at +offset_khz for positive and negative
    gamma alike."""

    def __init__(self, shape, flip=None, axis="x", backend=None, offset_khz=0.0, align=0.5):
        super().__init__(shape, flip, axis=axis, backend=backend)
        self.offset_khz, self.align = float(offset_khz), float(align)

    def calibrated_rf(self):
        rf = super().calibrated_rf()
        t = (np.arange(len(rf)) + 0.5) * self.shape.dt
        return rf * np.exp(2j * np.pi * self.offset_khz * (t - self.align * self.shape.duration))


# -- helpers
def _magnetization(n, along):
    M = np.zeros((3, n))
    M["xyz".index(along)] = 1.0
    return M

def run(elements, df_khz, start="x"):
    """Magnetization (3, n) after `elements` (pulses and Delays), starting along
    x, y or z at every offset of `df_khz`."""
    df = np.asarray(df_khz, dtype=float)
    return PulseSequence(list(elements)).run(_magnetization(len(df), start), df)

def duration(elements):
    """Total length (ms) of pulses and Delays."""
    return sum(e.duration if isinstance(e, Delay) else e.shape.duration for e in elements)

def relative_phase(A, B):
    """Transverse phase of A relative to B (deg, in (-180, 180])."""
    return np.degrees(np.angle((A[0] + 1j * A[1]) / (B[0] + 1j * B[1])))

def unwrap_from_far(phi, d):
    """Unwrap a phase (deg) on each side of d = 0, starting far from the pulse,
    where the phase is small, and moving toward it."""
    phi, d = np.asarray(phi, dtype=float), np.asarray(d, dtype=float)
    order = np.argsort(d)
    ph, ds = phi[order], d[order]
    res = ph.copy()
    neg, pos = ds < 0, ds >= 0
    res[neg] = np.degrees(np.unwrap(np.radians(ph[neg])))
    res[pos] = np.degrees(np.unwrap(np.radians(ph[pos][::-1])))[::-1]
    out = np.empty(len(phi))
    out[order] = res
    return out

def _pulses(elements):
    return [e for e in elements if not isinstance(e, Delay)]


## -- Bloch-Siegert phase
def bs_phase(elements, df_khz, excited_tol=0.01):
    """phi_BS (deg) of a sequence at the offsets `df_khz` (kHz from the carrier).

    The phase after `elements`, starting along x, minus the phase after a delay
    of the same length, unwrapped outward from the first pulse's frequency.
    Offsets that the sequence excites (Mz from z drops below 1 - excited_tol)
    are not a Bloch-Siegert phase: they are returned as NaN and flagged in
    `excited`."""
    els = list(elements)
    df = np.asarray(df_khz, dtype=float)
    first = _pulses(els)[0]
    phi = relative_phase(run(els, df), run([Delay(duration(els))], df))
    phi = unwrap_from_far(phi, df - getattr(first, "offset_khz", 0.0))
    excited = run(els, df, "z")[2] < 1.0 - excited_tol
    return dict(phi=np.where(excited, np.nan, phi), excited=excited)

def bs_theory(elements, df_khz, nucleus):
    """The two expressions for phi_BS (deg), summed over the pulses of `elements`:

    integral = sign * 360 * sum over t of (sqrt(D^2 + nu1(t)^2) - |D|) dt
    approx   = sign * 360 * sum over t of nu1(t)^2 dt / (2 |D|)

    with D the offset from each pulse and nu1 in kHz. Valid where no pulse
    excites; the caller masks the excited offsets."""
    df = np.asarray(df_khz, dtype=float)
    g = gyro_ratio(nucleus)
    integral, approx = np.zeros(len(df)), np.zeros(len(df))
    for P in _pulses(elements):
        nu1 = abs(g) * np.abs(P.calibrated_rf())        # kHz
        D = df - getattr(P, "offset_khz", 0.0)
        sgn = np.sign(D) * np.sign(g)
        Da = np.abs(D)[:, None]
        integral += sgn * 360 * np.sum((np.sqrt(Da ** 2 + nu1 ** 2) - Da) * P.shape.dt, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            approx += sgn * 360 * np.sum(nu1 ** 2 * P.shape.dt) / (2 * np.abs(D))

    return dict(integral=integral, approx=approx)


## -- compensation
def mirror_echo(pulse, df_khz, echo_half, refocus):
    """Method 1: the identical pulse at the mirror position of an echo,
    pulse - d - 180 - d - pulse, against the same echo with delays in place of
    the pulses. Returns the phase error (deg) with the first pulse alone (`one`)
    and with both pulses (`two`); `refocus` is the 180 deg pulse."""
    tau, d = pulse.shape.duration, echo_half
    ref = run([Delay(tau), Delay(d), refocus, Delay(d), Delay(tau)], df_khz)
    one = run([pulse, Delay(d), refocus, Delay(d), Delay(tau)], df_khz)
    two = run([pulse, Delay(d), refocus, Delay(d), pulse], df_khz)
    return dict(one=relative_phase(one, ref), two=relative_phase(two, ref))

def equivalent_delay(pulse, df_band_khz):
    """Methods 2 and 3: over a band, phi_BS of `pulse` is close to a constant
    plus a slope in offset. The slope is what a delay does, so lengthening the
    delay after the pulse by `t_eq` (ms) removes it; the constant `const` (deg)
    left at the band center is then removed by the phase of the next pulse or
    of the receiver.

    Returns t_eq, const, the free delay T (ms) used after the pulse, and the
    phase errors over the band without correction (`e_unc`), with the delay
    corrected (`e_dly`) and with both corrections (`e_fix`), all in deg."""
    df = np.asarray(df_band_khz, dtype=float)
    tau = pulse.shape.duration
    dfs = np.sort(df)
    phi = np.degrees(np.unwrap(np.radians(relative_phase(run([pulse], df), run([Delay(tau)], df))[np.argsort(df)])))
    dc = 0.5 * (dfs[0] + dfs[-1])
    slope = np.polyfit(dfs - dc, phi, 1)[0]         # deg per kHz
    free = run([Delay(1.0)], dfs)
    per_ms = np.polyfit(dfs, np.degrees(np.unwrap(np.angle(free[0] + 1j * free[1]))), 1)[0]
    t_eq = slope / per_ms                           # ms the pulse act like
    T = max(0.12, 2 * abs(t_eq) + 0.05)
    ref = run([Delay(tau), Delay(T)], df)
    e_unc = relative_phase(run([pulse, Delay(T)], df), ref)
    e_dly = relative_phase(run([pulse, Delay(T - t_eq)], df), ref)
    const = float(e_dly[np.argmin(np.abs(df - dc))])
    e_fix = (e_dly - const + 180.0) % 360.0 - 180.0
    return dict(t_eq=t_eq, const=const, T=T, e_unc=e_unc, e_dly=e_dly, e_fix=e_fix)