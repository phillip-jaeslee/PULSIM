"""
metrics.py -- figures of merit for a calibrated pulse.

Diagnostics only. Nothing here feeds back into propagation: a pulse's RF
amplitude is decided by its calibration strategy (calibration.py), and these
functions measure what that amplitude achieves.

For adiabatic inversion the only meaningful figure of merit is how completely
Mz is inverted. Transverse phase is deliberately scrambled by the sweep and its
not a target, so an operator fidelity would be measuring something the pulse
never promised. Excitation and refocusing need a different meausre.
"""

import numpy as np

from .pulse_oo import Pulse

def inversion_fidelity(pulse, offsets, b1_scales=(1.0,)):
    """F(offset, B1 scale) = (1 - Mz_final) / 2, starting from M0 = +z.

    1.0 is perfect inversion, 0.5 complete saturation, 0.0 no effect.

    offsets     : kHz
    b1_scales   : multipliers on the pulse's nominal nu1_max. 1.0 is nominal.
                  The spread stands for calibration error, probe B1
                  inhomogeniety and sample loading. It scales the amplitude
                  only -- the shape and its frequency sweep are untouched.

    returns     : array of shape (len(b1_scales), len(offsets))  
    """
    offsets = np.asarray(offsets, dtype=float)
    nominal = pulse.nu1_max

    out = np.empty((len(b1_scales), offsets.size))
    for i, scale in enumerate(b1_scales):
        scaled = Pulse(pulse.shape, axis=pulse.axis, backend=pulse.backend, nu1_max=scale * nominal)

        M = np.zeros((3, offsets.size))
        M[2] = 1.0
        out[i] = (1.0 - scaled.apply(M, offsets)[2]) / 2.0
    return out

def realized_q(pulse, b1_scales=(1.0,)):
    """ Adiabaticity at each B1 scale.

    Q goes as nu1 squared, so Q(s) = s^2 * Q(1). Plotting this alongside
    inversion_fidelity on the same axes is the point of lesson L4: the region
    where Q is about 5 is the region where inversion holds.
    """
    if pulse.shape.calibration_mode != "adiabatic":
        raise ValueError(f"Q is only defined for an adiabatic pulse; {type(pulse.shape).__name__} calibrates from the pulse area.")
    q_nominal = pulse.realized_q
    return np.asarray([s * s * q_nominal for s in b1_scales], dtype=float)

def fraction_above(fidelity, threshold=0.99):
    """Area fraction of an (offset, B1) map meeting a fidelity threshold.

    One number, for ranking pulses in a table.
    """
    return float(np.mean(np.asarray(fidelity) >= threshold))