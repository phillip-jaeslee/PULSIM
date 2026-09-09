"""
pulse_oo.py — Pulse: composition of an RFShape, a flip angle, and a Backend.

The single calibration entry point: it turns a normalized RFShape into a
physical RF field in mT, choosing the calibration strategy the shape
requires (see calibration.py), then steps a magnetization through it.
"""

import numpy as np

from PULSIM.backend import NumpyBackend
from PULSIM.calibration import AreaCalibration, signed_integral_of


class Pulse:
    """A single RF pulse: a shape, a target flip angle, and a rotation axis."""

    def __init__(self, shape, flip, axis="x", backend=None):
        self.shape = shape              # HAS-A RFShape
        self.flip = flip
        self.axis = axis
        self.backend = backend or NumpyBackend()   # HAS-A Backend

    @property
    def Gamma(self):
        """
        Read from the backend, not stored here separately. Calibration and
        rotation MUST share one Gamma, or the achieved flip angle drifts from
        the requested one -- see the 13C example above.
        """
        return self.backend.Gamma

    def calibrated_rf(self):
        """
        Scale the shapes' normalized envelope to a physical RF field im mT.

        Dispatches on how the shape can legitimately can calibrated:

        - Amplitude-modulated (fixed phase, possibly 0/180): the requested
          flip angle IS a rotation angle, and the amplitude follows from the
          pulse area. See AreaCalibration.
        
        - Phase-modulated: a flip angle is not defined. The Hamiltonian at
          different times do not commute, so the integral of the complex
          waveform is not a rotation angle and cannot be inverted for an
          amplitude. Raises rather than returning a plausible-looking wrong
          number. The Q/sweep-rate path for adiabatic shapes arrives in a
          later step.
        """
        envelope = self.shape.envelope()

        if np.any(envelope.imag != 0):
            raise ValueError(
                f"{type(self.shape).__name__} is phase-modulated, so a flip "
                f"angle is not defined for it: the RF phase varies during the "
                f"pulse, and the integral of a complex envelope is not a "
                f"rotation angle.\n"
                f"An adiabatic pulse is specified by its sweep and an "
                f"adiabaticity factor Q, not by a flip angle. Supply the RF "
                f"amplitude explicitly, or use the adiabatic calibration path."
            )

        cal = AreaCalibration(signed_integral_of(envelope))
        b1_max = cal.nu1_for(self.flip, self.shape.duration) / self.Gamma
        return envelope / np.abs(envelope).max() * b1_max

    def apply(self, M, df):
        """
        M  : (3, n_offsets) starting magnetization
        df : (n_offsets,) off-resonance frequencies, kHz
        returns the (3, n_offsets) magnetization after the full pulse
        """
        RF = self.calibrated_rf()
        dt = self.shape.dt
        for n in range(len(RF)):
            B = np.stack([
                np.full_like(df, np.real(RF[n])),
                np.full_like(df, np.imag(RF[n])),
                df / self.Gamma,
            ], axis=1)
            M = self.backend.rotate(M, dt, B, self.axis)
        return M