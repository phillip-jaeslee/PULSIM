"""
pulse_oo.py — Pulse: composition of an RFShape, a flip angle, and a Backend.

Not yet used by anything. Once it's proven equivalent to pulse.py's classes,
a later step points the driver scripts at this instead and pulse.py's
duplication gets deleted.
"""

import numpy as np

from .backend import NumpyBackend


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
        Scale the shape's envelope so applying it produces `flip` radians of
        rotation on resonance. Replaces the
        `flip * RF_org / sum(RF_org) / (2*pi*Gamma*dt)` line that's currently
        copy-pasted six times across pulse.py and bloch_pulse_simulation.py.
        """
        envelope = self.shape.envelope()
        scale = self.flip / np.sum(envelope) / (2 * np.pi * self.Gamma * self.shape.dt)
        if self.shape.is_adiabatic:
            scale *= 2
        return envelope * scale

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