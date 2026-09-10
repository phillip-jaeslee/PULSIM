"""
pulse_oo.py — Pulse: composition of an RFShape, a flip angle, and a Backend.

The single calibration entry point: it turns a normalized RFShape into a
physical RF field in mT, choosing the calibration strategy the shape
requires (see calibration.py), then steps a magnetization through it.
"""

import numpy as np
import warnings

from .backend import NumpyBackend

class Pulse:
    """A single RF pulse: a shape, a target flip angle, and a rotation axis."""

    def __init__(self, shape, flip=None, axis="x", backend=None, nu1_max=None):
        self.shape = shape              # HAS-A RFShape
        self.flip = flip
        self.axis = axis
        self.backend = backend or NumpyBackend()   # HAS-A Backend
        self._nu1_max = nu1_max

    @property
    def Gamma(self):
        """
        Read from the backend, not stored here separately. Calibration and
        rotation MUST share one Gamma, or the achieved flip angle drifts from
        the requested one -- see the 13C example above.
        """
        return self.backend.Gamma

    @property
    def nu1_max(self):
        """Peak nutation frequency actually used, kHz.

        Precedence: an explicit nu1_max wins; otherwise the shape's own
        calibration strategy decides -- signed pulse area for an
        amplitude-modulated shape, sweep rate and Q for an adiabatic one.
        """
        if self._nu1_max is not None:
            return float(self._nu1_max)

        if self.shape.calibration_mode == "area":
            if self.flip is None:
                raise ValueError(f"{type(self.shape).__name__} calibrates from the pulse area, "
                                 f"so it needs a flip angle. Pass flip= (radian), or nu1_max= (kHz) "
                                 f"to set the amplitude directly.")
            return self.shape.calibration.nu1_for(self.flip, self.shape.duration)

        calibration = self.shape.calibration
        self._warn_if_flip_is_meaningless()
        return calibration.nu1_for(self.shape.duration)

    @property
    def b1_max(self):
        """Peak B1 in mT -- derived from nu1_max, not a separate setting."""
        return self.nu1_max / self.Gamma

    @property
    def realized_q(self):
        """Adiabaticity actually achieved. None for non-adiabatic shapes."""
        if self.shape.calibration_mode != "adiabatic":
            return None
        return self.shape.calibration.q_for(self.nu1_max, self.shape.duration)

    def _warn_if_flip_is_meaningless(self):
        if self.flip is None:
            return
        q_mid = getattr(self.shape, "q_mid", None)
        detail = f" (q_mid={q_mid})" if q_mid is not None else ""
        warnings.warn(
            f"flip={self.flip:.4f} rad does not scale the RF amplitude for "
            f"{type(self.shape).__name__}, which is adiabatic: amplitude comes "
            f"from the sweep rate and Q{detail}. The flip angle is recorded as "
            f"the intended operation only.",
            UserWarning,
            stacklevel=3,
        )

    def calibrated_rf(self):
        """The shape's envelope scaled to a physical RF field in mT."""

        envelope = self.shape.envelope()
        return envelope / np.abs(envelope).max() * self.b1_max

    def _field(self, rf_n, df):
        """The (n_offsets, 3) [Bx, By, Bz] field seen during one RF step."""
        return np.stack([
            np.full_like(df, np.real(rf_n)),
            np.full_like(df, np.imag(rf_n)),
            df / self.Gamma,
        ], axis=1)

    def apply(self, M, df, trajectory=False):
        """
        M   : (3, n_offsets) starting magnetization
        df  : (n_offsets,) off-resonance frequencies, kHz

        trajectory=False (the default) returns the (3, n_offsets)
        magnetization after the full pulse

        trajectory=True returns the whole history instead, as an
        (n_steps + 1, 3, n_offsets) array: traj[0] is M exactly as handed in,
        traj[n] is M after the n-th RF step, and traj[-1] is precisely what
        the trajectory=False call would have returned. Every frame is a
        drop-in (3, n_offsets) M, which is what an animation walks through.
        apply() is the only place that knows how many steps a pulse takes,
        so it is the only place that can record them.
        """
        RF = self.calibrated_rf()
        dt = self.shape.dt

        if not trajectory:
            for n in range(len(RF)):
                M = self.backend.rotate(M, dt, self._field(RF[n], df), self.axis)
            return M

        M = np.asarray(M, dtype=float)
        traj = np.empty((len(RF) + 1,) + M.shape, dtype=float)
        traj[0] = M
        for n in range(len(RF)):
            M = self.backend.rotate(M, dt, self._field(RF[n], df), self.axis)
            traj[n + 1] = M
        return traj
