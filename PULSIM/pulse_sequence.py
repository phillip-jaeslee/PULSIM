"""
pulse_sequence.py — PulseSequence: chain several Pulses and run them in order.

Replaces the RF_temp / t_max_temp / Ns arrays and the np.append chains built
by hand in every test/RF_pulse_*.py script. A PulseSequence owns its own
concatenated .rf, .phase, and .time -- built once, correctly, here -- instead
of every driver script re-deriving them.

Between pulses it may hold a Delay -- the same liouville.Delay the density-
matrix simulator uses, so one Delay(1.79) means the same in both. In the Bloch
simulator a delay is free precession about z at each offset (plus relaxation,
if the pulses' backend has T1/T2), done exactly by bloch.bloch_delay.
"""

import numpy as np

from .bloch import bloch_delay
from .liouville import Delay


class PulseSequence:
    """An ordered list of Pulses (and Delays), applied one after another to the same M.

    delay_steps : frames a Delay is cut into when a trajectory is recorded, so
                  the precession can be drawn instead of jumping. Each step is
                  exact, so the result does not depend on it.
    """

    def __init__(self, pulses=None, delay_steps=50):
        self.pulses = list(pulses) if pulses is not None else []
        self.delay_steps = int(delay_steps)

    def append(self, pulse):
        self.pulses.append(pulse)
        return self

    def __len__(self):
        return len(self.pulses)

    def __iter__(self):
        return iter(self.pulses)

    def __getitem__(self, i):
        return self.pulses[i]

    #-- delays --
    def _relaxation(self):
        """T1, T2, M0 for the delays: those of the first pulse's backend (None,
        None, 1 if it has none), so a delay relaxes like the pulses around it."""
        for p in self.pulses:
            if not isinstance(p, Delay):
                b = p.backend
                return getattr(b, "T1", None), getattr(b, "T2", None), getattr(b, "M0", 1.0)
        return None, None, 1.0

    def _delay(self, M, delay, df, steps):
        """A Delay over 'steps' equal steps; returns the frame after every step."""
        df = np.asarray(df, dtype=float)
        df_eff = df if delay.include_offset else np.zeros_like(df)
        T1, T2, M0 = self._relaxation()
        dt = delay.duration / steps
        frames = []
        for _ in range(steps):
            # gamma cancels for free precession (Bz = df / gamma), so pass 1
            M = bloch_delay(M, dt, df_eff, 1.0, T1, T2, M0)
            frames.append(np.asarray(M, dtype=float).copy())
        return frames

    # -- running --
    def run(self, M, df, trajectory=False):
        """Apply every pulse and delay in order.

        Returns the final (3, n_offsets) M, or, with trajectory=True, an
        (n_steps + 1, 3, n_offsets) array spanning the whole sequence: one
        frame per RF step (delay_steps per Delay), plus the starting frame.
        Each element's own first frame repeats the previous one's last, so it
        is dropped -- the total is len(self.rf) + 1, which is what
        visualization.py expects against self.time.
        """
        if not trajectory:
            for el in self.pulses:
                if isinstance(el, Delay):
                    M = self._delay(M, el, df, 1)[-1]
                else:
                    M = el.apply(M, df)
            return M

        pieces = [np.asarray(M, dtype=float)[np.newaxis].copy()]    # the starting frame
        for el in self.pulses:
            if isinstance(el, Delay):
                traj = np.stack(self._delay(M, el, df, self.delay_steps))
                pieces.append(traj)
            else:
                traj = el.apply(M, df, trajectory=True)
                pieces.append(traj[1:])             # traj[0] repeats the previous frame
            M = traj[-1]
        return np.concatenate(pieces, axis=0)

    # -- traces for plotting ---
    @property
    def rf(self):
        """Concatenated calibrated RF envelope across every pulse, in order
        (zero during a Delay)."""
        if not self.pulses:
            return np.zeros(0, dtype=complex)
        return np.concatenate([np.zeros(self.delay_steps, dtype=complex)
                               if isinstance(el, Delay) else el.calibrated_rf()
                               for el in self.pulses])

    @property
    def phase(self):
        """Concatenated RF phase (degrees), across every pulse, in order (NaN
        during a Delay: there is no RF to have a phase)."""
        if not self.pulses:
            return np.zeros(0)
        return np.concatenate([np.full(self.delay_steps, np.nan)
                               if isinstance(el, Delay) else el.shape.phase_profile
                               for el in self.pulses])

    @property
    def time(self):
        """
        Concatenated time axis across every pulse and delay, each one picking
        up where the last left off -- replaces the by-hand `+ t_max_temp[0]`
        chains.
        """
        pieces = []
        t0 = 0.0
        for el in self.pulses:
            if isinstance(el, Delay):
                n, dt, duration = self.delay_steps, el.duration / self.delay_steps, el.duration
            else:
                n, dt, duration = el.shape.points, el.shape.dt, el.shape.duration
            pieces.append(np.arange(0, n, 1) * dt + t0)
            t0 += duration
        return np.concatenate(pieces) if pieces else np.zeros(0)
