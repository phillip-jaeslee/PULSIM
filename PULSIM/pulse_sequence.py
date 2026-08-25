"""
pulse_sequence.py — PulseSequence: chain several Pulses and run them in order.

Replaces the RF_temp / t_max_temp / Ns arrays and the np.append chains built
by hand in every test/RF_pulse_*.py script. A PulseSequence owns its own
concatenated .rf, .phase, and .time -- built once, correctly, here -- instead
of every driver script re-deriving them.
"""

import numpy as np

class PulseSequence:
    """An ordered list of Pulses, applied one after another to the same M."""

    def __init__(self, pulses=None):
        self.pulses = list(pulses) if pulses is not None else []

    def append(self, pulse):
        self.pulses.append(pulse)
        return self

    def __len__(self):
        return len(self.pulses)

    def __iter__(self):
        return iter(self.pulses)

    def __getitem__(self, i):
        return self.pulses[i]

    def run(self, M, df):
        """Apply every pulse in order. Returns the final (3, n_offsets) M."""
        for pulse in self.pulses:
            M = pulse.apply(M, df)
        return M

    @property
    def rf(self):
        """Concatenated calibrated RF envelope across every pulse, in order."""
        return np.concatenate([p.calibrated_rf() for p in self.pulses])

    @property
    def phase(self):
        """Concatenated RF phase (degrees), across every pulse, in order."""
        return np.concatenate([p.shape.phase_profile for p in self.pulses])

    @property
    def time(self):
        """
        Concatenated time axis across every pulse, each one picking up where
        the last left off -- replaces the by-hand `+ t_max_temp[0]` chains.
        """
        pieces = []
        t0 = 0.0
        for p in self.pulses:
            n = p.shape.points
            t_local = np.arange(0, n, 1) * p.shape.dt
            pieces.append(t_local + t0)
            t0 += p.shape.duration
        return np.concatenate(pieces)
    
