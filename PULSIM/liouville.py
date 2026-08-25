"""
liouville.py -- Liouville-space (density-matrix) pulse sequence propagation.

A Segment is one piece of a pulse sequence (a shaped pulse, a delay, an ideal pulse).
Every Segment knows how to hand back the Hamiltonian(s) that act during it; 
LiouvilleSequence does the acutal sigma -> exp(-iHt) sigma exp(iHt) propagation once, in on place,
for every segment uniformly -- not re-implemented per segment type.
"""

from abc import ABC, abstractmethod

import numpy as np
from .spin_operators import *
from scipy.linalg import expm

class Segment(ABC):
    """Once piece of a pulse sequence"""

    @abstractmethod
    def hamiltonians(self, spin_system):
        """
        Yield (H, dt) pairs describing this segment's evolution

        spin_system : a SpinSystem describing the offsets, couplings, and
                      per-spin gammas of every spin in play.
        """
        
class Delay(Segment):
    """A period with no RF, evolution under J-coupling and offset"""

    def __init__(self, duration, include_offset=True):
        self.duration = duration
        self.include_offset = include_offset

    def hamiltonians(self, spin_system):
        n_spins = spin_system.n_spins
        dim = 2 ** n_spins
        H = np.zeros((dim, dim), dtype=complex)
        for (i, j), J in spin_system.couplings.items():
            H = H + np.pi * (J / 1000.0) * product_operator(Iz(), i, Iz(), j, n_spins)
        if self.include_offset:
            for i, off in enumerate(spin_system.offsets):
                H = H + off * embed(Iz(), i, n_spins)
        yield H, self.duration

class IdealPulse(Segment):
    """A period with ideal RF """

    def __init__(self, flip, phase=0.0, duration=1e-6, channel=None):
        self.flip = flip
        self.phase = phase
        self.duration = duration
        self.channel = channel

    def hamiltonians(self, spin_system):
        n_spins = spin_system.n_spins
        dim = 2 ** n_spins
        H = np.zeros((dim, dim), dtype=complex)
        w1 = self.flip / self.duration
        targets = (spin_system.channel(self.channel) if self.channel is not None
                   else range(n_spins))
        for i in targets:
            H = H + w1 * (np.cos(self.phase) * embed(Ix(), i, n_spins) + np.sin(self.phase) * embed(Iy(), i, n_spins))
        yield H, self.duration


class ShapePulseSegment(Segment):
    """A period with shaped RF"""

    def __init__(self, pulse):
        self.pulse = pulse

    def hamiltonians(self, spin_system):
        n_spins = spin_system.n_spins
        dim = 2 ** n_spins
        RF      = self.pulse.calibrated_rf()
        dt      = self.pulse.shape.dt
        Gamma   = self.pulse.Gamma
        targets = spin_system.channel(Gamma)
        for n in range(len(RF)):
            H = np.zeros((dim, dim), dtype=complex)
            for (i, j), J in spin_system.couplings.items():
                H = H + np.pi * (J / 1000.0) * product_operator(Iz(), i, Iz(), j, n_spins)
            for i, off in enumerate(spin_system.offsets):
                H = H + off * embed(Iz(), i, n_spins)
            wx, wy = RF[n].real, RF[n].imag
            for i in targets:
                H = H + 2 * np.pi * Gamma * (wx * embed(Ix(), i, n_spins) + wy * embed(Iy(), i, n_spins))
            yield H, dt

class LiouvilleSequence:
    """An ordered list of Segments, propagated in Liouville space."""

    def __init__(self, segments, spin_system):
        self.segments = segments
        self.spin_system = spin_system

    def propagate(self, sigma0):
        sigma = sigma0
        for segment in self.segments:
            for H, dt in segment.hamiltonians(self.spin_system):
                U = expm(-1j * H * dt)
                sigma = U @ sigma @ U.conj().T
        return sigma

class RawShapedPulseSegment(Segment):
    """A period with a directly-specified RF trajectory, already in
    nutation-rate units (rad/ms) rather than calibrated to a target flip
    angle. ShapePulseSegment/Pulse.calibrated_rf() assume you want "this
    envelope scaled to deliver X radians of on-resonance rotation" -- but
    real hardware is sometimes better described as "this envelope at a
    fixed measured peak RF power (Hz)", which calibrated_rf()'s flip-based
    normalization can't represent (dividing a real flip by a complex-
    valued sum(envelope) doesn't reproduce a fixed physical power). This
    segment skips calibration entirely: whatever RF array you hand it is
    exactly what gets used, timestep by timestep."""

    def __init__(self, rf, dt, channel):
        self.rf = rf            # complex array: wx + i*wy, rad/ms, one value per sample
        self.dt = dt
        self.channel = channel  # nucleus label or raw gamma

    def hamiltonians(self, spin_system):
        n_spins = spin_system.n_spins
        dim = 2 ** n_spins
        targets = spin_system.channel(self.channel)
        for n in range(len(self.rf)):
            H = np.zeros((dim, dim), dtype=complex)
            for (i, j), J in spin_system.couplings.items():
                H = H + np.pi * (J / 1000.0) * product_operator(Iz(), i, Iz(), j, n_spins)
            for i, off in enumerate(spin_system.offsets):
                H = H + off * embed(Iz(), i, n_spins)
            wx, wy = self.rf[n].real, self.rf[n].imag
            for i in targets:
                H = H + wx * embed(Ix(), i, n_spins) + wy * embed(Iy(), i, n_spins)
            yield H, self.dt