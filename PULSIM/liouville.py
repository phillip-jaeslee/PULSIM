"""
liouville.py -- Liouville-space (density-matrix) pulse sequence propagation.

A Segment is one piece of a pulse sequence (a shaped pulse, a delay, an ideal pulse).
Every Segment knows how to hand back the Hamiltonian(s) that act during it; 
LiouvilleSequence does the acutal sigma -> exp(-iHt) sigma exp(iHt) propagation once, in on place,
for every segment uniformly -- not re-implemented per segment type.
"""

from abc import ABC, abstractmethod

import numpy as np
import warnings
from .spin_operators import *
from .relaxation import Relaxation
from scipy.linalg import expm

_AXIS_PHASE = {"x": 0.0, "y": np.pi / 2, "-x": np.pi, "-y": -np.pi / 2}

def _axis_phase(axis):
    """RF phase in radian for a Pulse.axis.

    Pulse.axis is either one of the four cardinal labels or a phase in radians
    already -- phase cycling produces the latter, so a dict lookup alone raises
    KeyError on every non-cardinal phase
    """
    if isinstance(axis, str):
        try:
            return _AXIS_PHASE[axis]
        except KeyError:
            raise ValueError(
                f"axis must be one of {sorted(_AXIS_PHASE)} or a phase in "
                f"radians; got {axis!r}"
            ) from None
        
    return float(axis)

def _static_hamiltonian(spin_system, include_offset=True):
    """J-coupling (+ offset) terms: everything that does not change from one
    RF sample to the next.

    Built once per segment instead of once per timestep. The numbers are
    identical -- produce_operator() and embed() do not depend on the RF -- but
    rebuilding them per sample cost more than the propagation itself.
    """
    n_spins = spin_system.n_spins
    dim = 2 ** n_spins
    H = np.zeros((dim, dim), dtype=complex)
    for (i, j), J in spin_system.couplings.items():
        H = H + np.pi * (J / 1000) * product_operator(Iz(), i, Iz(), j, n_spins)
    if include_offset:
        for i, off in enumerate(spin_system.offsets):
            H  = H + off * embed(Iz(), i, n_spins)

    return H

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
        yield _static_hamiltonian(spin_system, self.include_offset), self.duration

class IdealPulse(Segment):
    """A delta-function pulse: RF so strong that offset and J-coupling do not
    evolve during it.

    flip     : rotation angle in radians.
    phase    : RF phase. Either radians, or one of "x", "y", "-x", "-y" --
               normalised here so that ideal and shaped pulses take the same
               spellings (ShapePulseSegment reads Pulse.axis the same way).
    duration : cancels analytically and has no effect on the result. The
               Hamiltonian is w1 = flip/duration and the step is dt = duration,
               so the propagator is exp(-i*flip*I) whatever value is used --
               verified identical from 1e-3 down to 1e-12 ms. It exists only to
               give the segment a nominal length on a sequence diagram; do not
               tune it expecting the physics to change.
    channel  : nucleus label (or raw gamma) to irradiate. The default None
               means EVERY spin in the system, which is what a homonuclear
               non-selective pulse wants but is almost never intended in a
               heteronuclear system -- a bare IdealPulse(pi/2) there flips the
               carbons along with the protons. LiouvilleSequence warns when it
               sees this combination. Pass channel explicitly to be sure.

    The Hamiltonian deliberately omits the offset and coupling terms that
    Delay and the shaped segments include: adding them would make the pulse
    non-ideal (at a realistic 10 us duration it shifts the propagator by
    ~6e-3), which is the whole distinction this class exists to draw.
    """

    def __init__(self, flip, phase=0.0, duration=1e-6, channel=None):
        self.flip = flip
        self.phase = _axis_phase(phase)
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
        RF      = self.pulse.calibrated_rf() * np.exp(1j * _axis_phase(self.pulse.axis))
        dt      = self.pulse.shape.dt
        Gamma   = self.pulse.Gamma
        targets = spin_system.channel(Gamma)

        H0 = _static_hamiltonian(spin_system)
        SX = sum(embed(Ix(), i, n_spins) for i in targets)
        SY = sum(embed(Iy(), i, n_spins) for i in targets)

        for w in RF:
            yield H0 + 2 * np.pi * Gamma * (w.real * SX + w.imag * SY), dt

class LiouvilleSequence:
    """An ordered list of Segments, propagated in Liouville space."""

    def __init__(self, segments, spin_system, relaxation=None):
        self.segments = segments
        self.spin_system = spin_system
        self.relaxation = relaxation
        self._warn_unchanneled_ideal_pulses()

    def _warn_unchanneled_ideal_pulses(self):
        """An IdealPulse with channel=None irradiates every spin. Harmless in a
        homonuclear system; in a heteronuclear one it is almost always a missing
        channel= argument, and the sequence still runs and returns plausible
        numbers, so nothing else catches it. Checked here rather than in
        IdealPulse.__init__, which never sees the SpinSystem."""
        if len(set(self.spin_system.nuclei)) < 2:
            return
        n = sum(1 for seg in self.segments if isinstance(seg, IdealPulse) and seg.channel is None)

        if n:
            warnings.warn(
                f"{n} IdealPulse segment(s) have channel=None, so they irradiate every spin in this"
                f" heteronuclearsystem ({', '.join(self.spin_system.nuclei)}). Pass channel= to target one nucleus.", 
                stacklevel=2,
            )

    def propagate(self, sigma0):
        """Propagate sigma0 through every segment in order."""
        relax = self.relaxation
        sigma = sigma0
        if relax is None or relax.is_identity():
            for segment in self.segments:
                for H, dt in segment.hamiltonians(self.spin_system):
                    U = expm(-1j * H * dt)
                    sigma = U @ sigma @ U.conj().T
            return sigma
        n_spins = self.spin_system.n_spins
        for segment in self.segments:
            for H, dt in segment.hamiltonians(self.spin_system):
                # Relaxation and coherent evolution do not commute once spins
                # are coupled, so a long step (a Delay is a single step however
                # long it is) must be subdivided. H is constant across the
                # substeps, so it is exponentiated once.
                n_sub = 1 if relax.max_step is None else max(1, int(np.ceil(dt / relax.max_step)))
                sub = dt / n_sub
                U = expm(-1j * H * sub)
                half = 0.5 * sub
                for _ in range(n_sub):
                    sigma = relax.apply(sigma, half, n_spins)
                    sigma = U @ sigma @ U.conj().T
                    sigma = relax.apply(sigma, half, n_spins)

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
        targets = spin_system.channel(self.channel)

        H0 = _static_hamiltonian(spin_system)
        SX = sum(embed(Ix(), i, n_spins) for i in targets)
        SY = sum(embed(Iy(), i, n_spins) for i in targets)

        for w in self.rf:
            yield H0 + w.real * SX + w.imag * SY, self.dt