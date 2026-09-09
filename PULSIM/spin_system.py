"""
spin_system.py -- SpinSystem: a bundle of per-spin properties for
Liouville-space simulation.

Spins are identified by nucleus label ('H', '13C', ...) rather than a raw
gamma value -- gyro_ratio() is the single source of truth for what gamma
each supported nucleus has, so it's never re-typed (and never drifts, the
way GAMMA_H = 42.577478 vs 42.577478518 could) across tests/tutorials/call
sites.

SpinSystem.channel() is dual-mode: pass a nucleus label to look up its
gamma, or pass a raw gamma value directly (e.g. a Pulse's own calibration
Gamma, as ShapePulseSegment does -- Pulse/Backend still take a raw Gamma,
that part is unchanged).
"""

GYRO_RATIOS = {
    'H':   42.577478518,
    'D':    6.536,
    'T':   45.415,
    '13C': 10.7084,
    '15N': -4.316,
    '17O': -5.772,
    '19F': 40.078,
    '31P': 17.235,
    '35Cl': 4.176,
}


def gyro_ratio(nucleus):
    """Reduced gyromagnetic ratio gamma_bar = gamma/(2*pi), in kHz/mT.

    NOT gamma itself (rad/s/T). PHYSICS_SPECIFICATION.md section 1.3 requires
    that the two never share a name: gamma_bar is cyclic frequency per field,
    gamma is angular frequency per field. Values are signed -- see 15N, 17O --
    and the sign must not be removed.
    """
    try:
        return GYRO_RATIOS[nucleus]
    except KeyError:
        raise ValueError(f"{nucleus!r} may not be supported for the system.") from None


class SpinSystem:
    """
    Static description of a set of coupled spins.

    nuclei    : list of nucleus labels, one per spin (e.g. ['H', '13C']).
                Resolved to gyromagnetic ratios via gyro_ratio(), which
                also doubles as each spin's RF-channel identity -- spins
                sharing a nucleus get irradiated together by a pulse
                calibrated for that nucleus's gamma.
    offsets   : list of offset frequencies (rad/ms), one per spin.
                PULSIM's time base is ms, os an offset of f kHz is supplied
                as 2*pif -- see tests/test_louville.py.
    couplings : dict mapping (i, j) spin-index pairs -> J coupling (Hz).
                Converted internally from Hz to PULSIM's ms time base by
                Delay/ShapePulseSegment.
    """

    def __init__(self, nuclei, offsets, couplings=None):
        nuclei = list(nuclei)
        offsets = list(offsets)
        if len(nuclei) != len(offsets):
            raise ValueError(
                f"nuclei and offsets must have one entry per spin "
                f"(got {len(nuclei)} nuclei, {len(offsets)} offsets)"
            )
        if len(nuclei) == 0:
            raise ValueError("a SpinSystem needs at least one spin")
        self.nuclei = nuclei
        self.gammas = [gyro_ratio(n) for n in nuclei]
        self.offsets = offsets
        self.couplings = dict(couplings) if couplings else {}
        self.n_spins = len(nuclei)

    def channel(self, key, tol=1e-6):
        """
        Indices of every spin matching `key` -- pass a nucleus label
        ('H', '13C', ...) or a raw gamma value directly. What a
        channel-selective pulse iterates over to decide which spins it
        actually touches.
        """
        gamma = gyro_ratio(key) if isinstance(key, str) else key
        idx = [i for i, g in enumerate(self.gammas) if abs(g - gamma) < tol]
        if not idx:
            raise ValueError(f"no spin in this SpinSystem matches {key!r}")
        return idx

    def __repr__(self):
        return (f"SpinSystem(nuclei={self.nuclei}, offsets={self.offsets}, "
                f"couplings={self.couplings})")