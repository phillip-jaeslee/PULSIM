"""
relaxation.py -- phenomenological T1/T2 relaxation for Liouville-space
propagation.

The Bloch side already relaxes (backend.py -> bloch_relax_rotate_batch). This
is the density-matrix counterpart, deliberately built to the same contract:

  * Strang-split, relax(dt/2) -> propagate(dt) -> relax(dt/2), so it is
    second-order accurate in the time step rather than exact;
  * with no Relaxation attached, propagation is BIT FOR BIT what it was before
    this module existed -- guaranteed by an early return, not by an identity
    factor that happens to be 1.0;
  * for a single uncoupled spin it reproduces the Bloch backend's answer,
    which is what tests/ asserts rather than assumes.

MODEL. Each Cartesian product operator decays at a rate built from its own
factors: 1/T2 for every transverse (x or y) factor, 1/T1 for every
longitudinal (z) factor, nothing for identity. So Ix decays at 1/T2, Iz at
1/T1, and 2IzSy at 1/T1(I) + 1/T2(S). Longitudinal terms relax toward thermal
equilibrium, sum_i M0_i Iz_i; everything else relaxes toward zero.

WHAT THIS IS NOT. There is no relaxation superoperator here, so there is no
cross-relaxation and no NOE: a rate is attached to each operator rather than
derived from a mechanism and a correlation time. That is the honest limit of a
phenomenological model, and it is the same limit the Bloch side has.
"""

import numpy as np

# Pauli-style single-spin factors. Orthogonal under Tr(A B), which is what
# makes the decomposition below a projection rather than a solve.
_E = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_FACTORS = {"e": _E, "x": _X, "y": _Y, "z": _Z}


def _as_per_spin(value, n_spins, name):
    """Accept one number for every spin, or one per spin."""
    if value is None or np.isscalar(value):
        return [value] * n_spins
    value = list(value)
    if len(value) != n_spins:
        raise ValueError(f"{name} has {len(value)} entries, the system has {n_spins} spins")
    return value


class Relaxation:
    """Per-spin T1/T2 times, in ms, and the equilibrium each spin returns to.

    T1, T2 : a single value applied to every spin, or one value per spin.
             None means infinite, i.e. no relaxation on that axis. Zero or
             negative raises -- zero is not shorthand for instantaneous.
    M0     : equilibrium Iz per spin (scalar or per spin), default 1.0.
    """

    def __init__(self, T1=None, T2=None, M0=1.0, max_step=0.05):
        """max_step : longest propagation step, in ms, that may be taken as a
        single Strang step. Anything longer is subdivided.

        This matters because relaxation and coherent evolution do NOT commute
        once there is a J-coupling: evolution moves amplitude between operators
        of different rate (Ix relaxes at 1/T2, the 2IySz it evolves into at
        1/T2 + 1/T1), so a Delay of tens of ms taken in one step is visibly
        wrong -- measured at ~9e-3 for a 30 ms delay with J = 140 Hz. Equal T1
        and T2 does not rescue it, because the multi-spin rate is a sum either
        way. Subdividing costs one extra matrix multiply and one relaxation
        step per substep; the propagator exponentiates H only once per segment
        step, not once per substep.
        """
        if max_step is not None and max_step <= 0:
            raise ValueError(f"max_step must be positive (ms) or None; got {max_step!r}")
        self.T1, self.T2, self.M0, self.max_step = T1, T2, M0, max_step
        self._cache = {}

    def _check(self, values, name):
        for v in values:
            if v is None:
                continue
            if v <= 0:
                raise ValueError(f"{name} must be positive (ms) or None; got {v!r}")

    def is_identity(self):
        """True when nothing relaxes, so the caller can take the exact path."""
        flat = [self.T1, self.T2]
        return all(v is None or (not np.isscalar(v) and all(x is None for x in v))
                   for v in flat)

    # ---------------------------------------------------------------- basis
    def _basis(self, n_spins):
        """(4**n, 2**n, 2**n) operator basis and the decay rate of each."""
        key = n_spins
        if key in self._cache:
            return self._cache[key]

        T1 = self._as_checked(self.T1, n_spins, "T1")
        T2 = self._as_checked(self.T2, n_spins, "T2")

        ops, rates, labels = [], [], []
        for idx in np.ndindex(*([4] * n_spins)):
            names = ["exyz"[k] for k in idx]
            mat = np.array([[1.0 + 0j]])
            rate = 0.0
            for i, nm in enumerate(names):
                mat = np.kron(mat, _FACTORS[nm])
                if nm in ("x", "y"):
                    rate += 0.0 if T2[i] is None else 1.0 / T2[i]
                elif nm == "z":
                    rate += 0.0 if T1[i] is None else 1.0 / T1[i]
            ops.append(mat)
            rates.append(rate)
            labels.append("".join(names))

        ops = np.stack(ops)
        rates = np.asarray(rates)

        # Equilibrium, expressed in the same basis: sum_i M0_i Iz_i.
        M0 = _as_per_spin(self.M0, n_spins, "M0")
        dim = 2 ** n_spins
        sigma_eq = np.zeros((dim, dim), dtype=complex)
        for i in range(n_spins):
            mat = np.array([[1.0 + 0j]])
            for j in range(n_spins):
                mat = np.kron(mat, _Z / 2 if j == i else _E)
            sigma_eq = sigma_eq + (1.0 if M0[i] is None else M0[i]) * mat
        c_eq = np.einsum("kij,ji->k", ops, sigma_eq) / dim

        self._cache[key] = (ops, rates, c_eq, dim)
        return self._cache[key]

    def _as_checked(self, value, n_spins, name):
        values = _as_per_spin(value, n_spins, name)
        self._check(values, name)
        return values

    # ----------------------------------------------------------------- step
    def apply(self, sigma, tau, n_spins):
        """Relax `sigma` for `tau` ms. Closed form -- no matrix exponential."""
        ops, rates, c_eq, dim = self._basis(n_spins)
        decay = np.exp(-rates * tau)
        c = np.einsum("kij,ji->k", ops, sigma) / dim
        c = c * decay + c_eq * (1.0 - decay)
        return np.einsum("k,kij->ij", c, ops)
