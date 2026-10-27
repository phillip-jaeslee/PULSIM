import re
import warnings
import numpy as np
from functools import reduce

def embed(op, spin_index, n_spins):
    factors = [np.eye(2)] * n_spins
    factors[spin_index] = op
    return reduce(np.kron, factors)

def Ix():
    return np.array([[0, 1/2] , [1/2, 0]])

def Iy():
    return np.array([[0, -1j/2], [1j/2, 0]])

def Iz():
    return np.array([[1/2, 0], [0, -1/2]])

def E():
    return np.array([[1, 0], [0, 1]])

def product_operator(op1, idx1, op2, idx2, n_spins):
    return 2 * embed(op1, idx1, n_spins) @ embed(op2, idx2, n_spins)


DEFAULT_LABELS = ['I', 'S', 'K', 'L', 'M', 'N']
_AXES = {'x': Ix, 'y': Iy, 'z': Iz}

class SpinOperators:
    """
    Named product-operator basis for a SpinSystem, addressed by string.

        ops = SpinOperators(spin_system)
        ops['Ix']       # embed(Ix(), 0, n) -- x operator on spin 0 ("I")
        ops['2IxSz']    # 2 * embed(Ix(),0,n) @ embed(Iz(),1,n) -- matches product_operator()
        ops['IxSz']     # same matrix; notes that it is calculated as '2IxSz'
        ops.expectation(sigma, '2IxSz')                  # Tr(sigma A)/Tr(A A)
        ops.readout(sigma, ['Ix','Iy','Iz','2IxSz','2IySz','2IzSz'])   # -> dict

    The leading coefficient is optional but, if given, must equal 2**(k-1)
    for a k-spin product (none or 1 for one spin, 2 for two, 4 for three).

    Labels default to I, S, K, L, M, N for spin 0, 1, 2, ...; pass
    labels=[...] explicitly for >6 spins or a different naming scheme.

    A k-spin product carries a leading factor of 2**(k-1) (Sorensen et al.,
    Prog. NMR Spectrosc. 16, 163 (1983)), so product_operator()'s existing
    "2 *" for two-spin terms is just the k=2 case.
    """
        
    def __init__(self, spin_system, labels=None):
        self.spin_system = spin_system
        self.n = spin_system.n_spins
        if labels is None:
            if self.n > len(DEFAULT_LABELS):
                raise ValueError(
                    f"no default label set for a {self.n}-spin system; pass labels=[...] explicitly"
                )
            labels = DEFAULT_LABELS[:self.n]
        if len(labels) != self.n:
            raise ValueError(f"labels has {len(labels)} entries, spin_system has {self.n} spins")
        if len(set(labels)) != len(labels):
            raise ValueError(f"labels must be unique, got {labels}")
        self.labels = list(labels)
        self._label_to_index = {lab: i for i, lab in enumerate(self.labels)} # print: {'I': 0, 'S': 1}
        self._cache = {}

    @staticmethod
    def _split_coefficient(name):
        """'2IxSz' -> (2, 'IxSz'); 'IxSz' -> (None, 'IxSz')."""
        m = re.match(r'\d+', name)
        if m is None:
            return None, name
        return int(m.group()), name[m.end():]

    def _parse(self, name):
        """'IxSz' -> [(0, 'x'), (1, 'z')]."""
        terms = []
        i = 0
        labels_by_len = sorted(self._label_to_index, key=len, reverse=True)
        while i < len(name):
            match = next((lab for lab in labels_by_len if name.startswith(lab, i)), None)
            if match is None:
                raise ValueError(f"{name!r}: no spin label recognized at position {i}")
            i += len(match)
            if i >= len(name) or name[i] not in _AXES:
                raise ValueError(f"{name!r}: expected one of x/y/z right after {match!r}")
            axis = name[i]
            i += 1
            idx = self._label_to_index[match]
            if any(idx == t[0] for t in terms):
                raise ValueError(f"{name!r}: spin label {match!r} appears more than once")
            terms.append((idx, axis))
        if not terms:
            raise ValueError(f"{name!r}: empty operator name")
        return terms

    def __getitem__(self, name):
        if name not in self._cache:
            coef, body = self._split_coefficient(name)
            terms = self._parse(body)
            factor = 2 ** (len(terms) - 1)
            if coef is None:
                if factor > 1:
                    warnings.warn(
                        f"{name!r} is calculated as '{factor}{body}' "
                        f"(a {len(terms)}-spin product carries a factor of {factor})",
                        stacklevel=2,
                    )
            elif coef != factor:
                raise ValueError(
                    f"{name!r}: a {len(terms)}-spin product has coefficient {factor}, got {coef}"
                )
            op = None
            for idx, axis in terms:
                single = embed(_AXES[axis](), idx, self.n)
                op = single if op is None else op @ single
            self._cache[name] = factor * op
        return self._cache[name]

    # '|Ixy|', '|Sxy|', ...: the transverse magnitude of one spin, sqrt(<Ix>^2 + <Iy>^2)
    _MAGNITUDE = re.compile(r"\|([A-Za-z]+'?)xy\|")

    def expectation(self, sigma, name):
        """<A> = Tr(sigma A) / Tr (A A). A name like '|Ixy|' gives the transverse
        magnitude of that spin, sqrt(<Ix>^2 + <Iy>^2)"""
        m = self._MAGNITUDE.fullmatch(name.replace(" ", ""))
        if m:
            label = m.group(1)
            return float(np.hypot(self.expectation(sigma, label + "x"),
                                  self.expectation(sigma, label + "y")))
        A = self[name]
        return np.trace(sigma @ A).real / np.trace(A @ A).real

    def readout(self, sigma, names):
        return {name: self.expectation(sigma, name) for name in names}
    
