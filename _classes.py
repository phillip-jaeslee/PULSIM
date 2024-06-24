import itertools
import numbers

import numpy as np

from firstorder import first_order_spin_system, multiplet
from mat import reduce_peaks, add_lorentzians
from spin_system import qm_spinsystem
from _utils import low_high

from _descriptors import Number, Couplings

import torch

class Multiplet:
    v = Number() # fequency of the center of the multiplet
    I = Number() # total intensity
    J = Couplings() # 2D array-like

    def __init__(self, v, I, J, w=0.5):
        self.v = v
        self.I = I
        self.J = J
        self.w = w
        self._peaklist = multiplet((v, I), J)
    
    def __eq__(self, other):
        if hasattr(other, "peaklist") and callable(other.peaklist):
            return np.allclose(self.peaklist(), other.peaklist())


class SpinSystem:
    def __init__(self, v, J, w=0.5, second_order=True):
        self._nuclei_number = len(v)
        self.v = v
        self.J = J
        self.w = w
        self._second_order = second_order
        self._peaklist = self.peaklist()

    @property
    def v(self):
        return self._v
    
    @v.setter
    def v(self, vlist):
        if len(vlist) != self._nuclei_number:
            raise ValueError("v length must match J shape.")
        if not isinstance(vlist[0], numbers.Real):
            raise TypeError("v must be an array of numbers.")
        self._v = vlist
    
    @property
    def J(self):
        return self._J
    
    @J.setter
    def J(self, J_array):
        J = torch.tensor(J_array)
        m, n = J.shape
        if (m != n) or (m != self._nuclei_number):
            raise TypeError("J dimensions don't match v length.")
        if not torch.allclose(J, J.T):
            raise ValueError("J must be diagonal symmetric")
        for i in range(m):
            if J[i, i] != 0:
                raise ValueError("Diagonal elements of J must be 0.")
        self._J = J_array
    
    @property
    def second_order(self):
        return self._second_order
    
    @second_order.setter
    def second_order(self, boolean):
        if isinstance(boolean, bool):
            self._second_order = boolean
        else:
            raise TypeError("second_order must be a boolean.")
    
    def peaklist(self):
        if self._second_order:
            return qm_spinsystem(self._v, self._J)
        else:
            return first_order_spin_system(self._v, self._J)

    def __eq__(self,other):
        if hasattr(other, "peaklist") and callable(other.peaklist):
            return torch.allclose(self.peaklist(), other.peaklist())
    
    def __add__(self,other):
        if hasattr(other, "peaklist") and callable(other.peaklist):
            return Spectrum([self, other])
        else:
            return NotImplemented

class Spectrum:
    def __init__(self, components, vmin=None, v_max=None):
        combo = [extract_components(c) for c in components]
        result = list(itertools.chain.from_iterable(combo))
        self._components = result
        peaklists = [c.peaklist() for c in self._components]
        peaklists_merged = itertools.chain.from_iterable(peaklists)
        self._peaklist = sorted(reduce_peaks(peaklists_merged))
        if vmin is None:
            self._reset_vmin()
        else:
            self.vmin = vmin
        if vmax is None:
            self._reset_vmax()
        else:
            self.vmax = v_max
    
    def _add_peaklist(self, other):
        self._peaklist = sorted(reduce_peaks(itertoos.chain(self._peaklist, other.peaklist())))
        self._reset_minmax()
    
    def _reset_minmax(self):
        self._reset_vmin()
        self._reset_vmax()
    
    def _reset_vmin(self):
        self.vmin = min(self._peaklist)[0] - 50
    
    def _reset_vmax(self):
        self.vmax = max(self._peaklist)[0] + 50
    
    def default_limits(self):
        self._reset_minmax()
        return self.vmin, self.vmax
    
    def __eq__(self, other):
        if hasattr(other, "peaklist") and callable(other.peaklist):
            return torch.allclose(self._peaklist(), other.peaklist())
    
    def __add__(self, other):
        new_spectrum = Spectrum(self._components[:], vmin=self.vmin, vmax=self.vmax)
        new_spectrum += other
        return new_spectrum

    def __iadd__(self, other):
        if hasattr(other, "peaklist") and callable(other.peaklist):
            if isinstance(other, Spectrum):
                for component in other._components:
                    self.__iadd__(component)
            else:
                self._add_peaklist(other)
                self._components.append(other)
            return self
        else:
            raise TypeError("Item being added to Spectrum object not compatible")
        
    def peaklist(self):
        return self._peaklist
    
    def lineshape(self, points=800):
        vmin, vmax = low_high((self.vmin, self.vmax))
        x = torch.linspace(vmin, vmax, points)
        y = [add_lorentzians(x, c.peaklist(), c.w) for c in self._components]
        y_sum = torch.sum(y, 0)
        return x, y_sum
    
def extract_components(nmr_object, clist=None):
    if clist is None:
        clist = []
    if isinstance(nmr_object, Spectrum):
        for c in nmr_object._components:
            extract_components(c, clist)
    else:
        clist.append(nmr_object)
    return clist