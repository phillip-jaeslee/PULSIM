"""
backend.py — the Backend layer.

Backend is a CONTRACT: "rotate a batch of magnetization vectors, one per
offset frequency, through one time step." NumpyBackend is the first, and
simplest, thing that fulfills it — internally it's just a loop over
bloch_rotate, the function you already fixed and proved correct in Steps 1-3.
A TorchBackend comes later and fulfills the exact same contract.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from .bloch import bloch_rotate, torch_bloch_rotate, bloch_rotate_batch

class Backend(ABC):
    """Declares what every backend must be able to do. Never instantiated directly."""

    @abstractmethod
    def rotate(self, M, dt, B, axis="x"):
        """
        M    : (3, n_offsets) array — one magnetization vector per column
        B    : (n_offsets, 3) array — one [Bx, By, Bz] per offset
        returns: (3, n_offsets) array, M after this time step
        """

class NumpyBackend(Backend):
    """The numpy implementation: one bloch_rotate call per offset, in a loop."""

    def __init__(self, Gamma):
        self.Gamma = Gamma

    def rotate(self, M, dt, B, axis="x"):
        return bloch_rotate_batch(M, dt, B, axis, self.Gamma)

class TorchBackend(Backend):
    """The torch implementation. torch_bloch_rotate already handles every
    offset in one call, so there's no loop here — just getting the arrays
    into the shape it expects and back out again."""

    def __init__(self, Gamma, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Gamma = Gamma

    def rotate(self, M, dt, B, axis="x"):
        # torch_bloch_rotate wants (n_offsets, 3), NumpyBackend's array's are (3, n_offsets)
        M_t = torch.as_tensor(M.T, dtype=torch.float32, device=self.device)
        B_t = torch.as_tensor(B, dtype=torch.float32, device=self.device)

        result = torch_bloch_rotate(M_t, dt, B_t, axis, self.Gamma)

        return result.cpu().numpy().T