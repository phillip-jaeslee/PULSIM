"""
PULSIM -- Bloch-equation and Liouville-space NMR pulse simulation.

This __init__.py re-exports the main entry points so `from PULSIM import X`
works for everyday use. Anything not listed here is still reachable via its
submodule directly, e.g. `from PULSIM.mat_operator import cpu_rot_batch`.
"""

from .rf_shape import RFShape
from .backend import Backend, NumpyBackend, TorchBackend
from .pulse_oo import Pulse
from .pulse_sequence import PulseSequence
from .bloch import bloch_rotate, bloch_rotate_batch, torch_bloch_rotate, bloch_relax, bloch_relax_batch
from .liouville import Segment, Delay, IdealPulse, ShapePulseSegment, LiouvilleSequence
from .parallel import parallel_map
from .simulate import sim_hard_pulse, sim_import_shaped_pulse, sim_shaped_pulse, sim_own_shaped_pulse
from .spin_system import SpinSystem
from .metrics import inversion_fidelity, realized_q, fraction_above

__all__ = [
    "RFShape",
    "Backend", "NumpyBackend", "TorchBackend",
    "Pulse",
    "PulseSequence",
    "bloch_rotate", "bloch_rotate_batch", "torch_bloch_rotate", "bloch_relax", "bloch_relax_batch",
    "Segment", "Delay", "IdealPulse", "ShapePulseSegment", "LiouvilleSequence",
    "parallel_map",
    "sim_hard_pulse", "sim_import_shaped_pulse", "sim_shaped_pulse", "sim_own_shaped_pulse",
    "SpinSystem",
    "inversion_fidelity", "realized_q", "fraction_above",
]