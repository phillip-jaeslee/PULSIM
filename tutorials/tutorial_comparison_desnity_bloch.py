"""
tutorial_comparison_density_bloch.py

Tutorial: compare a shaped pulse's excitation profile computed two
independent ways -- the classical Bloch-vector formalism (PULSIM.pulse_oo.
Pulse, backed by bloch_rotate_batch) and the quantum density-matrix /
Liouville-space formalism (PULSIM.liouville.ShapePulseSegment +
LiouvilleSequence) -- and plots Mx/My/Mz vs. offset frequency for both.

For a single, uncoupled spin-1/2, these two formalisms are mathematically
equivalent (the SU(2)/SO(3) isomorphism): rotating a classical Bloch vector
by the Bloch equation and propagating sigma -> U sigma U^dagger under the
matching spin Hamiltonian must give the same expectation values <Ix>,
<Iy>, <Iz> at every offset. This script makes that equivalence visible
rather than just asserting it in a test (see tests/test_liouville.py for
the pytest version of the same check, on a single hard pulse).

Units note: Pulse.apply's offset grid `df` is in kHz (matches Gamma in
kHz/mT). LiouvilleSequence's `offsets` argument is documented as rad/s, so
the same df must be passed in as 2*pi*df -- confirmed numerically in
tests/test_liouville.py (test_shape_pulse_segment_matches_classical_stack).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from PULSIM.rf_shape import RFShape
from PULSIM.backend import NumpyBackend, TorchBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.liouville import ShapePulseSegment, LiouvilleSequence
from PULSIM.spin_operators import Ix, Iy, Iz, embed
from PULSIM.spin_system import SpinSystem


Gamma       = 42.577478      # kHz/mT
duration    = 1           # ms
points      = 1000
flip        = np.pi / 2
axis        = "x"
BW          = 30.0           # kHz, offset sweep width
N_OFFSETS   = 101
file_path   = "wave/GaussCascadeQ5"

shape = RFShape.create("file", path=file_path, duration=duration)
pulse = Pulse(shape, flip, axis=axis, backend=TorchBackend(Gamma=Gamma))

df = np.linspace(-BW / 2, BW / 2, N_OFFSETS)

# -- classical: Bloch-vector propagation ------------------
M0 = np.zeros((3, len(df)))
M0[2, :] = 1.0              # define into Mz
M_bloch = pulse.apply(M0, df)

# -- quantum: density matrix / Liouville-space propagation --
def spin_expectation(sigma, spin_index, n_spins):
    """<Ix>, <Iy>, <Iz> of one spin in an n_spins density matrix, via the
    trace formula <A> = Tr(sigma A) / Tr(A A) -- generalizes the single-
    spin element-wise-ratio trick to any n_spins (verified to reduce to
    the same numbers when n_spins == 1)."""
    out = []
    for op in (Ix(), Iy(), Iz()):
        A = embed(op, spin_index, n_spins)
        out.append(np.trace(sigma @ A).real / np.trace(A @ A).real)
    return np.array(out)

segment = ShapePulseSegment(pulse)
M_liouville = np.zeros((3, len(df)))
for f, off in enumerate(df):
    ss = SpinSystem(nuclei=['H'], offsets=[2 * np.pi * off], couplings={})
    seq = LiouvilleSequence([segment], ss)
    sigma_final = seq.propagate(Iz())
    M_liouville[:, f] = spin_expectation(sigma_final, 0, 1)

# -- quantum: density matrix / Liouville-space propagation, two J-coupled spins --
J_HZ = 140.0   # real Hz -- Delay/ShapePulseSegment convert to PULSIM's ms time base internally
M_liouville_2 = np.zeros((3, len(df)))
for f, off in enumerate(df):
    ss2 = SpinSystem(nuclei=['H', '13C'], offsets=[2 * np.pi * off, 0.0], couplings={(0, 1): J_HZ})
    seq_2 = LiouvilleSequence([segment], ss2)
    sigma0_2 = embed(Iz(), 0, 2)   # spin 0's polarization only
    sigma_final2 = seq_2.propagate(sigma0_2)
    M_liouville_2[:, f] = spin_expectation(sigma_final2, 0, 2)
    
# -- plot --------------------------------------------------
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
labels = ["Mx", "My", "Mz"]
for i, ax in enumerate(axs):
    ax.plot(df, M_bloch[i], label="Bloch", linewidth=2)
    ax.plot(df, M_liouville[i], "--", label="Liouville", linewidth=2)
    #ax.plot(df, M_liouville_2[i], "--", label="2-spin Liouville", linewidth=2)
    ax.set_ylabel(labels[i])
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
axs[-1].set_xlabel("offset (kHz)")
axs[0].set_title(f"Excitation profile: {shape!r}, flip={flip:.3f} rad, axis={axis}")
plt.tight_layout()
os.mkdir("tutorial_figures")
plt.savefig("tutorial_figures/tutorial_comparison_density_bloch.png")
plt.show()