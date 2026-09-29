"""
tutorial_comparison_density_bloch.py

Tutorial: compare a shaped pulse's excitation profile computed two
independent ways -- the classical Bloch-vector formalism (PULSIM.pulse_oo.
Pulse, backed by bloch_rotate_batch) and the quantum density-matrix
formalism (PULSIM.liouville.ShapePulseSegment + LiouvilleSequence) -- and
plots Mx/My/Mz vs. offset frequency for both.

For a single, uncoupled spin-1/2, these two formalisms are mathematically
equivalent (the SU(2)/SO(3) isomorphism): rotating a classical Bloch vector
by the Bloch equation and propagating sigma -> U sigma U^dagger under the
matching spin Hamiltonian must give the same expectation values <Ix>,
<Iy>, <Iz> at every offset. This script makes that equivalence visible
rather than just asserting it in a test (see tests/test_liouville.py for
the pytest version of the same check), and prints how closely they agree.

Units note: Pulse.apply's offset grid `df` is in kHz (matches Gamma in
kHz/mT). SpinSystem's `offsets` are angular frequencies in rad/ms, so the
same df is passed in as 2*pi*df.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.rf_shape import RFShape
from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.liouville import ShapePulseSegment, LiouvilleSequence
from PULSIM.spin_operators import Ix, Iy, Iz, embed
from PULSIM.spin_system import SpinSystem, gyro_ratio


Gamma       = gyro_ratio('H')   # kHz/mT
duration    = 1.0               # ms
points      = 1000
flip        = np.pi / 2
axis        = "x"
BW          = 30.0              # kHz, offset sweep width
N_OFFSETS   = 101

shape = RFShape.create("gausscasq5", duration=duration, points=points)
pulse = Pulse(shape, flip, axis=axis, backend=NumpyBackend(Gamma=Gamma))

df = np.linspace(-BW / 2, BW / 2, N_OFFSETS)

# -- classical: Bloch-vector propagation ------------------
M0 = np.zeros((3, len(df)))
M0[2, :] = 1.0              # start along +z
M_bloch = pulse.apply(M0, df)

# -- quantum: density-matrix propagation ------------------
def spin_expectation(sigma, spin_index, n_spins):
    """<Ix>, <Iy>, <Iz> of one spin in an n_spins density matrix, via the
    trace formula <A> = Tr(sigma A) / Tr(A A)."""
    out = []
    for op in (Ix(), Iy(), Iz()):
        A = embed(op, spin_index, n_spins)
        out.append(np.trace(sigma @ A).real / np.trace(A @ A).real)
    return np.array(out)

segment = ShapePulseSegment(pulse)
M_density = np.zeros((3, len(df)))
for f, off in enumerate(df):
    ss = SpinSystem(nuclei=['H'], offsets=[2 * np.pi * off], couplings={})
    sigma_final = LiouvilleSequence([segment], ss).propagate(Iz())
    M_density[:, f] = spin_expectation(sigma_final, 0, 1)

print(f"max |Bloch - density matrix| over {N_OFFSETS} offsets: "
      f"{np.abs(M_bloch - M_density).max():.1e}")

# -- plot --------------------------------------------------
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
labels = ["Mx", "My", "Mz"]
for i, ax in enumerate(axs):
    ax.plot(df, M_bloch[i], label="Bloch", linewidth=2)
    ax.plot(df, M_density[i], "--", label="density matrix", linewidth=2)
    ax.set_ylabel(labels[i])
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
axs[-1].set_xlabel("offset (kHz)")
axs[0].set_title(f"Excitation profile: 90x GaussCascadeQ5, {duration:g} ms")
plt.tight_layout()
os.makedirs("tutorial_figures", exist_ok=True)
plt.savefig("tutorial_figures/tutorial_comparison_density_bloch.png")
plt.show()