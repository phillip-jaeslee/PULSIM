"""
tutorial_dipolar_decoupling.py -- continuous-wave heteronuclear dipolar
decoupling, as used in the earliest CP-MAS solid-state NMR experiments
(Pines, Gibby & Waugh, "Proton-enhanced NMR of dilute spins in solids",
J. Chem. Phys. 59, 569 (1973)).

The secular (truncated) heteronuclear dipolar Hamiltonian has exactly the
same Iz(I)*Iz(S) form as a scalar J-coupling -- just much larger in a
rigid solid (no motional averaging) and orientation-dependent. This
script reuses PULSIM.liouville's `couplings` mechanism directly to
represent a static 1-bond C-H dipolar coupling (D ~ 20 kHz), and shows
the defining signature of CW decoupling: strong continuous RF on the I
(1H) channel, applied throughout the S-spin's evolution, suppresses the
S-spin's dephasing into antiphase coherence -- the quantum-mechanical
version of the classical "motional averaging" argument (fast I-spin
nutation time-averages <Iz(I)> toward zero, so S sees an averaged-out
coupling).

Verified numerically before writing this file: with NO decoupling RF, an
S-spin transverse coherence (Ix(S)) fully dephases into antiphase
coherence at T = 1/(2D) (retained in-phase fraction -> 0). Sweeping the
I-channel nutation rate w1 at fixed T shows the retained fraction climb
monotonically: ~0 at w1=0, ~0.61 at w1=D, ~0.96-1.0 once w1 is 5-15x D --
the textbook "RF power must exceed the coupling to decouple it" behavior.
"""

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.spin_operators import Ix, embed
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, ShapePulseSegment, LiouvilleSequence
from PULSIM.rf_shape import RFShape
from PULSIM.backend import NumpyBackend
from PULSIM.pulse_oo import Pulse

PI = np.pi
D_HZ = 20000.0                 # static 1-bond C-H dipolar coupling, Hz-scale
D_internal = D_HZ / 1000.0     # PULSIM's ms-based internal rate
T = 1.0 / (2 * D_internal)     # quarter-period point: fully dephased with no decoupling


def Ix2_retained(sigma):
    """Fraction of the original S-spin in-phase coherence (Ix(S)) still
    present, via the trace formula <A> = Tr(sigma A) / Tr(A A)."""
    A = embed(Ix(), 1, 2)
    return np.trace(sigma @ A).real / np.trace(A @ A).real


def run_decoupling(w1_hz, T=T, D_HZ=D_HZ):
    """S-spin starts as pure Ix(S) (as if just excited); evolves for T
    under the dipolar coupling while the I channel is continuously
    irradiated at nutation rate w1_hz. w1_hz = 0 means no decoupling."""
    ss = SpinSystem(nuclei=['H', '13C'], offsets=[0.0, 0.0], couplings={(0, 1): D_HZ})
    sigma0 = embed(Ix(), 1, 2)
    if w1_hz == 0.0:
        seq = LiouvilleSequence([Delay(T)], ss)
        return seq.propagate(sigma0)

    n_cycles = max(1, int(round(w1_hz / 1000.0 * T)))
    points = max(50, n_cycles * 8)   # enough samples per RF cycle to resolve it
    shape = RFShape.create("hard", duration=T, points=points)
    flip_total = 2 * PI * (w1_hz / 1000.0) * T   # total CW nutation angle over T
    pulse_H = Pulse(shape, flip_total, axis="x", backend=NumpyBackend(Gamma=gyro_ratio('H')))
    seg = ShapePulseSegment(pulse_H)
    seq = LiouvilleSequence([seg], ss)
    return seq.propagate(sigma0)


w1_values_khz = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200, 300])
retained = np.array([Ix2_retained(run_decoupling(w1 * 1000.0)) for w1 in w1_values_khz])

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(w1_values_khz, retained, 'o-')
ax.axvline(D_HZ / 1000.0, color='gray', linestyle=':', label=f"$w_1 = D$ ({D_HZ/1000:.0f} kHz)")
ax.set_xlabel(r"$^1$H decoupling nutation rate $w_1$ (kHz)")
ax.set_ylabel("retained in-phase S coherence")
ax.set_title(f"CW dipolar decoupling (D = {D_HZ/1000:.0f} kHz, T = {T:.4f} ms)")
ax.legend()
plt.tight_layout()
plt.show()

print(f"No decoupling:    retained = {retained[0]:.4f}  (fully dephased)")
print(f"w1 = D:           retained = {retained[np.argmin(np.abs(w1_values_khz - D_HZ/1000))]:.4f}")
print(f"w1 = 15x D:       retained = {retained[-1]:.4f}  (nearly fully decoupled)")