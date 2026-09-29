"""
tutorial_inept_shaped.py -- INEPT (Insensitive Nuclei Enhanced by
Polarization Transfer, Morris & Freeman, JACS 101, 760 (1979)) with real
band-selective shaped pulses in place of the hard 90s of tutorial_inept.py.

Sequence (non-refocused INEPT, I = 1H, S = 13C, J = 140 Hz ~ 1J(CH)):

    Iz(I) --eb2try(I)--> --Delta--> --180x(I),180x(S)--> --Delta--> --eb2x(I), 90x(S)-->

The two delays + simultaneous 180s form a spin echo: it refocuses each
spin's own chemical-shift offset while leaving the heteronuclear J-coupling
evolving across the full 2*Delta -- the asymmetry INEPT exploits to move
polarization from the sensitive I spin onto the insensitive S spin.

With hard pulses the antiphase S amplitude (coefficient of 2*Iz(I)*Iy(S)) is
-sin(2*pi*J*Delta), peaking at Delta = 1/(4J) (see tutorial_inept.py). The
1.5 ms E-BURP-2 pulses used here (eb2try going in: time-reversed, y phase;
eb2x coming out) let J evolve during the pulses themselves, so the optimum
moves to a SHORTER delay. The shift, 1.077 ms, is a property of the pulses,
not of J: it is the same shortening found in tutorial_delay_optimization.py
for J = 92 Hz, and the lab's empirical correction is 1.08 ms.

The pulses are PULSIM's analytic E-BURP-2, so the tutorial runs from a clean
clone; see tutorial_HN_shaped_refocusing.py for how they relate to the lab's
vendor shape files.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from PULSIM.spin_operators import Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, LiouvilleSequence, ShapePulseSegment
from PULSIM.pulse_oo import Pulse
from PULSIM.backend import NumpyBackend
from PULSIM.rf_shape import RFShape
from PULSIM.sequence_figure import draw_sequence

PI = np.pi
J_HZ = 140          # Hz, ~1J(CH)
duration = 1.5      # ms, each shaped pulse

backend = NumpyBackend(Gamma=gyro_ratio('H'))
eb2x = Pulse(RFShape.create("eburp2", duration=duration, points=500),
             PI / 2, axis="x", backend=backend)
eb2try = Pulse(RFShape.create("eburp2", duration=duration, points=500, time_reversed=True),
               PI / 2, axis="y", backend=backend)


def build_inept(Delta, off_I=0.0, off_S=0.0, refocus=True):
    """The sequence itself, as segments.

    Separated from the propagation so that the figure and the simulation are
    built from the same object -- the diagram is drawn by walking these
    segments, so it cannot describe a sequence other than the one that ran.
    """
    ss = SpinSystem(nuclei=['H', '13C'], offsets=[off_I, off_S], couplings={(0, 1): J_HZ})

    segments = [ShapePulseSegment(eb2try), Delay(Delta)]
    if refocus:
        segments.append(IdealPulse(PI, phase=0.0, channel='H'))
        segments.append(IdealPulse(PI, phase=0.0, channel='13C'))
    segments.append(Delay(Delta))
    segments.append(ShapePulseSegment(eb2x))
    segments.append(IdealPulse(PI / 2, phase=0.0, channel='13C'))

    return LiouvilleSequence(segments, ss)


def run_inept(Delta, off_I=0.0, off_S=0.0, refocus=True):
    """Propagate equilibrium Iz(I) through the shaped INEPT sequence, return
    the antiphase-S amplitude (coefficient of 2*Iz(I)*Iy(S))."""
    sigma_final = build_inept(Delta, off_I, off_S, refocus).propagate(embed(Iz(), 0, 2))
    A = product_operator(Iz(), 0, Iy(), 1, 2)
    return np.trace(sigma_final @ A).real / np.trace(A @ A).real

# -- transfer efficiency vs Delta ----------------------
Delta_hard = 1.0 / (4 * J_HZ / 1000.0)   # ms, the hard-pulse optimum 1/(4J)
deltas = np.linspace(0.0, 2 * Delta_hard, 60)
transfer = np.array([run_inept(d) for d in deltas])
theory = -np.sin(2 * PI * (J_HZ / 1000.0) * deltas)

best = minimize_scalar(lambda d: run_inept(d), bounds=(0.05, 2 * Delta_hard), method="bounded")
Delta_shaped = best.x

fig, axs = plt.subplots(2, 1, figsize=(7, 8))
axs[0].plot(deltas, transfer, 'o', label="simulated, eb2try / eb2x (1.5 ms)")
axs[0].plot(deltas, theory, '-', label=r"hard pulses: $-\sin(2\pi J \Delta)$")
axs[0].axvline(Delta_hard, color='gray', linestyle=':', label=r"$\Delta = 1/(4J)$")
axs[0].axvline(Delta_shaped, color='C0', linestyle='--', label=f"shaped optimum, {Delta_shaped:.3f} ms")
axs[0].set_xlabel("Delta (ms)")
axs[0].set_ylabel(r"antiphase S amplitude ($2 I_z(I) I_y(S)$)")
axs[0].set_title(f"Shaped INEPT transfer vs delay (J = {J_HZ:.0f} Hz)")
axs[0].legend()

draw_sequence(build_inept(Delta_shaped), ax=axs[1], to_scale=False,
              title=f"shaped INEPT, not to scale  ($\\Delta$ = {Delta_shaped:.2f} ms, "
                    f"1.5 ms shaped pulses)")
plt.tight_layout()
plt.savefig("tutorial_figures/tutorial_inept_shaped.png", dpi=150)
plt.show()

print(f"at Delta = 1/(4J) = {Delta_hard:.4f} ms: transfer {run_inept(Delta_hard):+.4f}")
print(f"shaped optimum     = {Delta_shaped:.4f} ms: transfer {run_inept(Delta_shaped):+.4f}")
print(f"shortened by {Delta_hard - Delta_shaped:.4f} ms (tutorial_delay_optimization.py: 1.077 ms)")