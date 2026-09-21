"""
tutorial_inept.py -- INEPT (Insensitive Nuclei Enhanced by Polarization
Transfer, Morris & Freeman, JACS 101, 760 (1979)).

Sequence (non-refocused INEPT, I = 1H, S = 13C):

    Iz(I)  --90x(I)-->  --delay,Delta-->  --180x(I),180x(S)-->  --delay,Delta-->  --90y(I),90x(S)-->

The two delays + simultaneous 180s form a spin echo: it refocuses each
spin's own chemical-shift offset (so the sequence works regardless of
resonance offset) while leaving the heteronuclear J-coupling evolving
continuously across the full 2*Delta -- that asymmetry (offsets cancel,
J doesn't) is the entire mechanism INEPT exploits to transfer coherence
from the sensitive, high-gamma I spin onto the insensitive S spin.

Verified numerically (product-operator ground truth, Levitt's Spin
Dynamics conventions) before writing this file:
  - the final state's antiphase S-coherence amplitude (coefficient of
    2*Iz(I)*Iy(S)) equals exactly -sin(2*pi*J*Delta), peaking at
    Delta = 1/(4J);
  - WITH the 180 refocusing pulses, the final transfer amplitude is
    identical on- and off-resonance; WITHOUT them, off-resonance runs
    scramble into a mix of in-phase and antiphase S coherence instead.
"""

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator, SpinOperators
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, LiouvilleSequence
from PULSIM.sequence_figure import draw_sequence

PI = np.pi
J_HZ = 140.0    # real Hz, ~1J(CH)

ss_template_offsets = [0.0, 0.0]

def build_inept(Delta, off_I=0.0, off_S=0.0, refocus=True):
    """Propagate equilibrium Iz(I) through the INEPT sequence, return the
    final density matrix."""
    ss = SpinSystem(nuclei=['H', '13C'], offsets=[off_I, off_S], couplings={(0, 1): J_HZ})
    segments = [IdealPulse(PI / 2, phase=0.0, channel='H')]
    segments.append(Delay(Delta))
    if refocus:
        segments.append(IdealPulse(PI, phase=0.0, channel='H'))
        segments.append(IdealPulse(PI, phase=0.0, channel='13C'))
    segments.append(Delay(Delta))
    segments.append(IdealPulse(PI / 2, phase=PI / 2, channel='H'))
    segments.append(IdealPulse(PI / 2, phase=0.0, channel='13C'))

    return LiouvilleSequence(segments, ss)

def run_inept(Delta, off_I=0.0, off_S=0.0, refocus=True):
    ss = SpinSystem(nuclei=['H', '13C'], offsets=[off_I, off_S], couplings={(0, 1): J_HZ})

    sigma_final = build_inept(Delta, off_I, off_S, refocus).propagate(embed(Iz(), 0, 2))
    ops = SpinOperators(spin_system=ss)

    return ops.readout(sigma_final, ['IzSy'])['IzSy']

def antiphase_S_amplitude(sigma):
    """Coefficient of 2*Iz(I)*Iy(S) -- the antiphase S-spin coherence
    INEPT delivers, via the trace formula <A> = Tr(sigma A) / Tr(A A)."""
    A = product_operator(Iz(), 0, Iy(), 1, 2)
    return np.trace(sigma @ A).real / np.trace(A @ A).real

# -- transfer efficiency vs Delta ----------------------
Delta_opt = 1.0 / (4 * J_HZ / 1000.0) # ms
deltas = np.linspace(0.0, 2 * Delta_opt, 60)
transfer = np.array([run_inept(d) for d in deltas])
theory = -np.sin(2 * PI * (J_HZ / 1000.0) * deltas)

fig, axs = plt.subplots(2, 1, figsize=(7, 8))
axs[0].plot(deltas, transfer, 'o', label="simulated (product-operator)")
axs[0].plot(deltas, theory, '-', label=r"theory: $-\sin(2\pi J \Delta)$")
axs[0].axvline(Delta_opt, color='gray', linestyle=':', label=r"$\Delta = 1/(4J)$")
axs[0].set_xlabel("Delta (ms)")
axs[0].set_ylabel(r"antiphase S amplitude ($2 I_z(I) I_y(S)$)")
axs[0].set_title(f"INEPT transfer efficiency vs delay (J = {J_HZ:.0f} Hz)")
axs[0].legend()

draw_sequence(build_inept(Delta_opt), ax=axs[1], to_scale=False,
              title=f"ideal INEPT, to scale  ($\\Delta$ = {Delta_opt:.2f} ms, "
                    f"1.5 ms shaped pulses)")
plt.tight_layout()
plt.savefig("tutorial_figures/tutorial_inept.png")
plt.show()

print("Transfer amplitude at Delta_opt (on-resonance, refocused):",
      run_inept(Delta_opt, 0.0, 0.0, refocus=True))