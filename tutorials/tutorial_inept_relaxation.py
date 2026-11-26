"""
tutorial_inept_relaxation.py -- the optimal INEPT delay when the 1H
coherence relaxes (T2) while J evolves.

Sequence (non-refocused INEPT, I = 1H, S = 13C):

    Iz(I) --90x(I)-- Delta --180x(I),180x(S)-- Delta --90y(I),90x(S)--

Without relaxation the transfer amplitude is sin(2*pi*J*Delta), largest at
Delta = 1/(4J). With 1H transverse relaxation the coherence also decays
over the whole echo, 2*Delta:

    transfer(Delta) = sin(2*pi*J*Delta) * exp(-2*Delta / T2)

Setting d/dDelta = 0 gives the optimum

    Delta_opt = arctan(pi * J * T2) / (2 * pi * J)

which returns to 1/(4J) as T2 -> infinity and moves to shorter delays as
T2 shortens. This tutorial checks the formula against the density-matrix
simulation with phenomenological relaxation (relaxation.py).

T1 is left out here (T1 = None) so that the formula is exact: in the
relaxation model the antiphase term 2IySz decays at 1/T2(I) + 1/T1(S),
so a short 13C T1 would add a second rate.
"""

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.spin_operators import Iz, embed, SpinOperators
from PULSIM.spin_system import SpinSystem
from PULSIM.liouville import Delay, IdealPulse, LiouvilleSequence
from PULSIM.relaxation import Relaxation
from PULSIM.pulse_diagnostics import optimize_delay

PI = np.pi
J_HZ = 140.0                        # 1J(CH), Hz
J = J_HZ / 1000                     # kHz, because times are in ms
T2_LIST = [None, 20.0, 5.0, 2.0]    # 1H T2, ms (None = no relaxation)

ss = SpinSystem(nuclei=['H', '13C'], offsets=[0.0, 0.0], couplings={(0, 1): J_HZ})
ops = SpinOperators(spin_system=ss)

def build_inept(Delta, T2_H):
    """INEPT with 1H T2 relaxation during the delays."""
    segments = [
        IdealPulse(PI / 2, phase=0.0, channel='H'),
        Delay(Delta),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(Delta),
        IdealPulse(PI / 2, phase=PI / 2, channel='H'),
        IdealPulse(PI / 2, phase=0.0, channel='13C'),
    ]
    relax = Relaxation(T1=None, T2=[T2_H, None])    # for T2, [1H, 13C]
    return LiouvilleSequence(segments, ss, relaxation=relax)

def transfer(sigma):
    """Antiphase 13C coherence 2IzSy (sign flipped so the peak is positive)."""
    return -ops.readout(sigma, ['2IzSy'])['2IzSy']

def theory(Delta, T2_H):
    decay = 1.0 if T2_H is None else np.exp(-2 * Delta / T2_H)
    return np.sin(2 * PI * J * Delta) * decay

def theory_opt(T2_H):
    if T2_H is None:
        return 1.0 / (4 * J)
    return np.arctan(PI * J * T2_H) / (2 * PI * J)

# -- transfer vs Delta, simulation and formula --
textbook = 1.0 / (4 * J)
deltas = np.linspace(0.0, 2 * textbook, 40)
fine = np.linspace(0.0, 2 * textbook, 400)

fig, ax = plt.subplots(figsize=(7, 4))
for T2_H in T2_LIST:
    label = r"no relaxation" if T2_H is None else rf"$T_2$($^1$H) = {T2_H:g} ms"
    sim = [transfer(build_inept(d, T2_H).propagate(embed(Iz(), 0, 2))) for d in deltas]
    line, = ax.plot(fine, theory(fine, T2_H), '-', lw=1, label=label)
    ax.plot(deltas, sim, 'o', ms=3, color=line.get_color())
    ax.axvline(theory_opt(T2_H), color=line.get_color(), ls=':', lw=0.8)

ax.set_xlabel(r"$\Delta$ (ms)")
ax.set_ylabel(r"transfer ($2I_zS_y$)")
ax.set_ylim(-1, 1)
ax.set_title(rf"INEPT delay with $^1$H $T_2$ relaxation (J = {J_HZ:.0f} Hz)", fontsize=10)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("tutorial_figures/tutorial_inept_relaxation.png")
plt.show()

# -- numerical optimum vs formula --------------------------------------------
print(f"textbook 1/(4J) = {textbook:.4f} ms\n")
print(f"{'T2(1H), ms':>12} {'formula':>10} {'simulated':>10} {'transfer':>10}")
for T2_H in T2_LIST:
    run = lambda d, T2_H=T2_H: build_inept(d, T2_H).propagate(embed(Iz(), 0, 2))
    result = optimize_delay(run, transfer, bounds=(0.05, 1.5 * textbook))
    name = "inf" if T2_H is None else f"{T2_H:g}"
    print(f"{name:>12} {theory_opt(T2_H):>10.4f} {result.x:>10.4f} "
          f"{transfer(run(result.x)):>10.4f}")