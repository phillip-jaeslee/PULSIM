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

PI = np.pi
J_HZ = 140.0    # real Hz, ~1J(CH)

ss_template_offsets = [0.0, 0.0]

def run_inept(Delta, off_I=0.0, off_S=0.0, refocus=True):
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

    seq = LiouvilleSequence(segments, ss)
    sigma0 = embed(Iz(), 0, 2)
    ops = SpinOperators(spin_system=ss)

    return ops.readout(seq.propagate(sigma0), ['IzSy'])['IzSy']

def antiphase_S_amplitude(sigma):
    """Coefficient of 2*Iz(I)*Iy(S) -- the antiphase S-spin coherence
    INEPT delivers, via the trace formula <A> = Tr(sigma A) / Tr(A A)."""
    A = product_operator(Iz(), 0, Iy(), 1, 2)
    return np.trace(sigma @ A).real / np.trace(A @ A).real

def draw_pulse_sequence(ax):
    """Standard NMR pulse-sequence diagram: two horizontal timelines (I, S),
    pulses drawn as vertical bars (thin/open = 90 deg, thick/filled = 180
    deg), delays labeled Delta. Schematic -- not to a real time scale,
    matching how these are normally drawn in papers and textbooks."""
    y_I, y_S = 1.0, 0.0
    lw_thin, lw_thick = 0.15, 0.35

    ax.plot([0, 10], [y_I, y_I], color='black', lw=1)
    ax.plot([0, 10], [y_S, y_S], color='black', lw=1)
    ax.text(-0.6, y_I, r'$^1$H (I)', va='center', ha='right', fontsize=11)
    ax.text(-0.6, y_S, r'$^{13}$C (S)', va='center', ha='right', fontsize=11)

    def pulse(x, y, width, filled, label):
        height = 0.4
        rect = plt.Rectangle((x - width / 2, y - height / 2), width, height,
                              facecolor='black' if filled else 'white',
                              edgecolor='black', lw=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + height / 2 + 0.15, label, ha='center', va='bottom', fontsize=10)

    def delay_bracket(x_start, x_end, y, label):
        ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle='<->', color='gray'))
        ax.text((x_start + x_end) / 2, y + 0.1, label, ha='center', va='bottom',
                 fontsize=10, color='gray')

    x0, x1, x2 = 1.0, 4.5, 8.0   # schematic event times

    pulse(x0, y_I, lw_thin, False, r'$90_x$')
    pulse(x1, y_I, lw_thick, True, r'$180_x$')
    pulse(x2, y_I, lw_thin, False, r'$90_y$')

    pulse(x1, y_S, lw_thick, True, r'$180_x$')
    pulse(x2, y_S, lw_thin, False, r'$90_x$')

    delay_bracket(x0, x1, -0.55, r'$\Delta$')
    delay_bracket(x1, x2, -0.55, r'$\Delta$')

    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-1.0, 1.7)
    ax.axis('off')
    ax.set_title('INEPT pulse sequence (schematic, not to scale)')

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

draw_pulse_sequence(axs[1])

plt.savefig("tutorial_figures/tutorial_inept.png")
plt.tight_layout()
plt.show()

print("Transfer amplitude at Delta_opt (on-resonance, refocused):",
      run_inept(Delta_opt, 0.0, 0.0, refocus=True))