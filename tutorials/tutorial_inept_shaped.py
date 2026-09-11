"""
tutorial_inept.py -- INEPT (Insensitive Nuclei Enhanced by Polarization
Transfer, Morris & Freeman, JACS 101, 760 (1979)).

Sequence (non-refocused INEPT, I = 1H, S = 15N):

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

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, RawShapedPulseSegment, LiouvilleSequence
from PULSIM.rf_shape import RFShape

PI = np.pi
J_HZ = 92          # real Hz, ~1J(NH)
duration = 1.5     # ms

GAMMA_H = gyro_ratio('H')
GAMMA_N = gyro_ratio('15N')
rfPow = 2 * PI * 2730.78242 / 1000.0   # real hardware peak power, rad/ms
                                        # (2730.78242 Hz -> rad/s -> rad/ms;
                                        # see tutorial_HN_shaped_refocusing.py)

def hardware_rf(path):
    """EBURP2/EBURP2tr are calibrated to a fixed real peak power, not a
    target flip angle -- calibrated_rf()'s flip-based normalization can't
    reproduce that (dividing a real flip by sum(envelope) doesn't recover a
    fixed physical power, and for eb2try's y-phased/imaginary envelope it
    also silently rotates the nutation axis). Build the RF trajectory
    directly instead, same as tutorial_HN_shaped_refocusing.py."""
    shape = RFShape.create("file", path=path, duration=duration)
    rf = np.conj(shape.envelope()) * rfPow / 100.0   # /100: file stores 0-100%
    return rf, shape.dt

def run_inept(Delta, off_I=0.0, off_S=0.0, refocus=True):
    """Propagate equilibrium Iz(I) through the shaped INEPT sequence, return
    the antiphase-S amplitude (coefficient of 2*Iz(I)*Iy(S))."""
    ss = SpinSystem(nuclei=['H', '15N'], offsets=[off_I, off_S], couplings={(0, 1): J_HZ})

    rf1, dt1 = hardware_rf('wave/eb2try_1.5m_ofs0Hz.500')
    rf2, dt2 = hardware_rf('wave/eb2x_1.5m_ofs0Hz.500')

    segments = [RawShapedPulseSegment(rf1, dt1, channel='H')]
    segments.append(Delay(Delta))
    if refocus:
        segments.append(IdealPulse(PI, phase=0.0, channel='H'))
        segments.append(IdealPulse(PI, phase=0.0, channel='15N'))
    segments.append(Delay(Delta))
    segments.append(RawShapedPulseSegment(rf2, dt2, channel='H'))
    segments.append(IdealPulse(PI / 2, phase=0.0, channel='15N'))

    seq = LiouvilleSequence(segments, ss)
    sigma0 = embed(Iz(), 0, 2)
    sigma_final = seq.propagate(sigma0)

    A = product_operator(Iz(), 0, Iy(), 1, 2)
    return np.trace(sigma_final @ A).real / np.trace(A @ A).real

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

    def shaped_pulse(x, y, width, label):
        """Shaped pulse (EBURP2-type envelope): drawn as a smooth amplitude
        bump instead of a hard-pulse rectangle, to visually distinguish
        eb2try/eb2x from the ideal 180s in the same diagram."""
        height = 0.4
        xs = np.linspace(x - width / 2, x + width / 2, 60)
        env = 0.5 * (1 - np.cos(2 * np.pi * (xs - (x - width / 2)) / width))
        ax.fill_between(xs, y - height / 2 * env, y + height / 2 * env,
                         facecolor='0.75', edgecolor='black', lw=1.2, zorder=3)
        ax.text(x, y + height / 2 + 0.15, label, ha='center', va='bottom', fontsize=10)

    def delay_bracket(x_start, x_end, y, label):
        ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle='<->', color='gray'))
        ax.text((x_start + x_end) / 2, y + 0.1, label, ha='center', va='bottom',
                 fontsize=10, color='gray')

    x0, x1, x2 = 1.0, 4.5, 8.0   # schematic event times

    shaped_pulse(x0, y_I, 0.7, 'EBURP2$^{tr}$\n$90_x$')
    pulse(x1, y_I, lw_thick, True, r'$180_x$')
    shaped_pulse(x2, y_I, 0.7, 'EBURP2\n$90_y$')
    
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
plt.tight_layout()
plt.savefig("tutorial_figures/tutorial_inept_shaped.png", dpi=150)
plt.show()

print("Transfer amplitude at Delta_opt (on-resonance, refocused):",
      run_inept(Delta_opt, 0.0, 0.0, refocus=True))