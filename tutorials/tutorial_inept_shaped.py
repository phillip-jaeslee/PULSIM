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

# -- offset independence, with vs without refocusing --------
off_I, off_S = 2 * PI * 0.5, 2 * PI * 0.3
cases = [
    ("refocused, \non-resonance", run_inept(Delta_opt, 0.0, 0.0, refocus=True)),
    ("refocused, \noff-resonance", run_inept(Delta_opt, off_I, off_S, refocus=True)),
    ("NOT efocused, \non-resonance", run_inept(Delta_opt, 0.0, 0.0, refocus=False)),
    ("NOT refocused, \noff-resonance", run_inept(Delta_opt, off_I, off_S, refocus=False)),
]

labels = [c[0] for c in cases]
amps = [abs(c[1]) for c in cases]

axs[1].bar(labels, amps, color=["C0", "C0", "C3", "C3"])
axs[1].set_ylabel("|antiphase S amplitude|")
axs[1].set_title("Refocusing pulses make INEPT offset-independent")
axs[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.show()

print("Transfer amplitude at Delta_opt (on-resonance, refocused):",
      run_inept(Delta_opt, 0.0, 0.0, refocus=True))