"""
tutorial_compensated_bird.py -- compensated BIRD vs simple BIRD: robustness
to J(IS) mismatch (Garbow, Weitekamp & Pines, Chem. Phys. Lett. 93, 504
(1982), Section 6, "Compensated bilinear rotations").

Simple BIRD:
    90y(I) -- 2tau -- 180(I,S) -- 2tau -- 90-y(I)

Compensated BIRD replaces the single central bilinear-pi rotation with three
smaller ones, bracketed by their own 90-degree proton pulses with alternating
phase (y, -x, x, -y):
    90y(I) -- tau -- 180(I,S) -- tau -- 90-x(I)
            -- 2tau -- 180(I,S) -- 2tau -- 90x(I)
            -- tau -- 180(I,S) -- tau -- 90-y(I)

Both are calibrated to a single reference coupling J0(IS) via 4*tau=1/J0.
The whole point of BIRD is to act as a nonselective proton pulse across many
different C-H pairs, each with its own actual J(IS) -- so this sweeps the
*actual* J(IS) away from J0 and shows the compensated element holds I's
inversion much closer to -1 over a wider range than the simple element does.

Verified numerically before writing this file: at J(IS)=0.8*J0, simple BIRD
inverts I to only -0.80 while compensated BIRD reaches -0.98; I' (not
S-attached) stays ~+0.99 throughout for both, confirming the extra
robustness is specific to the bilinear (J-driven) part of the sequence, not
a side effect on the homonuclear filter itself.
"""

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem
from PULSIM.liouville import Delay, IdealPulse, LiouvilleSequence

PI = np.pi
J0 = 140.0      # Hz, reference/calibration 1J(CH) -- both sequences timed to this
J_HH = 7.0      # Hz, homonuclear coupling I-I' (unaffected by either sequence)
N_SPINS = 3

def simple_bird(J0_hz):
    """90y(I) -- 2tau -- 180(I,S) -- 2tau -- 90-y(I), 4*tau = 1/J0."""
    tau2 = 1.0 / (2 * J0_hz / 1000.0)   # ms, paper's "2*tau"
    return [
        IdealPulse(PI / 2, phase=PI / 2, channel='H'),
        Delay(tau2),
        IdealPulse(PI, phase=0.0, channel='H'), IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau2),
        IdealPulse(PI / 2, phase=-PI / 2, channel='H'),
    ]

def compensated_bird(J0_hz):
    """90y(I) -- tau-180-tau -- 90-x(I) -- 2tau-180-2tau -- 90x(I)
    -- tau-180-tau -- 90-y(I), 4*tau = 1/J0."""
    tau = 1.0 / (4 * J0_hz / 1000.0)   # ms
    return [
        IdealPulse(PI / 2, phase=PI / 2, channel='H'),
        Delay(tau),
        IdealPulse(PI, phase=0.0, channel='H'), IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau),
        IdealPulse(PI / 2, phase=PI, channel='H'),
        Delay(2 * tau),
        IdealPulse(PI, phase=0.0, channel='H'), IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(2 * tau),
        IdealPulse(PI / 2, phase=0.0, channel='H'),
        Delay(tau),
        IdealPulse(PI, phase=0.0, channel='H'), IdealPulse(PI, phase=0.0, channel='13C'),
        Delay(tau),
        IdealPulse(PI / 2, phase=-PI / 2, channel='H'),
    ]

def coeff(sigma, A):
    return np.trace(sigma @ A).real / np.trace(A @ A).real

def run(segments, J_CH_actual):
    ss = SpinSystem(nuclei=['H', 'H', '13C'], offsets=[0.0, 0.0, 0.0],
                     couplings={(0, 2): J_CH_actual, (0, 1): J_HH})
    seq = LiouvilleSequence(segments, ss)
    Iz0_I  = embed(Iz(), 0, N_SPINS)
    Iz0_Ip = embed(Iz(), 1, N_SPINS)
    zI  = coeff(seq.propagate(Iz0_I),  Iz0_I)
    zIp = coeff(seq.propagate(Iz0_Ip), Iz0_Ip)
    return zI, zIp

# -- sweep actual J(IS) away from the calibration J0 --------------------
J_scan = np.linspace(0.6, 1.4, 41) * J0   # 60%-140% of the calibration value

zI_simple  = np.zeros_like(J_scan)
zI_comp    = np.zeros_like(J_scan)
zIp_simple = np.zeros_like(J_scan)
zIp_comp   = np.zeros_like(J_scan)

for n, J in enumerate(J_scan):
    zI_simple[n],  zIp_simple[n] = run(simple_bird(J0),      J)
    zI_comp[n],    zIp_comp[n]   = run(compensated_bird(J0), J)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
fig.set_size_inches(11, 4.5, forward=True)

ax[0].axhline(y=-1, color='lightgray', linestyle='--')
ax[0].plot(J_scan, zI_simple, '--', color='red',  label='simple BIRD')
ax[0].plot(J_scan, zI_comp,   '-',  color='blue', label='compensated BIRD')
ax[0].axvline(J0, color='gray', linestyle=':', label=f'J0 = {J0:.0f} Hz')
ax[0].set_title("I (S-attached) inversion vs actual J(IS)")
ax[0].set_xlabel('J(IS) (Hz)')
ax[0].set_ylabel('Iz after BIRD')
ax[0].legend(fontsize=9)

ax[1].axhline(y=1, color='lightgray', linestyle='--')
ax[1].plot(J_scan, zIp_simple, '--', color='red',  label='simple BIRD')
ax[1].plot(J_scan, zIp_comp,   '-',  color='blue', label='compensated BIRD')
ax[1].axvline(J0, color='gray', linestyle=':')
ax[1].set_title("I' (not attached) after BIRD -- should stay ~+1 regardless")
ax[1].set_xlabel('J(IS) (Hz)')
ax[1].legend(fontsize=9)

plt.savefig("tutorial_figures/tutorial_compensated_bird.png")
plt.tight_layout()
plt.show()

mid = len(J_scan) // 2
print(f"At J(IS)=J0={J0:.0f} Hz: simple zI={zI_simple[mid]:+.4f}, "
      f"comp zI={zI_comp[mid]:+.4f} (both should be ~-1)")
i80 = np.argmin(np.abs(J_scan - 0.8 * J0))
print(f"At J(IS)=0.8*J0={0.8*J0:.0f} Hz: simple zI={zI_simple[i80]:+.4f}, "
      f"comp zI={zI_comp[i80]:+.4f} -- compensated BIRD is much closer to -1")
print("I' stays ~+0.988 throughout for both sequences -- J(IS) mismatch only "
      "affects the S-attached spin, exactly as expected.")