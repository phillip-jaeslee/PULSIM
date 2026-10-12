"""
tutorial_single_shaped.py -- what J-coupling does DURING a single shaped
pulse.

A single 90x EBURP1 (1 ms) acts on a 1H-15N pair starting from Iz(H), swept
across resonance offset, twice: with the heteronuclear coupling on
(J = 92 Hz, a real 1J(NH)) and with it off (J = 0). With J = 0 the pulse only
rotates Iz into transverse Ix/Iy. With J on, the coupling keeps evolving for
the whole millisecond the pulse lasts, so part of the transverse I
magnetization turns into antiphase 2IxSz/2IySz -- a term a hard pulse would
never create. Here it reaches about 6 % (2IySz). The same J evolution during
the pulse is what shifts the optimal INEPT delay for finite shaped pulses
(see tutorial_delay_optimization.py and tutorial_inept_shaped.py).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator, SpinOperators
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, RawShapedPulseSegment, LiouvilleSequence, ShapePulseSegment
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.backend import NumpyBackend

PI, twoPI = np.pi, 2 * np.pi

J_HZ = 92.0                         # Hz, 1J(NH)
tp_ms = 1                           # shaped pulse length, ms
N = 65                              # offsets to sweep (odd: centre is on resonance)
W_hz = np.linspace(-1, 1, N) * 4000.0

GAMMA_H = gyro_ratio('H')
GAMMA_C = gyro_ratio('13C')
GAMMA_N = gyro_ratio('15N')

# -- Apply a 90x EBURP1 pulse ------------------------
shape = RFShape.create("eburp1", duration=tp_ms, points=500)
pulse = Pulse(shape, PI / 2, axis = "x", backend=NumpyBackend(Gamma=GAMMA_H))
seg = ShapePulseSegment(pulse)


def run(sigma0, off_hz, J, keys):
    off_H = twoPI * (off_hz / 1000.0)
    ss = SpinSystem(nuclei=['H', '15N'], offsets=[off_H, 0.0], couplings={(0, 1): J})
    seq = LiouvilleSequence([seg], ss)
    ops = SpinOperators(spin_system=ss)
    return ops.readout(seq.propagate(sigma0), keys)

Iz0 = embed(Iz(), 0 , 2)
keys = ['Ix', 'Iy', 'Iz', '2IxSz', '2IySz', '2IzSz']
TEX = {'Ix': r'$I_x$', 'Iy': r'$I_y$', 'Iz': r'$I_z$',
       '2IxSz': r'$2I_xS_z$', '2IySz': r'$2I_yS_z$', '2IzSz': r'$2I_zS_z$'}
with_J = {k: np.zeros(N) for k in keys}
no_J = {k: np.zeros(N) for k in keys}


for n, off in enumerate(W_hz):
    r_with = run(Iz0, off, J_HZ, keys)
    r_no = run(Iz0, off, 0.0, keys)
    for k in keys:
        with_J[k][n] = r_with[k]
        no_J[k][n] = r_no[k]

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=True, constrained_layout=True)
fig.set_size_inches(11, 4.5, forward=True)

ax[0].axhline(y=0, color='lightgray', linestyle='-')
for k, color in zip(keys, ['red', 'blue', 'green', 'orange', 'cyan', 'olive']):
    ax[0].plot(W_hz, with_J[k], '-', color=color, label=TEX[k])
ax[0].set_title(f'Single 90x EBURP1, WITH J={J_HZ:.0f} Hz')
ax[0].set_xlabel('offset (Hz)')
ax[0].legend(fontsize=9)
ax[0].invert_xaxis()

ax[1].axhline(y=0, color='lightgray', linestyle='-')
for k, color in zip(keys, ['red', 'blue', 'green', 'orange', 'cyan', 'olive']):
    ax[1].plot(W_hz, no_J[k], '-', color=color, label=TEX[k])
ax[1].set_title('Same pulse, J = 0 (idealized, coupling-free)')
ax[1].set_xlabel('offset (Hz)')
ax[1].legend(fontsize=9)

plt.savefig("tutorial_figures/tutorial_single_shaped_EBURP1.png")
plt.show()

print(f"On-resonance: with J, 2IxSz={with_J['2IxSz'][N//2]:.4f}; "
      f"with J=0, 2IxSz={no_J['2IxSz'][N//2]:.4f} (should be ~0)")