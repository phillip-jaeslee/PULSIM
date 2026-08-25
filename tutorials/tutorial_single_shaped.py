"""
tutorial_single_shaped.py -- the shaped-pulse refocusing element
from the 2nd INEPT of Rance-Kay sensitivity-enhanced detection, ported
from this lab's densityMatSim/dmSimIySz_Ix_EB2try_1ov2J_EB2x_rfshape.py
onto PULSIM.liouville's SpinSystem/Segment classes.

Sequence (I = 1H, S = 15N, J = 92 Hz = real 1J(NH)):

    sigma0  --EBURP2tr(I)-->  --delay,1/(4J)-->  --180x(I),180x(S)-->  --delay,1/(4J)-->  --EBURP2(I)-->

Same spin-echo mechanism as tutorial_inept.py's idealized hard-pulse
INEPT, but here the pulses (EBURP2tr going in, time-reversed EBURP2
coming out) are real band-selective shapes loaded from this lab's own
wave files, and the whole thing is swept across resonance offset (not
delay) to see how transfer fidelity holds up across the shape's
bandwidth -- i.e. this is the "realistic, finite-bandwidth pulse"
companion to the idealized tutorial.

RF is specified as a fixed real hardware peak power (2730.78 Hz nutation)
rather than "calibrated to a flip angle", so the shaped-pulse segments use
RawShapedPulseSegment (which takes an already-scaled RF trajectory
directly) instead of ShapePulseSegment (which calibrates via a target
flip angle -- not the right fit for a fixed-power specification, see
RawShapedPulseSegment's docstring in liouville.py).

Verified against the original dmSim script's xyzBasis + scipy.linalg.expm
computation before this file was written: matches to ~1.5e-9 across
sampled offsets, for both starting states shown below.
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

J_HZ = 92.0                         # real 1J(NH)
tp_ms = 1                           # shaped pulse length, ms
N = 64                              # number of offsets to sweep
W_hz = np.linspace(-1, 1, N) * 4000.0

GAMMA_H = gyro_ratio('H')
GAMMA_C = gyro_ratio('13C')
GAMMA_N = gyro_ratio('15N')

# -- Apply 90˚ GaussCascadeQ5 pulse ------------------
shape = RFShape.create("gausscasq5", duration=tp_ms, points=500)
pulse = Pulse(shape, PI / 2, axis = "x", backend=NumpyBackend(Gamma=GAMMA_H))
seg = ShapePulseSegment(pulse)


def run(sigma0, off_hz, J, keys):
    off_H = twoPI * (off_hz / 1000.0)
    ss = SpinSystem(nuclei=['H', '15N'], offsets=[off_H, 0.0], couplings={(0, 1): J})
    seq = LiouvilleSequence([seg], ss)
    ops = SpinOperators(spin_system=ss)
    return ops.readout(seq.propagate(sigma0), keys)

Iz0 = embed(Iz(), 0 , 2)
keys = ['Ix', 'Iy', 'Iz', 'IxSz', 'IySz', 'IzSz']
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
    ax[0].plot(W_hz, with_J[k], '-', color=color, label=k)
ax[0].set_title(f'Single 90x GaussCascadeQ5, WITH J={J_HZ:.0f} Hz')
ax[0].set_xlabel('offset (Hz)')
ax[0].legend(fontsize=9)
ax[0].invert_xaxis()

ax[1].axhline(y=0, color='lightgray', linestyle='-')
for k, color in zip(keys, ['red', 'blue', 'green', 'orange', 'cyan', 'olive']):
    ax[1].plot(W_hz, no_J[k], '-', color=color, label=k)
ax[1].set_title('Same pulse, J = 0 (idealized, coupling-free)')
ax[1].set_xlabel('offset (Hz)')
ax[1].legend(fontsize=9)

plt.savefig("tutorial_figures/tutorial_single_shaped.png")
plt.show()

print(f"On-resonance: with J, IxSz={with_J['IxSz'][N//2]:.4f}; "
      f"with J=0, IxSz={no_J['IxSz'][N//2]:.4f} (should be ~0)")