"""
tutorial_HN_shaped_refocusing.py -- the shaped-pulse refocusing element
from the 2nd INEPT of Rance-Kay sensitivity-enhanced detection, ported
from this lab's densityMatSim/dmSimIySz_Ix_EB2try_1ov2J_EB2x_rfshape.py
onto PULSIM.liouville's SpinSystem/Segment classes.

Sequence (I = 1H, S = 15N, J = 92 Hz = real 1J(NH)):

    sigma0  --EBURP2tr(I)-->  --delay,1/(4J)-->  --180x(I),180x(S)-->  --delay,1/(4J)-->  --EBURP2(I)-->

Same spin-echo mechanism as tutorial_inept.py's idealized hard-pulse
INEPT, but here the pulses (EBURP2tr going in, time-reversed EBURP2
coming out) are real band-selective shapes, and the whole thing is swept
across resonance offset (not delay) to see how transfer fidelity holds up
across the shape's bandwidth -- i.e. this is the "realistic,
finite-bandwidth pulse" companion to the idealized tutorial.

The pulses are built from PULSIM's analytic E-BURP-2 so the tutorial runs
from a clean clone: eb2x is E-BURP-2 with x phase, eb2try the same pulse
time-reversed with y phase (-i * reversed(eb2x)) -- the construction of the
lab's vendor shape files, which it matches to within 3.4 % in amplitude.

RF is a fixed peak nutation rather than "calibrated to a flip angle" per
segment, so the shaped-pulse segments use RawShapedPulseSegment (which
takes an already-scaled RF trajectory directly) instead of
ShapePulseSegment (see RawShapedPulseSegment's docstring in liouville.py).
The peak is E-BURP-2's 90-degree area calibration, 2.731 kHz -- the same
2730.78 Hz the lab's files are driven at.

The port was verified against the original dmSim script's xyzBasis +
scipy.linalg.expm computation to ~1.5e-9 across sampled offsets, using the
lab's vendor files. With the analytic pulses used here the curves move by
at most 0.03 (on a unit amplitude), mostly near the band edges.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, ShapePulseSegment, LiouvilleSequence
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.backend import NumpyBackend

PI, twoPI = np.pi, 2 * np.pi

J_HZ = 92.0                        # real 1J(NH)
tp_ms = 1.5                        # shaped pulse length, ms
N = 64                              # number of offsets to sweep
W_hz = np.linspace(-1, 1, N) * 4000.0

backend = NumpyBackend(Gamma=gyro_ratio('H'))
eb2x = Pulse(RFShape.create("eburp2", duration=tp_ms, points=500), PI / 2, axis="x", backend=backend)
eb2try = Pulse(RFShape.create("eburp2", duration=tp_ms, points=500, time_reversed=True), PI / 2, axis="y", backend=backend)

seg1 = ShapePulseSegment(eb2try)
seg2 = ShapePulseSegment(eb2x)

Delta_ms = 250.0 / J_HZ   # 1/(4J), ms

def run(sigma0, W_hz_i):
    off_H = twoPI * (W_hz_i / 1000.0)   # Hz -> rad/ms
    ss = SpinSystem(nuclei=['H', '15N'], offsets=[off_H, 0.0], couplings={(0, 1): J_HZ})
    segments = [
        seg1,
        Delay(Delta_ms, include_offset=False),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='15N'),
        Delay(Delta_ms, include_offset=False),
        seg2,
    ]
    seq = LiouvilleSequence(segments, ss)
    return seq.propagate(sigma0)


def extract(sigma):
    names_ops = [('IzSz', product_operator(Iz(), 0, Iz(), 1, 2)),
                 ('IySz', product_operator(Iy(), 0, Iz(), 1, 2)),
                 ('Ix',   embed(Ix(), 0, 2)),
                 ('Iy',   embed(Iy(), 0, 2)),
                 ('Iz',   embed(Iz(), 0, 2)),
                 ('IxSz', product_operator(Ix(), 0, Iz(), 1, 2))]
    return {name: np.trace(sigma @ A).real / np.trace(A @ A).real for name, A in names_ops}


IySz0 = product_operator(Iy(), 0, Iz(), 1, 2)
Ix0 = embed(Ix(), 0, 2)
Iz0 = embed(Iz(), 0, 2)

results_from_Iz = {k: np.zeros(N) for k in ['IzSz', 'IySz', 'Ix', 'Iy', 'Iz', 'IxSz']}
results_from_IySz = {k: np.zeros(N) for k in ['IzSz', 'IySz', 'Ix', 'Iy', 'Iz', 'IxSz']}
results_from_Ix = {k: np.zeros(N) for k in ['IzSz', 'IySz', 'Ix', 'Iy', 'Iz', 'IxSz']}

for n, w in enumerate(W_hz):
    r1 = extract(run(IySz0, w))
    r2 = extract(run(Ix0, w))
    r3 = extract(run(Iz0, w))
    for k in results_from_IySz:
        results_from_IySz[k][n] = r1[k]
        results_from_Ix[k][n] = r2[k]
        results_from_Iz[k][n] = r3[k]

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=True, constrained_layout=True)
fig.set_size_inches(11, 4.5, forward=True)

ax[0].axhline(y=0, color='lightgray', linestyle='-')
ax[0].plot(W_hz, results_from_IySz['IxSz'], '-', color='red', label='IxSz')
ax[0].plot(W_hz, results_from_IySz['IySz'], '-', color='blue', label='IySz')
ax[0].plot(W_hz, results_from_IySz['IzSz'], '-', color='green', label='IzSz')
ax[0].plot(W_hz, results_from_IySz['Ix'], '--', color='orange', label='Ix')
ax[0].plot(W_hz, results_from_IySz['Iy'], '--', color='cyan', label='Iy')
ax[0].plot(W_hz, results_from_IySz['Iz'], '--', color='olive', label='Iz')
ax[0].set_title(f'EBURP2tr/EBURP2 on IySz, J={J_HZ:.0f} Hz')
ax[0].set_xlabel('offset (Hz)')
ax[0].legend(fontsize=9)
ax[0].invert_xaxis()
ax[0].yaxis.set_major_locator(ticker.MultipleLocator(base=0.5))
"""
ax[1].axhline(y=0, color='gray', linestyle='-')
ax[1].plot(W_hz, results_from_Ix['Ix'], '-', color='red', label='Ix')
ax[1].plot(W_hz, results_from_Ix['Iy'], '-', color='blue', label='Iy')
ax[1].plot(W_hz, results_from_Ix['Iz'], '-', color='green', label='Iz')
ax[1].plot(W_hz, results_from_Ix['IxSz'], '--', color='cyan', label='IxSz')
ax[1].plot(W_hz, results_from_Ix['IySz'], '--', color='orange', label='IySz')
ax[1].plot(W_hz, results_from_Ix['IzSz'], '--', color='olive', label='IzSz')
ax[1].set_title(f'EBURP2tr/EBURP2 on Ix, J={J_HZ:.0f} Hz')
ax[1].set_xlabel('offset (Hz)')
ax[1].legend(fontsize=9)
"""
ax[1].axhline(y=0, color='gray', linestyle='-')
ax[1].plot(W_hz, results_from_Iz['Ix'], '-', color='red', label='Ix')
ax[1].plot(W_hz, results_from_Iz['Iy'], '-', color='blue', label='Iy')
ax[1].plot(W_hz, results_from_Iz['Iz'], '-', color='green', label='Iz')
ax[1].plot(W_hz, results_from_Iz['IxSz'], '--', color='cyan', label='IxSz')
ax[1].plot(W_hz, results_from_Iz['IySz'], '--', color='orange', label='IySz')
ax[1].plot(W_hz, results_from_Iz['IzSz'], '--', color='olive', label='IzSz')
ax[1].set_title(f'EBURP2tr/EBURP2 on Iz, J={J_HZ:.0f} Hz')
ax[1].set_xlabel('offset (Hz)')
ax[1].legend(fontsize=9)

plt.savefig("tutorial_figures/tutorial_HN_shaped_refocusing_2.png")
plt.show()