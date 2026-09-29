"""
tutorial_delay_optimization.py -- why the textbook 1/(4J) INEPT delay is
wrong for real (finite-duration) shaped pulses, quantified per-pulse and
then corrected exactly, using this lab's eb2try/eb2x refocusing element
(see tutorial_HN_shaped_refocusing.py) as the test case.

The two pulses are built from PULSIM's analytic E-BURP-2, so the tutorial
runs from a clean clone:
    eb2x   = E-BURP-2, x phase
    eb2try = the same pulse time-reversed, y phase  (-i * reversed(eb2x))
The lab's vendor shape files are the same construction to within 3.4 % in
amplitude; with them the optimal shortening is 1.0795 ms instead of the
1.077 ms printed here, against the lab's empirical correction of 1.08 ms.

Both pulses are driven at the 90-degree area calibration of E-BURP-2
(Pulse(..., flip=pi/2)), which is exactly the peak RF the lab's files use.
"""

import numpy as np

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, RawShapedPulseSegment, LiouvilleSequence
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_diagnostics import effective_coupling_generator, optimize_delay
from PULSIM.pulse_oo import Pulse
from PULSIM.backend import NumpyBackend

PI, twoPI = np.pi, 2 * np.pi

J_HZ = 92.0
tp_ms = 1.5

eburp2 = RFShape.create("eburp2", duration=tp_ms, points=500)
nu1_khz = Pulse(eburp2, PI / 2, backend=NumpyBackend(Gamma=gyro_ratio('H'))).nu1_max
dt = eburp2.dt

env_x = eburp2.envelope() / np.abs(eburp2.envelope()).max()
env_try = -1j * env_x[::-1]                     # time-reversed, y phase

RF1 = np.conj(env_try) * twoPI * nu1_khz        # eb2try, rad/ms
RF2 = np.conj(env_x) * twoPI * nu1_khz          # eb2x,   rad/ms
seg1 = RawShapedPulseSegment(RF1, dt, channel='H')
seg2 = RawShapedPulseSegment(RF2, dt, channel='H')

print(f"E-BURP-2, {tp_ms} ms, 90-degree calibration: nu1_max = {nu1_khz:.5f} kHz\n")

# -- goal 2: per-pulse effective J-evolution time --------------------------
for name, rf in [("eb2try", RF1), ("eb2x", RF2)]:
    Mx_int, My_int, Mz_int = effective_coupling_generator(rf, dt)
    leakage = np.hypot(Mx_int, My_int)
    print(f"{name}: t_eff_z = {Mz_int:+.4f} ms (delay-correctable), "
          f"leakage = {leakage:.4f} ms (not delay-correctable)")

# -- goal 1: exact numerical optimum, no approximation ----------------------
IySz0 = product_operator(Iy(), 0, Iz(), 1, 2)


def run_sequence(Delta, W_hz=0.0):
    off_H = twoPI * (W_hz / 1000.0)
    ss = SpinSystem(nuclei=['H', '15N'], offsets=[off_H, 0.0], couplings={(0, 1): J_HZ})
    segments = [
        seg1,
        Delay(Delta, include_offset=False),
        IdealPulse(PI, phase=0.0, channel='H'),
        IdealPulse(PI, phase=0.0, channel='15N'),
        Delay(Delta, include_offset=False),
        seg2,
    ]
    return LiouvilleSequence(segments, ss).propagate(IySz0)


def Ix_amplitude(sigma):
    A = embed(Ix(), 0, 2)
    return np.trace(sigma @ A).real / np.trace(A @ A).real


textbook = 250.0 / J_HZ   # 1/(4J), ms
# The search range must contain the optimum: a bounded search that stops on
# its own bound reports the bound, not an answer. The shortening here is
# about 1.08 ms, so the lower bound sits well below it, and we check.
bounds = (0.2 * textbook, textbook + 0.3)
result = optimize_delay(run_sequence, lambda s: -Ix_amplitude(s), bounds=bounds)
margin = 1e-3 * (bounds[1] - bounds[0])
assert bounds[0] + margin < result.x < bounds[1] - margin, \
    f"optimum {result.x:.4f} ms is on the search bound {bounds}; widen it"

print(f"\ntextbook 1/(4J)   = {textbook:.4f} ms  -> Ix = {Ix_amplitude(run_sequence(textbook)):.4f}")
print(f"optimized delay   = {result.x:.4f} ms  -> Ix = {Ix_amplitude(run_sequence(result.x)):.4f}")
print(f"shortened by {textbook - result.x:.4f} ms relative to textbook "
      f"(this lab's own empirical correction: 1.08 ms)")