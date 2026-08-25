"""
tutorial_delay_optimization.py -- why the textbook 1/(4J) INEPT delay is
wrong for real (finite-duration) shaped pulses, quantified per-pulse and
then corrected exactly, using this lab's real eb2try/eb2x refocusing
element (see tutorial_HN_shaped_refocusing.py) as the test case.
"""

import numpy as np

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import Delay, IdealPulse, RawShapedPulseSegment, LiouvilleSequence, ShapePulseSegment
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_diagnostics import effective_coupling_generator, optimize_delay
from PULSIM.pulse_oo import Pulse
from PULSIM.backend import NumpyBackend

PI, twoPI = np.pi, 2 * np.pi

J_HZ = 92.0
tp_ms = 1.5
rfPow_rad_ms = (twoPI * 2730.78242) / 1000.0

shape1 = RFShape.create("file", path="wave/eb2try_1.5m_ofs0Hz.500", duration=tp_ms)
shape2 = RFShape.create("file", path="wave/eb2x_1.5m_ofs0Hz.500", duration=tp_ms)
RF1 = np.conj(shape1.envelope()) * rfPow_rad_ms / 100.0
RF2 = np.conj(shape2.envelope()) * rfPow_rad_ms / 100.0
seg1 = RawShapedPulseSegment(RF1, shape1.dt, channel='H')
seg2 = RawShapedPulseSegment(RF2, shape2.dt, channel='H')

GAMMA_H = gyro_ratio('H')
GAMMA_C = gyro_ratio('13C')
GAMMA_N = gyro_ratio('15N')

shape = RFShape.create("gausscasq5", duration=tp_ms, points=500)
pulse = Pulse(shape, PI / 2, axis="x", backend=NumpyBackend(Gamma=GAMMA_H))
seg1 = ShapePulseSegment(pulse)

# -- goal 2: per-pulse effective J-evolution time --------------------------
for name, rf, dt in [("eb2try", RF1, shape1.dt), ("eb2x", RF2, shape2.dt)]:
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
result = optimize_delay(run_sequence, lambda s: -Ix_amplitude(s), bounds=(textbook - 1.0, textbook + 0.3))

print(f"\ntextbook 1/(4J)   = {textbook:.4f} ms  -> Ix = {Ix_amplitude(run_sequence(textbook)):.4f}")
print(f"optimized delay   = {result.x:.4f} ms  -> Ix = {Ix_amplitude(run_sequence(result.x)):.4f}")
print(f"shortened by {textbook - result.x:.4f} ms relative to textbook "
      f"(this lab's own empirical correction: 1.08 ms)")