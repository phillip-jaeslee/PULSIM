"""
Checks that PulseSequence.run() correctly chains multiple Pulse.apply() calls:
feeding M from one pulse into the next, and aggregating .rf/.time/.phase across
pulses. The reference here is calling .apply() on each Pulse manually, in a loop,
with no PulseSequence involved -- an independent check of PulseSequence's own
chaining logic, not a physics test (bloch_rotate's correctness is covered
separately by test_bloch_rotate_correctness.py / test_torch_bloch_rotate_correctness.py).
"""

import numpy as np
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.backend import TorchBackend

Gamma = 42.58
BW = 6
N = 1000
file_path = "wave/sine.jhl"

df = np.linspace(-BW/2, BW/2, num=N)
M0 = np.tile(np.array([0, 0, 1]), (N, 1)).T.astype(float)

backend = TorchBackend(Gamma=Gamma)
sine_shape = RFShape.create("file", path=file_path, duration=0.6)
hard_shape = RFShape.create("hard", duration=0.02, points=N)

# ---- manual reference: call .apply() on each Pulse directly, no PulseSequence ----
M_manual = M0.copy()
for p in [Pulse(sine_shape, flip=np.pi/2, axis="x", backend=backend),
          Pulse(hard_shape, flip=np.pi,   axis="x", backend=backend),
          Pulse(sine_shape, flip=np.pi/2, axis="x", backend=backend)]:
    M_manual = p.apply(M_manual, df)

# ---- PulseSequence path ----
seq = PulseSequence([
    Pulse(sine_shape, flip=np.pi/2, axis="x", backend=backend),
    Pulse(hard_shape, flip=np.pi,   axis="x", backend=backend),
    Pulse(sine_shape, flip=np.pi/2, axis="x", backend=backend),
])
M_new = seq.run(M0.copy(), df)

diff = np.abs(M_manual - M_new)
print("max |difference| between manual Pulse.apply() chain and PulseSequence:", diff.max())
print("agree (atol=1e-4, float32 precision over 3 chained pulses):", np.allclose(M_manual, M_new, atol=1e-4))