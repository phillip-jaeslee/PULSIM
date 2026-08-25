import numpy as np
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.backend import NumpyBackend

M0 = np.array([[0.0], [0.0], [1.0]])
df = np.array([0.0])
backend = NumpyBackend()

hard90 = RFShape.create("hard", duration=0.02, points=1000)
p1 = Pulse(hard90, flip=np.radians(90), axis="x", backend=backend)
p2 = Pulse(hard90, flip=np.radians(90), axis="x", backend=backend)

seq = PulseSequence([p1, p2])
M_final = seq.run(M0.copy(), df)
mx, my, mz = M_final[:, 0]
mxy = np.hypot(mx, my)
print(f"two 90-deg pulses -> Mxy={mxy:.5f} Mz={mz:.5f}  (expect Mxy=0, Mz=-1)")

print("len(seq):", len(seq))
print("seq.rf length:", seq.rf.shape, "(expect (2000,) = 1000+1000)")
print("seq.time monotonic:", np.all(np.diff(seq.time) > 0))
print("second pulse starts right at 0.02:", np.isclose(seq.time[1000], 0.02, atol=2e-5))

for i, p in enumerate(seq):   # __iter__ at work
    print(f"iterated pulse {i}: flip={np.degrees(p.flip):.0f} deg")

seq = PulseSequence([p1, p2])
print(seq[0])          # should print a Pulse, not raise TypeError
print(seq[0] is p1)    # should be True