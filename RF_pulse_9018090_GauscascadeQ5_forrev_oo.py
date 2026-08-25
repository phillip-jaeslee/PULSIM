"""
RF_pulse_9018090_GauscascadeQ5_forrev_oo.py

Object-oriented rewrite of RF_pulse_9018090_GauscascadeQ5_forrev.py.
Same three-pulse sequence (sine -> hard -> sine), same physics, same plot.
Proven equivalent to the original in Step 10 of the refactor.

Replaces:
  the six-argument torch_pulse.* calls         -> Pulse
  RF_temp / t_max_temp / Ns bookkeeping        -> PulseSequence.rf / .time / .phase
  three np.append chains                       -> gone
"""


import numpy as np

from PULSIM.rf_shape import RFShape
from PULSIM.backend import TorchBackend
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from visualization import plot_pulse, save_figure

Gamma = 42.58   # kHz/mT -- matches the original script exactly
BW = 6          # kHz
N = 1000
file_path = "wave/sine.jhl"

df = np.linspace(-BW/2, BW/2, num=N)
M0 = np.tile(np.array([0, 0, 1]), (N, 1)).T.astype(float)

backend = TorchBackend(Gamma=Gamma)
sine_shape = RFShape.create("file", path=file_path, duration=0.6)
hard_shape = RFShape.create("hard", duration=0.02, points=N)

seq = PulseSequence([
    Pulse(sine_shape, flip=np.pi /2, axis="x", backend=backend),
    Pulse(hard_shape, flip=np.pi, axis="x", backend=backend),
    Pulse(sine_shape, flip=np.pi/2, axis="x", backend=backend),
])

print("running 3-pulse sequence")
M = seq.run(M0.copy(), df)

fig = plot_pulse(M, np.abs(seq.rf), seq.phase, df, seq.time,
                  label_Mx=False, label_My=False, label_Mxy=True)

save_figure(fig, save=True, file_path="test_oo")