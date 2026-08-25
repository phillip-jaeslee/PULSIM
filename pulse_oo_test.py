import numpy as np
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.backend import NumpyBackend
from input_parameter import gyro_ratio

hard = RFShape.create("hard", duration=0.02, points=1000)
Gamma_H = gyro_ratio("H")
Gamma_C = gyro_ratio("13C")

p_H = Pulse(hard, flip=np.radians(90), backend=NumpyBackend(Gamma=Gamma_H))
p_C = Pulse(hard, flip=np.radians(90), backend=NumpyBackend(Gamma=Gamma_C))

amp_H = np.abs(p_H.calibrated_rf()).mean()
amp_C = np.abs(p_C.calibrated_rf()).mean()

print(f"RF amplitude, 1H:  {amp_H:.4f}")
print(f"RF amplitude, 13C: {amp_C:.4f}")
print(f"ratio (13C/1H):    {amp_C/amp_H:.4f}  (expect {Gamma_H/Gamma_C:.4f}, i.e. NOT 1.0)")