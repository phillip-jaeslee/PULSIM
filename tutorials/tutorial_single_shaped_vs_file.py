"""
tutorial_single_shaped_vs_file.py -- the same pulse, defined analytically
and read back from a shape file, must give the same physics.

A single 90x GaussCascadeQ5 (1 ms) acts on a 1H-15N pair (J = 92 Hz,
1J(NH)) starting from Iz(H), swept across resonance offset. The pulse is
used twice:

  * analytic:  RFShape.create("gausscasq5", ...)
  * from file: the same waveform written to a shape file
               (amplitude in percent, phase in degrees -- the columns the
               importer reads) and loaded back with FileShape.

Any difference between the two is an import error: resampling, scaling,
phase sign, or the flip-angle calibration of an imported shape. The file is
written here, to a temporary directory, so the tutorial needs no vendor
shape files; the lab's own vendor GaussCascadeQ5 agrees with the analytic
shape to 0.9 % in amplitude, which is a difference between two designs,
not an import error.
"""

import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.spin_operators import Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.liouville import LiouvilleSequence, ShapePulseSegment
from PULSIM.rf_shape import RFShape, FileShape
from PULSIM.pulse_oo import Pulse
from PULSIM.backend import NumpyBackend

PI, twoPI = np.pi, 2 * np.pi

J_HZ = 92.0                         # Hz, 1J(NH)
tp_ms = 1.0                         # shaped pulse length, ms
N = 201                             # offsets to sweep (odd: centre is on resonance)
W_hz = np.linspace(-1, 1, N) * 4000.0

backend = NumpyBackend(Gamma=gyro_ratio('H'))


def write_shape_file(path, envelope, title):
    """Amplitude (% of peak) and phase (deg) columns, FileShape's convention:
    it rebuilds the waveform as amplitude * exp(-i * phase)."""
    amp = np.abs(envelope) / np.abs(envelope).max() * 100.0
    phase = (-np.degrees(np.angle(envelope))) % 360.0
    lines = [f"##TITLE= {title}", f"##NPOINTS= {len(envelope)}", "##XYPOINTS= (XY..XY)"]
    lines += [f"{a:.6e}, {p:.6e}" for a, p in zip(amp, phase)]
    lines.append("##END=")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


analytic = RFShape.create("gausscasq5", duration=tp_ms, points=1000)

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "gausscasq5_pulsim.jhl")
    write_shape_file(path, analytic.envelope(), "GaussCascadeQ5 written by PULSIM")
    from_file = FileShape(path=path, duration=tp_ms)
    from_file.envelope()                                 # read while the file exists

pulse_analytic = Pulse(analytic, PI / 2, axis="x", backend=backend)
pulse_file = Pulse(from_file, PI / 2, axis="x", backend=backend)

env_a = analytic.envelope() / np.abs(analytic.envelope()).max()
env_f = from_file.envelope() / np.abs(from_file.envelope()).max()
print(f"waveform:  max |file - analytic| = {np.abs(env_f - env_a).max():.1e}")
print(f"90 deg calibration:  analytic {pulse_analytic.nu1_max:.6f} kHz, "
      f"file {pulse_file.nu1_max:.6f} kHz")


def run(pulse, off_hz):
    ss = SpinSystem(nuclei=['H', '15N'], offsets=[twoPI * off_hz / 1000.0, 0.0],
                    couplings={(0, 1): J_HZ})
    return LiouvilleSequence([ShapePulseSegment(pulse)], ss).propagate(embed(Iz(), 0, 2))


OPS = {'Ix': embed(Ix(), 0, 2), 'Iy': embed(Iy(), 0, 2), 'Iz': embed(Iz(), 0, 2),
       'IxSz': product_operator(Ix(), 0, Iz(), 1, 2),
       'IySz': product_operator(Iy(), 0, Iz(), 1, 2),
       'IzSz': product_operator(Iz(), 0, Iz(), 1, 2)}


def extract(sigma):
    return {k: np.trace(sigma @ A).real / np.trace(A @ A).real for k, A in OPS.items()}


keys = list(OPS)
res_a = {k: np.zeros(N) for k in keys}
res_f = {k: np.zeros(N) for k in keys}
for n, off in enumerate(W_hz):
    ra, rf = extract(run(pulse_analytic, off)), extract(run(pulse_file, off))
    for k in keys:
        res_a[k][n], res_f[k][n] = ra[k], rf[k]

worst = max(np.abs(res_f[k] - res_a[k]).max() for k in keys)
print(f"offset sweep:  max |file - analytic| over all six terms = {worst:.1e}")

colors = ['red', 'blue', 'green', 'orange', 'cyan', 'olive']
fig, ax = plt.subplots(1, 2, sharex=True, constrained_layout=True)
fig.set_size_inches(11, 4.5, forward=True)

ax[0].axhline(y=0, color='lightgray', linestyle='-')
for k, c in zip(keys, colors):
    ax[0].plot(W_hz, res_a[k], '-', color=c, label=f"{k} analytic")
    ax[0].plot(W_hz[::10], res_f[k][::10], 'o', color=c, markersize=3)
ax[0].set_title(f'90x GaussCascadeQ5, J = {J_HZ:.0f} Hz: analytic (lines), file (dots)')
ax[0].set_xlabel('offset (Hz)')
ax[0].legend(fontsize=8)
ax[0].invert_xaxis()

for k, c in zip(keys, colors):
    ax[1].plot(W_hz, res_f[k] - res_a[k], '-', color=c, label=k)
ax[1].set_title('file - analytic (import error only)')
ax[1].set_xlabel('offset (Hz)')
ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax[1].legend(fontsize=8)

plt.savefig("tutorial_figures/tutorial_single_shaped_vs_file.png")
plt.show()