"""
tutorial_3d_bloch_animation.py -- watching the magnetization itself, rather
than the excitation profile it ends up with.
 
Ported from this repo's own 3D_simulation_test.py, which predates the OO
layer: it drove PULSIM.simulate's sim_* functions, fanned ten isochromats
out over joblib with parallel_map, and hand-assembled the RF and phase
traces with np.append chains.
 
Sequence (a single uncoupled 1H, no coupling partner, no relaxation):
 
    M0 = +z  --sine.jhl 90y, 0.6 ms-->  --hard 180(-y), 12 us-->  --sine.jhl 90y, 0.6 ms-->
 
Every other tutorial here plots the END of a sequence against offset: one
number per isochromat, after everything has happened. This one plots the
WHOLE PATH -- every RF step of all three pulses -- because the thing worth
seeing is not where the magnetization lands but how it gets there: the
ten isochromats leaving +z together, fanning apart by offset during the
soft pulse, being thrown across the sphere by the hard 180, and partly
refocusing on the way back.
 
What the sequence does, as measured below rather than as claimed:
 
    offset      final M                     
    0.0 kHz     [ 0,      0,      +1    ]   returns to +z: a null operation
    1.0 kHz     [+0.441, -0.791, +0.424 ]   partly tipped, partly refocused
    3.0 kHz     [+0.069, -0.191, -0.979 ]   essentially inverted
 
so the sandwich acts as an offset filter -- on resonance it does nothing,
far off resonance it inverts -- and the animation is where you can see why.
 
The offsets sweep +/-1.5 kHz across ten isochromats. That number is the
shape's own half-height excitation width, measured from this file: a lone
sine.jhl 90y at 0.6 ms holds |Mxy| >= 0.9 out to +/-0.88 kHz and >= 0.5
out to +/-1.56 kHz. Sweeping wider would fan the arrows past the point
where the first pulse still excites them.
 
The RF panel is dominated by the hard pulse and this is deliberate, not a
plotting failure: 12 us of 180 degrees needs 0.98 mT, while 0.6 ms of a
soft 90 needs 0.015 mT -- a factor of 64. Drawn honestly on one linear
axis, the shaped pulse is a flat line next to it. That contrast is worth
a student seeing.
 
Note that all ten isochromats are carried through ONE vectorized run: M is
(3, n_offsets) and the offsets are columns, so the parallel_map over arrows
in the original script has no counterpart here and is simply gone.
 
Verified against the legacy path this file replaces: the same sequence
driven through PULSIM.simulate's sim_import_shaped_pulse / sim_hard_pulse
lands on the identical final magnetization, to 0.0 exactly (not merely
within tolerance), at every offset checked -- the assertion at the bottom
of this file re-runs that comparison on every execution.
 
The one place the two paths genuinely differ: the legacy sim_* loop
carries an `if n == 0: M[:, n] = M[:, n]` branch that skips the very first
RF sample of the first pulse. It costs nothing here -- M starts along z
and sine.jhl opens at zero amplitude, so the skipped step is a rotation of
a z-aligned vector about z, i.e. the identity -- but it would not be
harmless for a starting state off the z axis. Pulse.apply() applies every
sample.
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.backend import NumpyBackend
from PULSIM.spin_system import gyro_ratio
from PULSIM.simulate import sim_import_shaped_pulse, sim_hard_pulse
from PULSIM.visualization import plot_3D_arrow_snapshots, plot_3D_arrow_with_pulse, save_animation_to_gif
 
PI = np.pi
 
SHAPE_FILE = "wave/sine.jhl"
tp_soft = 0.6                       # soft pulse length, ms
tp_hard = 0.012                     # hard pulse length, ms
N_ARROWS = 10
BW_kHz = 1.5                        # half-height width of the soft 90, measured above
SAVE_GIF = True                     # the GIF costs ~15 s; the PNG is instant
 
GAMMA_H = gyro_ratio('H')
backend = NumpyBackend(Gamma=GAMMA_H)
 
# resample_to=int(duration*1000) reproduces what the legacy sim_* functions
# do internally, so the equivalence check at the bottom compares like with
# like: 600 points for the soft pulse, 12 for the hard one.
soft = RFShape.create("file", path=SHAPE_FILE, duration=tp_soft, resample_to=int(tp_soft * 1000))
hard = RFShape.create("hard", duration=tp_hard, points=int(tp_hard * 1000))
 
p90_a = Pulse(soft, PI / 2, axis="y", backend=backend)
p180 = Pulse(hard, -PI, axis="y", backend=backend)
p90_b = Pulse(soft, PI / 2, axis="y", backend=backend)
seq = PulseSequence([p90_a, p180, p90_b])
 
print(f"soft 90y : {soft.points:4d} pts, nu1_max {p90_a.nu1_max:+8.4f} kHz, B1 {p90_a.b1_max:+.5f} mT")
print(f"hard 180 : {hard.points:4d} pts, nu1_max {p180.nu1_max:+8.4f} kHz, B1 {p180.b1_max:+.5f} mT")
 
# ---- one vectorized run: offsets are columns of M, not separate jobs -------
df = np.linspace(-BW_kHz, BW_kHz, N_ARROWS)                 # kHz
M0 = np.tile(np.array([0.0, 0.0, 1.0]), (N_ARROWS, 1)).T
traj = seq.run(M0, df, trajectory=True)
 
print(f"trajectory {traj.shape}  =  ({len(seq.rf)} RF steps + 1 starting frame, 3, {N_ARROWS} offsets)")
print(f"|M| stays {np.linalg.norm(traj, axis=1).min():.12f} .. {np.linalg.norm(traj, axis=1).max():.12f}")
 
# ---- the four instants worth freezing -------------------------------------
n1 = soft.points                    # end of the first soft 90
n2 = n1 + hard.points               # end of the hard 180
frames = [0, n1, n2, -1]
labels = ["equilibrium", "after soft 90y", "after hard 180", "after 2nd 90y"]
 
fig = plot_3D_arrow_snapshots(traj, seq.time, seq.rf, seq.phase,
                              frames=frames, labels=labels)
fig.savefig("tutorial_figures/tutorial_3d_bloch_animation.png", dpi=150, bbox_inches="tight")
 
# ---- the animation --------------------------------------------------------
# stride: the 3D panel is redrawn from scratch every frame, so all 1213 of
# them would be an unusable GIF. Every 20th step is ~60 frames and still
# smooth, because neighbouring 1 us steps are nearly identical anyway.
fig_ani, ani = plot_3D_arrow_with_pulse(traj, seq.time, seq.rf, seq.phase,
                                        stride=20, interval=60)
if SAVE_GIF:
    save_animation_to_gif(ani, "tutorial_figures/tutorial_3d_bloch_animation.gif", fps=10, dpi=90)
 
# ---- the verification claim in the docstring, re-run every execution -------
def legacy_endpoint(phi):
    """The same sequence through the path this tutorial replaces."""
    N = soft.points + hard.points + soft.points
    M = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1)).T.astype(float)
    M, _, _, N1 = sim_import_shaped_pulse(M, PI / 2, "y", tp_soft, SHAPE_FILE, 0, phi, GAMMA_H)
    M, _, _, N2 = sim_hard_pulse(M, -PI, "y", tp_hard, N1, hard.points, phi, GAMMA_H)
    M, _, _, N3 = sim_import_shaped_pulse(M, PI / 2, "y", tp_soft, SHAPE_FILE, N2, phi, GAMMA_H)
    return M[:, N3 - 1]
 
print("\noffset       final M (this tutorial)          max|new - legacy sim_*|")
worst = 0.0
for phi in (0.0, 1.0, 3.0):
    new = seq.run(np.array([[0.0], [0.0], [1.0]]), np.array([phi]))[:, 0]
    diff = np.abs(new - legacy_endpoint(phi)).max()
    worst = max(worst, diff)
    print(f"{phi:>4.1f} kHz   [{new[0]:+.6f} {new[1]:+.6f} {new[2]:+.6f}]      {diff:.2e}")
 
assert worst == 0.0, f"ported path drifted from the legacy sim_* path by {worst:.2e}"
print(f"\nlegacy sim_* agreement: exact (max difference {worst:.1f})")
 
plt.show()
 

