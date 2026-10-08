"""
tutorial_profile_time_course.py -- the excitation profile in real time:
not just where every offset ends up, but how the whole profile gets there,
step by step through the pulse.

Every other profile in these tutorials is a snapshot taken AFTER the pulse.
Here the profile is recorded at every RF step, so it can be played back,
scrubbed with a slider, or drawn as one time-offset map. Two pulses, one
uncoupled 1H, no relaxation, offsets swept over +/-4 kHz:

    gausscasq5 90x, 3 ms   (amplitude-modulated, the Colab widget's default)
    hypsec, 4 ms, Q = 5    (adiabatic full passage, Bruker's defaults)

What the time course shows, as measured below rather than as claimed:

  gausscasq5 -- the final profile is a clean 90 degree band (|Mxy| > 0.9 out
  to +/-0.94 kHz), but that is not how it gets there. On resonance the
  magnetization goes all the way to Mz = -1.000 at t = 0.91 ms, a full
  inversion, and only then comes back up to the transverse plane. Halfway
  through (t = 1.5 ms) the whole band inside +/-1.06 kHz is inverted
  (Mz < -0.5). The cascade is built from Gaussians that rotate past 90 and
  come back; the end point alone hides that completely.

  hypsec -- each offset is inverted when the frequency sweep passes it, so
  the inversion travels across the band instead of growing from the centre:
  Mz first crosses zero at +2 kHz at 1.48 ms, on resonance at 2.00 ms (the
  middle of the pulse), and at -2 kHz at 2.44 ms. The final band
  (Mz < -0.9) is +/-2.10 kHz.

The RF phase is drawn under the RF amplitude, because amplitude alone hides
half of what the pulses do:

  gausscasq5 -- the RF phase only ever takes 0 or 180 degrees: a purely
  amplitude-modulated pulse whose negative lobes are 180-degree phase flips.

  hypsec -- the slope of the RF phase IS the sweep: its instantaneous
  frequency runs from +2.50 kHz at the start to -2.50 kHz at the end. Each
  offset inverts as the sweep passes it: inside +/-1 kHz, Mz crosses zero
  within 0.015 ms of the moment the instantaneous RF frequency equals that
  offset. Toward the band edge the passage is less adiabatic and the match
  loosens: at +2 kHz (where the sweep starts, with the RF still weak) Mz
  crosses 0.105 ms BEFORE the sweep arrives; at -2 kHz, 0.025 ms after.

Cost: the whole time course is one call, seq.run(M, df, trajectory=True),
and costs only ~30 % more than the final profile alone (401 offsets x 1000
steps: ~0.2 s). Every later frame is drawn from that stored array, which is
why scrubbing and animation can be instant: nothing is re-simulated.

Three outputs, all from the same trajectory:
  tutorial_figures/tutorial_profile_time_course.png  -- time-offset maps
      of Mz, the profile at four instants, and the RF amplitude and phase
      with those four instants marked (for print)
  tutorial_figures/tutorial_profile_time_course.gif  -- the profile played
      back over time beside the RF amplitude and phase (SAVE_GIF)
  an interactive window with a time slider (INTERACTIVE; needs a desktop
      matplotlib backend -- in a headless run it is simply not shown)

Checked on every execution (assertions at the bottom): the last frame of
the trajectory equals the ordinary final profile exactly; |M| stays 1 to
1e-12 at every step; the measured numbers quoted above, phase included.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.backend import NumpyBackend
from PULSIM.spin_system import gyro_ratio

PI = np.pi

N_OFFSETS = 401
BW_kHz = 4.0                        # sweep +/-BW_kHz
POINTS = 1000                       # RF steps per pulse
SAVE_GIF = True                     # ~20 s; the PNG is instant
INTERACTIVE = True                  # time-slider window (desktop backends only)

backend = NumpyBackend(Gamma=gyro_ratio('H'))
df = np.linspace(-BW_kHz, BW_kHz, N_OFFSETS)                     # kHz
M0 = np.tile(np.array([0.0, 0.0, 1.0]), (N_OFFSETS, 1)).T

gauss_shape = RFShape.create("gausscasq5", duration=3.0, points=POINTS)
hypsec_shape = RFShape.create("hypsec", duration=4.0, points=POINTS)
cases = {
    "gausscasq5 90x, 3 ms": PulseSequence([Pulse(gauss_shape, PI / 2, axis="x", backend=backend)]),
    # adiabatic: no flip angle -- amplitude comes from the sweep rate and Q
    "hypsec, 4 ms":        PulseSequence([Pulse(hypsec_shape, axis="x", backend=backend)]),
}

# ---- the whole time course: one call per pulse -----------------------------
runs = {}
for name, seq in cases.items():
    t0 = time.time()
    traj = seq.run(M0.copy(), df, trajectory=True)          # (n_steps + 1, 3, n_offsets)
    elapsed = time.time() - t0
    t = np.arange(traj.shape[0]) * seq[0].shape.dt          # ms; frame j = after j steps
    runs[name] = dict(seq=seq, traj=traj, t=t)
    print(f"{name:<22} {traj.shape[0] - 1} steps x {N_OFFSETS} offsets in {elapsed:.2f} s")

p_hyp = cases["hypsec, 4 ms"][0]
print(f"hypsec calibration: nu1_max {p_hyp.nu1_max:.3f} kHz, Q {p_hyp.realized_q:.2f}")

def band(mask):
    """Offsets where mask holds, as (lowest, highest), or None."""
    sel = df[mask]
    return (sel.min(), sel.max()) if sel.size else None

def frame_at(run, t_ms):
    return int(np.argmin(np.abs(run["t"] - t_ms)))

def rf_phase(seq, floor=1e-3):
    """Phase (0-360 deg) of the RF the spins feel: the calibrated envelope's own
    phase plus the pulse axis. NaN where the RF is off."""
    axis_deg = {"x": 0.0, "y": 90.0, "-x": 180.0, "-y": 270.0}
    out = []
    for p in seq:
        rf = np.asarray(p.calibrated_rf(), dtype=complex)
        ax = axis_deg[p.axis] if isinstance(p.axis, str) else np.degrees(float(p.axis))
        ph = (np.degrees(np.angle(rf)) + ax) % 360.0
        ph[np.abs(rf) < floor * np.abs(rf).max()] = np.nan
        out.append(ph)
    return np.concatenate(out)

def break_wraps(ph, jump=180.0):
    """Blank the sample after each +/-180 (or 360 -> 0) wrap, so a plotted phase
    line breaks there instead of drawing a false vertical jump."""
    ph = np.array(ph, dtype=float)
    d = np.abs(np.diff(ph))
    ph[1:][np.nan_to_num(d) > jump] = np.nan
    return ph

# ---- the numbers quoted in the docstring -----------------------------------
c = N_OFFSETS // 2                                          # on resonance
g = runs["gausscasq5 90x, 3 ms"]
k_min = int(np.argmin(g["traj"][:, 2, c]))
g_min_mz, g_min_t = g["traj"][k_min, 2, c], g["t"][k_min]
g_half = band(g["traj"][frame_at(g, 1.5), 2] < -0.5)
g_final = band(np.hypot(*g["traj"][-1, :2]) > 0.9)
print("\ngausscasq5 90x")
print(f"  on resonance, Mz reaches {g_min_mz:+.3f} at t = {g_min_t:.2f} ms (pulse is 3 ms)")
print(f"  at t = 1.5 ms, Mz < -0.5 over {g_half[0]:+.2f} .. {g_half[1]:+.2f} kHz")
print(f"  final |Mxy| > 0.9 over {g_final[0]:+.2f} .. {g_final[1]:+.2f} kHz")
g_rf_phases = np.unique(np.round(rf_phase(g["seq"])[np.isfinite(rf_phase(g["seq"]))], 6))
print(f"  RF phase takes only the values {g_rf_phases} deg (amplitude-modulated)")

h = runs["hypsec, 4 ms"]
first_neg = np.array([h["t"][np.argmax(h["traj"][:, 2, i] < 0)]
                      if (h["traj"][:, 2, i] < 0).any() else np.nan
                      for i in range(N_OFFSETS)])
h_final = band(h["traj"][-1, 2] < -0.9)
print("\nhypsec")
for f in (2.0, 1.0, 0.0, -1.0, -2.0):
    i = int(np.argmin(np.abs(df - f)))
    print(f"  offset {f:+.1f} kHz: Mz first < 0 at t = {first_neg[i]:.2f} ms")
print(f"  final Mz < -0.9 over {h_final[0]:+.2f} .. {h_final[1]:+.2f} kHz")
# the sweep: instantaneous RF frequency = d(phase)/dt / 2pi, in kHz (t in ms)
h_rf = np.asarray(h["seq"][0].calibrated_rf())
t_rf = np.arange(len(h_rf)) * h["seq"][0].shape.dt
f_inst = np.gradient(np.unwrap(np.angle(h_rf)), t_rf) / (2 * PI)
print(f"  instantaneous RF frequency: {f_inst[5]:+.2f} kHz at the start, "
      f"{f_inst[len(f_inst) // 2]:+.3f} at the middle, {f_inst[-5]:+.2f} at the end")
sweep_lag = {}
for f in (2.0, 1.0, 0.0, -1.0, -2.0):
    i = int(np.argmin(np.abs(df - f)))
    t_pass = float(np.interp(f, f_inst[::-1], t_rf[::-1]))   # f_inst falls with time
    sweep_lag[f] = first_neg[i] - t_pass
    print(f"  offset {f:+.1f} kHz: sweep passes at {t_pass:.3f} ms, Mz crosses 0 at "
          f"{first_neg[i]:.3f} ms (lag {sweep_lag[f]:+.3f} ms)")

# ---- figure 1: time-offset maps + four snapshots (for print) ---------------
fig, axs = plt.subplots(2, 3, figsize=(16, 8), width_ratios=[1, 1.1, 1.1])
for row, (name, run) in enumerate(runs.items()):
    traj, t = run["traj"], run["t"]

    ax = axs[row, 0]
    im = ax.imshow(traj[:, 2, :], aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-1, vmax=1, extent=[df[0] * 1000, df[-1] * 1000, t[0], t[-1]])
    ax.set(xlabel="offset (Hz)", ylabel="time into the pulse (ms)", title=f"{name}: Mz(offset, t)")
    ax.set_xticks(np.arange(-4000, 4001, 2000))
    fig.colorbar(im, ax=ax, label="Mz")
    if run is h:
        ax.plot(df * 1000, first_neg, "k--", lw=0.8, label="Mz first < 0")
        ax.legend(loc="upper right", fontsize=8)

    ax = axs[row, 1]
    T = t[-1]
    for frac, color in zip((0.25, 0.5, 0.75, 1.0), ("0.75", "0.55", "0.3", "k")):
        j = frame_at(run, frac * T)
        ax.plot(df * 1000, traj[j, 2], "-", color=color, lw=1.2, label=f"Mz, t = {t[j]:.2f} ms")
        ax.plot(df * 1000, np.hypot(traj[j, 0], traj[j, 1]), ":", color=color, lw=1.2)
    ax.axhline(0, color="0.85", lw=0.5)
    ax.set(xlabel="offset (Hz)", ylabel="magnetization",
           title="profile at four instants (solid Mz, dotted |Mxy|)")
    ax.legend(loc="lower right", fontsize=8)

    ax = axs[row, 2]
    seq = run["seq"]
    ax.plot(np.asarray(seq.time), np.abs(np.asarray(seq.rf)), color="0.35", lw=1)
    ax.set(xlabel="time (ms)", ylabel="|B1| (mT)", xlim=(0, T),
           title="RF amplitude (grey) and phase (blue)")
    ax_p = ax.twinx()
    ax_p.plot(np.asarray(seq.time), break_wraps(rf_phase(seq)), color="tab:blue", lw=1, alpha=0.7)
    ax_p.set_ylim(-5, 365)
    ax_p.set_yticks([0, 90, 180, 270, 360])
    ax_p.set_ylabel("RF phase (deg)", color="tab:blue")
    ax_p.tick_params(axis="y", labelcolor="tab:blue")
    for frac, color in zip((0.25, 0.5, 0.75, 1.0), ("0.75", "0.55", "0.3", "k")):
        ax.axvline(t[frame_at(run, frac * T)], color=color, ls="--", lw=1)

fig.tight_layout()
fig.savefig("tutorial_figures/tutorial_profile_time_course.png", dpi=150, bbox_inches="tight")

# ---- shared drawing for the animation and the slider -------------------------
def build_player(fig):
    """Profile panels (one per pulse) plus RF panels with a moving cursor.
    Returns update(frac) -> redraws every panel at fraction frac of each
    pulse. Only set_data / set_xdata: no re-simulation, no axes rebuilt."""
    gs = fig.add_gridspec(2, len(runs), height_ratios=[2.2, 1], hspace=0.45, wspace=0.45)
    artists = []
    for col, (name, run) in enumerate(runs.items()):
        seq, traj, t = run["seq"], run["traj"], run["t"]
        ax = fig.add_subplot(gs[0, col])
        lines = {lab: ax.plot([], [], lw=1.2, label=lab)[0] for lab in ("Mx", "My", "Mz")}
        lines["|Mxy|"] = ax.plot([], [], "k--", lw=1, label="|Mxy|")[0]
        ax.set(xlim=(df[0] * 1000, df[-1] * 1000), ylim=(-1.05, 1.05),
               xlabel="offset (Hz)", ylabel="magnetization")
        ax.axhline(0, color="0.85", lw=0.5)
        ax.legend(loc="lower right", fontsize=7)
        title = ax.set_title(name)

        ax_rf = fig.add_subplot(gs[1, col])
        rf = np.abs(np.asarray(seq.rf))
        ax_rf.plot(np.asarray(seq.time), rf, color="0.35", lw=1)
        cursor = ax_rf.axvline(0, color="r", lw=1.2)
        ax_rf.set(xlabel="time (ms)", ylabel="|B1| (mT)", xlim=(0, t[-1]))
        ax_rp = ax_rf.twinx()
        ax_rp.plot(np.asarray(seq.time), break_wraps(rf_phase(seq)), color="tab:blue",
                   lw=0.9, alpha=0.6)
        ax_rp.set_ylim(-5, 365)
        ax_rp.set_yticks([0, 180, 360])
        ax_rp.set_ylabel("RF phase (deg)", color="tab:blue")
        ax_rp.tick_params(axis="y", labelcolor="tab:blue")
        artists.append((run, lines, title, cursor, name))

    def update(frac):
        for run, lines, title, cursor, name in artists:
            traj, t = run["traj"], run["t"]
            j = min(int(round(frac * (len(t) - 1))), len(t) - 1)
            M = traj[j]
            x = df * 1000
            lines["Mx"].set_data(x, M[0]); lines["My"].set_data(x, M[1])
            lines["Mz"].set_data(x, M[2]); lines["|Mxy|"].set_data(x, np.hypot(M[0], M[1]))
            title.set_text(f"{name}   t = {t[j]:.2f} ms")
            cursor.set_xdata([t[j], t[j]])
    return update

# ---- figure 2: the animation -------------------------------------------------
if SAVE_GIF:
    fig_ani = plt.figure(figsize=(11, 6.5))
    update = build_player(fig_ani)
    fracs = np.linspace(0, 1, 101)                          # 1 % of each pulse per frame
    ani = FuncAnimation(fig_ani, lambda k: update(fracs[k]), frames=len(fracs), interval=60)
    t0 = time.time()
    ani.save("tutorial_figures/tutorial_profile_time_course.gif", writer="pillow", fps=15, dpi=80)
    print(f"\nwrote the GIF ({len(fracs)} frames) in {time.time() - t0:.1f} s")
    plt.close(fig_ani)

# ---- figure 3: scrub through the pulse with a slider ---------------------------
if INTERACTIVE:
    fig_live = plt.figure(figsize=(11, 7.2))
    update = build_player(fig_live)
    fig_live.subplots_adjust(bottom=0.15)
    slider = Slider(fig_live.add_axes([0.15, 0.04, 0.7, 0.03]), "fraction of pulse",
                    0.0, 1.0, valinit=1.0)
    slider.on_changed(lambda v: (update(v), fig_live.canvas.draw_idle()))
    update(1.0)

# ---- checks: re-run every execution ---------------------------------------------
for name, run in runs.items():
    final = run["seq"].run(M0.copy(), df)
    assert np.array_equal(run["traj"][-1], final), f"{name}: last frame != final profile"
    norm = np.linalg.norm(run["traj"], axis=1)
    assert np.abs(norm - 1).max() < 1e-12, f"{name}: |M| drifted by {np.abs(norm - 1).max():.1e}"
assert g_min_mz < -0.999 and abs(g_min_t - 0.91) < 0.01
assert np.allclose(g_half, (-1.06, 1.06)) and np.allclose(g_final, (-0.94, 0.94))
assert abs(first_neg[c] - 2.0) < 0.01                       # inverts at the pulse midpoint
fn = first_neg[np.abs(df) <= 2.0]
assert np.all(np.diff(fn) <= 1e-9), "hypsec inversion should travel monotonically across the band"
assert np.allclose(h_final, (-2.10, 2.10))
assert set(g_rf_phases) <= {0.0, 180.0}, "gausscasq5 should be purely amplitude-modulated"
assert abs(f_inst[5] - 2.5) < 0.01 and abs(f_inst[-5] + 2.5) < 0.01
assert all(abs(sweep_lag[f]) < 0.015 for f in (1.0, 0.0, -1.0)), sweep_lag
assert all(abs(sweep_lag[f]) < 0.11 for f in (2.0, -2.0)), sweep_lag
print("\nchecks passed: last frame == final profile exactly, |M| = 1 to 1e-12, quoted numbers "
      "(RF phase included) reproduced")

plt.show()
