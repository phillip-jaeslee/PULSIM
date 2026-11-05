"""
tutorial_bloch_siegert.py -- the Bloch-Siegert shift: what a selective pulse
does to the spins it does NOT excite, and three ways to remove it.

Setting (a single uncoupled 13C, Bloch engine, no relaxation):
    carrier on C' (176 ppm); a C'-selective Q3 180x, 0.35 ms, at the carrier.
    Calpha (40-70 ppm) sits 16-21 kHz upfield at 150 MHz -- far outside the
    Q3 band. The textbook says the pulse leaves Calpha alone. It does not.

WHY
In the rotating frame a spin at offset Delta from the RF sees the vector sum
of Delta (along z) and the RF w1(t) (in the xy plane), so during the pulse it
precesses at
        w_eff = sqrt(Delta^2 + w1^2)  ~  Delta + w1^2 / (2 Delta)    (|Delta| >> w1)
The field barely tilts away from z, so nothing is excited -- but it is a little
STRONGER than Delta, and the spin turns a little further than it would have
during a plain delay. The extra angle is the Bloch-Siegert phase:
        phi_BS = int [ sqrt(Delta^2 + w1^2) - |Delta| ] dt  ~  (1 / 2 Delta) int w1^2 dt
Three properties follow from the formula and are checked below:
    * it goes as w1^2: the RF phase (x, y, -x) does not matter, so phase
      cycling cannot remove it;
    * it has the sign of Delta: spins on the two sides of the carrier turn
      in opposite directions;
    * for a fixed flip angle int w1^2 dt ~ 1/tau: shorter pulses do more harm.
(Bloch and Siegert's original 1940 shift is the same formula with
Delta = 2 w0: the counter-rotating half of the linearly polarized RF.)

HOW IT IS MEASURED
Run the same starting magnetization (+x) through the pulse, and through a
delay of the same length. The delay carries all the ordinary offset
precession, so the phase difference of the two results is phi_BS alone.

PART 1 -- the phase itself (figure tutorial_bloch_siegert_shift.png)
    (a) Calpha at -18 kHz: |Mxy| stays 0.998, but it has turned by -98 deg.
    (b) phi_BS against offset: a 1/Delta curve, odd about the carrier.
        At -18 kHz: simulation -97.9, w1^2/(2 Delta) -96.4, effective-field
        integral -92.8 deg. The integral assumes the field direction changes
        slowly (adiabatically); Q3 changes sign (envelope minimum -0.93), and
        at each sign change the tilt flips abruptly, which it does not capture.
    (c1) RF phase x, y, -x: identical curves.
    (c2) 0.35 / 0.7 / 1.4 ms: -98 / -48 / -24 deg at Calpha (~ 1/tau).

PART 2 -- removing it (figure tutorial_bloch_siegert_compensation.png)
    Method 1, the same pulse at the mirror position of an echo:
        C' pulse -- d -- 180(Calpha) -- d -- the identical C' pulse
      The 180 inverts the phase gathered before it, so the first pulse's
      phi_BS becomes -phi_BS and the second adds +phi_BS. Identical pulses give
      identical phi_BS(Delta), so this cancels at EVERY offset, with no
      calculation. Residual over 40-70 ppm: < 1 deg (one pulse alone: +80..+110).
    Methods 2 + 3, a delay and a phase. Over the Calpha band, linearize
        phi_BS(Delta) ~ A / Delta ~ 2A/Dc  -  (A/Dc^2) Delta       (Dc = band centre)
      The term in Delta looks exactly like free precession (360 deg Delta t),
      so the pulse acts like a short extra (here negative) delay:
        slope -5.38 deg/kHz / 360 deg/(kHz ms) = -14.9 us
      -> LENGTHEN the following delay by 15 us (method 2). What is left is the
      constant 2A/Dc ~ 2 phi_BS(Dc) = -195 = +165 deg (twice phi_BS, because the
      delay fix itself adds -97 deg at the band centre), which no delay can
      touch -> shift the PHASE of the next pulse (or the receiver) by +165 deg
      (method 3). Residual: < 2 deg, the curvature of 1/Delta. The corrected
      sequence is run for real in panel (c): after a 90 deg readout its Mz
      matches the no-C'-pulse reference to 0.03.
    (Method 1 is exact but needs room for one more pulse; methods 2 + 3 add
    nothing to the sequence but are a linear fit, so they degrade as the band
    to be corrected widens.)

Both figures draw the pulse sequences above the results. They are drawn by
PULSIM.sequence_figure.draw_sequence from the very element lists that are run
(see draw()), to scale in ms, so a diagram cannot disagree with its curve.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.rf_shape import RFShape
from PULSIM.pulse_oo import Pulse
from PULSIM.pulse_sequence import PulseSequence
from PULSIM.backend import NumpyBackend
from PULSIM.liouville import Delay, IdealPulse, ShapePulseSegment, LiouvilleSequence, _axis_phase
from PULSIM.spin_system import SpinSystem, gyro_ratio
from PULSIM.sequence_figure import draw_sequence

GAMMA_C = gyro_ratio('13C')
SHAPE, TAU = "gausscasq3", 0.35      # C'-selective Q3 180, ms
CA = -18.0                           # Calpha offset from C', kHz (120 ppm at 150 MHz)
BAND = (-21.0, -16.0)                # Calpha band (40-70 ppm), kHz

os.makedirs("tutorial_figures", exist_ok=True)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6})
DEG_TICKS = np.arange(-180, 181, 60)


# -- building blocks ---------------------------------------------------------
def q3(tau=TAU, phase="x"):
    """The C'-selective 180. phase: "x", "y", or a float in rad."""
    shape = RFShape.create(SHAPE, duration=tau, points=500)
    return Pulse(shape, np.pi, axis=phase, backend=NumpyBackend(Gamma=GAMMA_C))


def hard(flip_deg, phase):
    """1 us hard pulse (~ 250-500 kHz): ideal for every offset used here."""
    shape = RFShape.create("hard", duration=0.001, points=10)
    return Pulse(shape, np.radians(flip_deg), axis=phase, backend=NumpyBackend(Gamma=GAMMA_C))


def run(elements, df, start=(1, 0, 0)):
    """Final (3, n_offsets) M of a PulseSequence started from `start` at every offset."""
    M0 = np.tile(np.asarray(start, float)[:, None], (1, df.size))
    return PulseSequence(elements).run(M0, df)


def draw(elements, ax, label, t_max):
    """Draw the SAME element list that run() propagates, so the diagram cannot
    disagree with the simulation. draw_sequence reads a LiouvilleSequence, so
    each Bloch element is wrapped: a Delay as itself, a shaped Pulse as a
    ShapePulseSegment, and the 1 us hard pulses as IdealPulses (they are ideal
    for every offset here, and a 1 us bar would be invisible to scale).
    t_max: a common time axis for diagrams that are compared with each other."""
    segs = []
    for el in elements:
        if isinstance(el, Delay):
            segs.append(el)
        elif el.shape.duration <= 0.002:                       # the hard() pulses
            segs.append(IdealPulse(el.flip, phase=_axis_phase(el.axis), channel="13C"))
        else:
            segs.append(ShapePulseSegment(el))
    draw_sequence(LiouvilleSequence(segs, SpinSystem(nuclei=["13C"], offsets=[0.0])),
                  ax=ax, to_scale=True)
    for t in list(ax.texts):
        if t.get_style() == "italic":                          # the "drawn at minimum width" note
            t.remove()
        else:
            t.set_fontsize(6)
    ax.set_xlim(-0.10 * t_max, 1.04 * t_max)
    ax.set_xlabel("")
    ax.tick_params(labelsize=6)
    ax.set_title(label, loc="left", fontsize=7)


def rel_phase(Ma, Mb):
    """Transverse phase of Ma relative to Mb, deg, in (-180, 180]."""
    return np.degrees(np.angle((Ma[0] + 1j * Ma[1]) / (Mb[0] + 1j * Mb[1])))


def bs_phase(pulse, df):
    """phi_BS (deg): the pulse run minus a delay of the same length, both from +x."""
    return rel_phase(run([pulse], df), run([Delay(pulse.shape.duration)], df))


def unwrap_from_edges(phi, df):
    """Unwrap each side of the carrier starting far out, where phi_BS is small."""
    out = np.array(phi, float)
    neg, pos = df < 0, df > 0
    out[neg] = np.degrees(np.unwrap(np.radians(phi[neg])))
    out[pos] = np.degrees(np.unwrap(np.radians(phi[pos][::-1])))[::-1]
    return out


def excited(pulse, df, tol=0.01):
    """Offsets the pulse really excites (from +z, |Mz| drops): not Bloch-Siegert."""
    return run([pulse], df, start=(0, 0, 1))[2] < 1 - tol


def bs_theory(pulse, df):
    """Effective-field integral and w1^2/(2 Delta) approximation, deg, signed like Delta."""
    nu1 = GAMMA_C * np.abs(pulse.calibrated_rf())      # kHz
    dt = pulse.shape.dt                                 # ms
    D = np.abs(df)[:, None]
    integral = 360 * np.sum((np.sqrt(D**2 + nu1**2) - D) * dt, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):   # Delta = 0: the formula diverges
        approx = np.sign(df) * 360 * np.sum(nu1**2 * dt) / (2 * np.abs(df))
    return np.sign(df) * integral, approx


# =============================================================================
# PART 1 -- the Bloch-Siegert phase
# =============================================================================
df = np.linspace(-30, 30, 1201)                      # kHz
p0 = q3()
iCA = np.argmin(np.abs(df - CA))

M_pulse, M_delay = run([p0], df), run([Delay(TAU)], df)
phi = unwrap_from_edges(rel_phase(M_pulse, M_delay), df)
integral, approx = bs_theory(p0, df)
exc = excited(p0, df)

print(f"Q3 180, {TAU} ms: nu1_max = {p0.nu1_max:.3f} kHz, excites "
      f"{df[exc].min():.1f} .. {df[exc].max():.1f} kHz")
print(f"Calpha ({CA:g} kHz): |Mxy| = {np.hypot(*M_pulse[:2, iCA]):.4f}, "
      f"phi_BS simulated {phi[iCA]:.1f}, w1^2/(2 Delta) {approx[iCA]:.1f}, "
      f"effective-field integral {integral[iCA]:.1f} deg")

fig = plt.figure(figsize=(7, 6.4))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1.25, 1.25], height_ratios=[0.42, 1, 1],
                      hspace=0.8, wspace=0.38)
plot_axes = []

# the two runs whose phase difference is phi_BS
top = gs[0, :].subgridspec(1, 2, wspace=0.25)
draw([p0], fig.add_subplot(top[0, 0]), "run A: the C′ pulse", TAU)
draw([Delay(TAU)], fig.add_subplot(top[0, 1]),
     r"run B: a delay of the same length     $\phi_{BS}$ = phase(A) − phase(B)", TAU)

# (a) one Calpha spin, seen from +z
ax = fig.add_subplot(gs[1, 0])
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(th), np.sin(th), color="0.8", lw=0.6)
ax.axhline(0, color="0.85", lw=0.5); ax.axvline(0, color="0.85", lw=0.5)
for M, c, lab in [(M_delay, "0.5", f"delay {TAU:g} ms"), (M_pulse, "C3", f"Q3 180° ({TAU:g} ms)")]:
    ax.annotate("", xy=(M[0, iCA], M[1, iCA]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2))
    ax.plot([], [], color=c, label=lab)
a0 = np.arctan2(M_delay[1, iCA], M_delay[0, iCA])
arc = np.linspace(a0, a0 + np.radians(phi[iCA]), 50)
ax.plot(0.35 * np.cos(arc), 0.35 * np.sin(arc), color="k", lw=0.8)
ax.text(0.45 * np.cos(arc[25]), 0.45 * np.sin(arc[25]), r"$\phi_{BS}$", ha="center", va="center")
ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
ax.set_xlabel(r"$M_x$"); ax.set_ylabel(r"$M_y$")
ax.set_title(f"(a) C$\\alpha$ at {CA:g} kHz, start $M_x$\n"
             f"$\\phi_{{BS}}$ = {phi[iCA]:.0f}°,  $|M_{{xy}}|$ = 1")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.62), frameon=False)

# (b) against offset, with theory
ax = fig.add_subplot(gs[1, 1:]); plot_axes.append(ax)
ax.axvspan(df[exc].min(), df[exc].max(), color="0.9", lw=0)
ax.text(0, 175, "excited\n(not BS)", ha="center", va="top", fontsize=6, color="0.4")
ax.plot(df, np.where(~exc, phi, np.nan), color="k", lw=1.6, label="PULSIM (pulse − delay)")
ax.plot(df, np.where(~exc, integral, np.nan), color="C0", lw=0.9,
        label=r"effective field: $\int(\sqrt{\Delta^2+\omega_1^2}-|\Delta|)\,dt$")
ax.plot(df, np.where(~exc, approx, np.nan), color="C1", lw=0.9, ls="--",
        label=r"approx.: $\frac{1}{2\Delta}\int\omega_1^2\,dt$")
ax.plot(df[iCA], phi[iCA], "o", ms=3, color="C3")
ax.set_xlabel("offset from carrier (kHz)"); ax.set_ylabel(r"$\phi_{BS}$ (deg)")
ax.set_title(f"(b) Bloch–Siegert phase vs offset, Q3 180° {TAU:g} ms")
ax.legend(loc="upper left", frameon=False)

# (c1) RF phase does not matter
ax = fig.add_subplot(gs[2, 0:2]); plot_axes.append(ax)
for ph, lab, c, ls in [("x", "x", "k", "-"), ("y", "y", "C0", "--"), (np.pi, "−x", "C1", ":")]:
    p = q3(phase=ph)
    ax.plot(df, np.where(~excited(p, df), unwrap_from_edges(bs_phase(p, df), df), np.nan),
            color=c, ls=ls, lw=1.2, label=f"phase {lab}")
ax.set_xlabel("offset from carrier (kHz)"); ax.set_ylabel(r"$\phi_{BS}$ (deg)")
ax.set_title(r"(c1) RF phase does not matter ($\phi_{BS}\propto\omega_1^2$)")
ax.legend(loc="upper left", frameon=False)

# (c2) longer pulse, smaller phase
ax = fig.add_subplot(gs[2, 2]); plot_axes.append(ax)
for tau, c in zip([0.35, 0.7, 1.4], ["k", "C0", "C1"]):
    p = q3(tau=tau)
    u = unwrap_from_edges(bs_phase(p, df), df)
    ax.plot(df, np.where(~excited(p, df), u, np.nan), color=c, lw=1.1,
            label=f"{tau:g} ms ({u[iCA]:.0f}°)")
    print(f"  {tau:g} ms Q3 180: phi_BS at Calpha = {u[iCA]:.1f} deg")
ax.set_xlabel("offset (kHz)")
ax.set_title(r"(c2) longer pulse, smaller $\phi_{BS}$ ($\propto 1/\tau$)" + "\n(value at C$\\alpha$)")
ax.legend(loc="upper left", frameon=False)

for a in plot_axes:
    a.axhline(0, color="0.8", lw=0.5)
    a.set_xlim(df[0], df[-1]); a.set_ylim(-180, 180); a.set_yticks(DEG_TICKS)
fig.suptitle("Bloch–Siegert shift: a 180° on C′ rotates Cα without exciting it", fontsize=10, y=0.95)
fig.savefig("tutorial_figures/tutorial_bloch_siegert_shift.png", dpi=200, bbox_inches="tight")


# =============================================================================
# PART 2 -- three ways to remove it, over the Calpha band
# =============================================================================
dfb = np.linspace(*BAND, 401)
DC = 0.5 * (BAND[0] + BAND[1])                       # band centre, kHz
print(f"\nCalpha band {BAND[0]:g} .. {BAND[1]:g} kHz, centre {DC:g} kHz")

# -- method 1: the identical pulse at the mirror position of an echo ----------
d = 0.5                                              # ms, each half of the echo
# Each sequence is written once and both run and drawn. The pulse's own 0.35 ms
# counts as evolution time, so the reference puts a 0.35 ms delay where it sits.
SEQ_ECHO_REF = [Delay(TAU), Delay(d), hard(180, "x"), Delay(d), Delay(TAU)]
SEQ_ECHO_ONE = [q3(), Delay(d), hard(180, "x"), Delay(d), Delay(TAU)]
SEQ_ECHO_TWO = [q3(), Delay(d), hard(180, "x"), Delay(d), q3()]
echo_ref, echo_one, echo_two = (run(s, dfb) for s in (SEQ_ECHO_REF, SEQ_ECHO_ONE, SEQ_ECHO_TWO))
m1_one, m1_two = rel_phase(echo_one, echo_ref), rel_phase(echo_two, echo_ref)
print(f"method 1: one C' pulse in the echo {m1_one.min():.0f} .. {m1_one.max():.0f} deg; "
      f"with the mirror pulse max |error| = {np.abs(m1_two).max():.2f} deg")

# -- methods 2 + 3: what the delay and the phase must be ---------------------
# phi_BS over the band, and its straight-line fit around the centre
phi_b = np.degrees(np.unwrap(np.radians(bs_phase(q3(), dfb))))
slope, _ = np.polyfit(dfb - DC, phi_b, 1)                        # deg / kHz
# how much phase a delay gives per kHz per ms, in PULSIM's sign convention (+-360)
per_ms = np.polyfit(dfb, np.degrees(np.unwrap(np.angle(
    (lambda M: M[0] + 1j * M[1])(run([Delay(1.0)], dfb))))), 1)[0]
t_eq = slope / per_ms                                            # ms the pulse "acts like"
print(f"method 2: phi_BS slope {slope:.2f} deg/kHz, a delay gives {per_ms:.0f} deg/(kHz ms) "
      f"-> the pulse acts like {t_eq * 1000:+.1f} us of delay: change the next delay "
      f"by {-t_eq * 1000:+.1f} us")

T = 0.12                                             # ms of free delay after the C' pulse
SEQ_REF = [Delay(TAU), Delay(T)]                     # no C' pulse at all
SEQ_UNC = [q3(), Delay(T)]
SEQ_DLY = [q3(), Delay(T - t_eq)]                    # method 2
ref, unc, dly = (run(s, dfb) for s in (SEQ_REF, SEQ_UNC, SEQ_DLY))
const = rel_phase(dly, ref)[np.argmin(np.abs(dfb - DC))]
print(f"method 3: after the delay fix a constant {const:+.1f} deg is left "
      f"-> shift the next pulse's phase by {const:+.1f} deg; "
      f"residual max |error| = {np.abs(rel_phase(dly, ref) - const).max():.2f} deg")

# Execute it: a 90 deg readout (phase y in the reference) stores the phase on z.
y = np.pi / 2
SEQ_RO = {"reference: no C′ pulse": SEQ_REF + [hard(90, y)],
          "no correction": SEQ_UNC + [hard(90, y)],
          f"method 2: delay {-t_eq * 1000:+.0f} µs": SEQ_DLY + [hard(90, y)],
          f"method 2 + 3: and 90° phase y {const:+.0f}°": SEQ_DLY + [hard(90, y + np.radians(const))]}
Mz_ref, Mz_unc, Mz_dly, Mz_fix = (run(s, dfb)[2] for s in SEQ_RO.values())
print(f"executed: max |Mz(corrected) - Mz(reference)| = {np.abs(Mz_fix - Mz_ref).max():.3f} "
      f"(uncorrected {np.abs(Mz_unc - Mz_ref).max():.2f})")

fig = plt.figure(figsize=(7, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=0.32, wspace=0.42)
# sequences: method 1 (for panel a) on the left; the four readout sequences
# (for panels b and c) on the right
seq_cols = gs[0, :].subgridspec(1, 2, wspace=0.12)
left = seq_cols[0, 0].subgridspec(4, 1, hspace=0.9)
t1 = sum(el.duration if isinstance(el, Delay) else el.shape.duration for el in SEQ_ECHO_TWO)
seq_axes_l = []
for k, (lab, seq) in enumerate([("(a) reference: delays only", SEQ_ECHO_REF),
                                ("(a) one C′ pulse", SEQ_ECHO_ONE),
                                ("(a) + identical pulse in the other half", SEQ_ECHO_TWO)]):
    seq_axes_l.append(fig.add_subplot(left[k, 0]))
    draw(seq, seq_axes_l[-1], lab, t1)
right = seq_cols[0, 1].subgridspec(4, 1, hspace=0.9)
t2 = max(sum(el.duration if isinstance(el, Delay) else el.shape.duration for el in seq)
         for seq in SEQ_RO.values())
seq_axes_r = []
for k, (lab, seq) in enumerate(SEQ_RO.items()):
    seq_axes_r.append(fig.add_subplot(right[k, 0]))
    draw(seq, seq_axes_r[-1], "(b, c) " + lab, t2)
for stack in (seq_axes_l, seq_axes_r):             # one time axis per stack, at the bottom
    for a in stack[:-1]:
        a.set_xticklabels([])
seq_axes_l[-1].set_xlabel("time (ms)", fontsize=6)
seq_axes_r[-1].set_xlabel("time (ms)   (b) is the same without the final 90°", fontsize=6)
axs = [fig.add_subplot(gs[1, k]) for k in range(3)]

ax = axs[0]
ax.plot(dfb, m1_one, color="k", label="one C′ pulse")
ax.plot(dfb, m1_two, color="C0", label="+ identical pulse in the other half")
ax.set_ylabel("phase error (deg)")
ax.set_title("(a) Method 1: same pulse, mirrored\nposition in the echo")

ax = axs[1]
ax.plot(dfb, rel_phase(unc, ref), color="k", label="no correction")
ax.plot(dfb, rel_phase(dly, ref), color="C0", label=f"delay {-t_eq * 1000:+.0f} µs")
ax.plot(dfb, rel_phase(dly, ref) - const, color="C1", ls="--",
        label=f"+ next-pulse phase {const:+.0f}°")
ax.set_title("(b) Methods 2+3: what is left\nafter each correction")

for a in axs[:2]:
    a.set_ylim(-180, 180); a.set_yticks(DEG_TICKS)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)

ax = axs[2]
ax.plot(dfb, Mz_ref, color="0.6", lw=3, label="reference (no C′ pulse)")
ax.plot(dfb, Mz_unc, color="k", lw=0.9, label="no correction")
ax.plot(dfb, Mz_dly, color="C0", lw=0.9, ls=":", label="delay only")
ax.plot(dfb, Mz_fix, color="C1", lw=0.9, ls="--", label="delay + 90° phase")
ax.set_ylim(-1, 1); ax.set_ylabel(r"$M_z$ after 90° readout")
ax.set_title("(c) Executed: corrected sequence\nmatches the reference")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)

for a in axs:
    a.set_xlabel("Cα offset from C′ (kHz)")
fig.suptitle(f"Removing the Bloch–Siegert phase on Cα (Q3 180° on C′, {TAU:g} ms)",
             fontsize=10, y=0.95)
fig.savefig("tutorial_figures/tutorial_bloch_siegert_compensation.png", dpi=200,
            bbox_inches="tight")
plt.show()
