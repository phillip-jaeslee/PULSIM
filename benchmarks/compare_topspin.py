"""
compare_topspin.py -- excitation profiles from TopSpin (NMR-Sim) against
PULSIM's two simulators: the Bloch equation (Pulse.apply) and the density
matrix (LiouvilleSequence, one spin).

The profiles in Excitation_profiles/ were computed with NMR-Sim in TopSpin 4.2.0
(August 2024); the date of each is in its header.

Run it from the repository root. It needs wave/ (the Bruker shape files, not
distributed with the repository) at the root:

    python benchmarks/compare_topspin.py

It prints, for every pulse in CASES, the largest difference from TopSpin of
each simulator, and draws the pulses named in SHOW (one row each: TopSpin, Bloch, density matrix)
to compare_topspin.png and compare_topspin.svg next to this script.

Conventions found while matching the TopSpin files (each checked numerically):
  * NMR-Sim reports My with the opposite sign to PULSIM; it is flipped here
    (MY_SIGN) so the curves can be compared directly. Mx and Mz agree as is.
  * The "600" in Eburp600 / Gauss600 / G4600 / Seduce600 / sine600 is the pulse
    length (0.6 ms). GausscascadeQ5 is 1.0 ms and Hypsec1000 1.0 ms.
  * HypSec is adiabatic: its amplitude comes from Q = 5 at the centre of the
    sweep (TopSpin's default), not from a flip angle.
  * Eburp600 is reproduced by PULSIM's analytic E-BURP-1 (to ~0.015, TopSpin's
    own table differs slightly); every other pulse uses the vendor shape file.

Not included, because the settings TopSpin used are not known:
  Ramp600.csv, RF_pulse_9018090_sine_benchmark.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # PULSIM from this clone

from matplotlib.ticker import MaxNLocator

import numpy as np
import matplotlib.pyplot as plt

from PULSIM.backend import NumpyBackend
from PULSIM.liouville import LiouvilleSequence, ShapePulseSegment
from PULSIM.pulse_oo import Pulse
from PULSIM.rf_shape import RFShape
from PULSIM.spin_operators import SpinOperators
from PULSIM.spin_system import SpinSystem, gyro_ratio

HERE = Path(__file__).resolve().parent        # benchmarks/: Excitation_profiles/, the figure
PROFILES = HERE / "Excitation_profiles"
WAVE = HERE.parent / "wave"                   # wave/ at the repository root
OUT = HERE / "compare_topspin"                # .png and .svg are added
NUCLEUS = "H"
MY_SIGN = -1.0                                # NMR-Sim My = -(PULSIM My)

# figure size and text: 7-inch-wide figure, 8 pt text (as in PULSIM_colab.ipynb)
FIG_WIDTH = 7.0                               # inches -- the final printed width
ROW_HEIGHT = 1.55                             # inches per pulse (one row each)
FONT = 8.0                                    # every label, tick and legend
FONT_FAMILY = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]   # first one installed wins

# the pulses drawn, one row each (all CASES are still compared in the printed table)
SHOW = ["Gaussian 90°, 0.6 ms",                    # amplitude-modulated shape
        "HypSec (adiabatic), 1 ms",                # amplitude and frequency modulated
        "Q5 3 ms – hard 180° – Q5 rev 3 ms"]       # a train of three pulses


# ---- the pulses ------------------------------------------------------------
def backend():
    return NumpyBackend(Gamma=gyro_ratio(NUCLEUS))


def file_pulse(name, duration_ms, flip_deg=None, axis="x", **shape_kw):
    """A vendor shape from wave/, calibrated like TopSpin: by flip angle, or
    (adiabatic files, flip_deg=None) by the shape's own Q."""
    shape = RFShape.create("file", path=str(WAVE / name), duration=duration_ms, **shape_kw)
    if flip_deg is None:
        return Pulse(shape, axis=axis, backend=backend())
    return Pulse(shape, np.deg2rad(flip_deg), axis=axis, backend=backend())


def pulsim_pulse(name, duration_ms, flip_deg, points=1000, axis="x"):
    """One of PULSIM's analytic shapes."""
    shape = RFShape.create(name, duration=duration_ms, points=points)
    return Pulse(shape, np.deg2rad(flip_deg), axis=axis, backend=backend())


def hard(duration_ms, flip_deg, axis="x"):
    return pulsim_pulse("hard", duration_ms, flip_deg, points=10, axis=axis)


# (panel title, TopSpin file, function building the pulse train)
CASES = [
    ("Gaussian 90°, 0.6 ms",        "Gauss600.csv",       lambda: [file_pulse("Gaus1.1000", 0.6, 90)]),
    ("G4 90°, 0.6 ms",              "G4600.csv",          lambda: [file_pulse("G4.1000", 0.6, 90)]),
    ("sine 90°, 0.6 ms",            "sine600.csv",        lambda: [file_pulse("sine.jhl", 0.6, 90)]),
    ("SEDUCE 90°, 0.6 ms",          "Seduce600.csv",      lambda: [pulsim_pulse("seduce1", 0.6, 90)]),
    ("E-BURP-1 90°, 0.6 ms",        "Eburp600.csv",       lambda: [pulsim_pulse("eburp1", 0.6, 90)]),
    ("Gauss cascade Q5 90°, 1 ms",  "GausscascadeQ5.csv", lambda: [file_pulse("GaussCascadeQ5", 1.0, 90)]),
    ("HypSec (adiabatic), 1 ms",    "Hypsec1000.csv",     lambda: [file_pulse("HypSec", 1.0, q_mid=5.0)]),
    ("hard 90°, 20 µs",             "hard20.csv",         lambda: [hard(0.02, 90)]),
    ("Q5 3 ms – hard 180° – Q5 rev 3 ms", "GaQ5_3m_20u_GaQ5rev_3m.csv",
     lambda: [file_pulse("GaussCascadeQ5", 3.0, 90), hard(0.02, 180),
              file_pulse("GaussCascadeQ5_rev", 3.0, 90)]),
]


# ---- reading TopSpin ---------------------------------------------------------
def read_topspin(path):
    """NMR-Sim excitation profile: offset (Hz) and Mx, My, Mz."""
    rows = [line.split() for line in open(path) if line.strip() and not line.startswith(";")]
    data = np.array([[float(r[0].rstrip(":"))] + [float(v) for v in r[1:4]] for r in rows])
    return data[:, 0], data[:, 1:4].T


# ---- the two PULSIM simulators -------------------------------------------------
def bloch_profile(pulses, f_hz):
    M = np.zeros((3, len(f_hz)))
    M[2] = 1.0
    for p in pulses:
        M = p.apply(M, f_hz / 1000.0)                 # offsets in kHz
    return M


def density_profile(pulses, f_hz):
    out = np.empty((3, len(f_hz)))
    for k, f in enumerate(f_hz):
        ss = SpinSystem(nuclei=[NUCLEUS], offsets=[2 * np.pi * f / 1000.0])   # rad/ms
        ops = SpinOperators(ss)
        sigma = LiouvilleSequence([ShapePulseSegment(p) for p in pulses], ss).propagate(ops["Iz"])
        out[:, k] = [ops.expectation(sigma, "I" + a) for a in "xyz"]
    return out


# ---- the figure -------------------------------------------------------------------
def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT_FAMILY,
                         "svg.fonttype": "none",     # SVG keeps real, editable text
                         "pdf.fonttype": 42,         # PDF embeds the TrueType font
                         "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
                         "xtick.labelsize": FONT, "ytick.labelsize": FONT,
                         "legend.fontsize": FONT, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    from matplotlib import font_manager
    used = font_manager.get_font(font_manager.findfont(
        font_manager.FontProperties(family=["sans-serif"]))).family_name
    print(f"font: {used}" + ("" if used == "Arial" else "   (Arial not found -- install it for Arial)"))

    # every case: the table; the SHOW cases: kept for the figure
    results = {}
    print(f"{'pulse':38s} {'Bloch - TopSpin':>16s} {'density - TopSpin':>18s}")
    for title, csv, build in CASES:
        f, ts = read_topspin(PROFILES / csv)
        ts = ts * np.array([[1.0], [MY_SIGN], [1.0]])
        pulses = build()
        mb = bloch_profile(pulses, f)
        md = density_profile(pulses, f)
        db, dd = np.abs(mb - ts).max(), np.abs(md - ts).max()
        print(f"{title:38s} {db:16.4f} {dd:18.4f}")
        results[title] = (f, ts, mb, md, db, dd)

    # one row per pulse, one column per source; each panel holds Mx, My, Mz
    colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    fig, axs = plt.subplots(len(SHOW), 3, figsize=(FIG_WIDTH, ROW_HEIGHT * len(SHOW) + 0.45),
                            sharey=True, squeeze=False)
    for row, title in zip(axs, SHOW):
        f, ts, mb, md, db, dd = results[title]
        fk = f / 1000.0                                   # plot in kHz: shorter tick labels
        panels = [("TopSpin (NMR-Sim)", ts),
                  ("Bloch (PULSIM)", mb),
                  ("Density matrix (PULSIM)", md)]
        for ax, (name, M) in zip(row, panels):
            for i, a in enumerate("xyz"):
                ax.plot(fk, M[i], color=colors[a], lw=1.0)
            ax.axhline(0, color="0.85", lw=0.4, zorder=0)
            ax.set(xlim=(fk.min(), fk.max()), ylim=(-1.05, 1.05), yticks=[-1, 0, 1])
            ax.set_title(name, pad=2)
            ax.tick_params(pad=1.5)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        row[0].set_ylabel("magnetization", labelpad=1.5)
        label = row[0].annotate(f"{title} ——— max |Δ| from TopSpin: Bloch {db:.4f}, "
                                f"density matrix {dd:.4f}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 13),
                                textcoords="offset points", ha="left", va="bottom",
                                fontweight="bold")
        label.set_in_layout(False)                        # the row label may span all columns
    for ax in axs[-1]:
        ax.set_xlabel("offset (kHz)", labelpad=1.5)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[a], lw=1.5, label=f"$M_{a}$") for a in "xyz"]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               handlelength=2.0, columnspacing=1.5, borderaxespad=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=2.4, w_pad=1.0)
    for ext in ("png", "svg"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    print(f"saved {OUT}.png and {OUT}.svg")
    plt.show()


if __name__ == "__main__":
    main()
