"""
sequence_figure.py -- draw a pulse sequence from the segments that are
actually propagated.

The diagram in a paper and the sequence in the code are usually two separate
thing, and they drift. The module removes the second copy: it walks a
LiouvilleSequence's own segment list, so the picture cannot disagree with the
simulation beside it.

Two renderings, and the difference between them is the teaching point:

    to_scale=False      the textbook drawing. Narrow open bar = 90 degrees,
                        wide filled bar = 180, delays as labeled brackets.
                        Every element gets comparable visual weight.
    
    to_scale=True       real milliseconds on the axis. A shaped pulse occupies
                        its true width and is drawn as its actual envelop.

Drawn to scale, a 2 ms selective pulse next to a 1.7 ms delay is visibly the
longer of the two -- which is the whole argument about finite pulse duration,
made before a word of text.

matplotlib is imported at module level, so this module is NOT imported by
PULSIM/__init__.py: 'import PULSIM' must keep working on a base install with
only numpy and scipy (see the base-only CI job). Import it explicitly:

    from PULSIM.sequence_figure import draw_sequence
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .liouville import Delay, IdealPulse, ShapePulseSegment, RawShapedPulseSegment

__all__ = ["draw_sequence", "describe_sequence"]

_PHASE_NAMES = {0: "x", 90: "y", 180: "-x", 270: "-y"}

def _phase_label(phase_rad):
    deg = round(np.degrees(phase_rad)) % 360
    return _PHASE_NAMES.get(deg, f"{deg:g}˚")

def _item(kind, duration, channels=(), env=None, flip=None, phase=0.0, name=None):
    return dict(kind=kind, duration=float(duration), channels=tuple(channels),
                env=env, flip=flip, phase=float(phase), name=name)

def describe_sequence(sequence):
    """Reduce a LiouvilleSequence to a list of plain dicts.

    Separated from the drawing so the layout can be inspected, tested, or
    rendered by something other than matplotib.
    """
    ss = sequence.spin_system
    n = ss.n_spins
    out = []

    for seg in sequence.segments:
        if isinstance(seg, Delay):
            out.append(_item("delay", seg.duration))
        elif isinstance(seg, IdealPulse):
            ch = tuple(ss.channel(seg.channel)) if seg.channel is not None else tuple(range(n))
            out.append(_item("ideal", seg.duration, ch, flip=seg.flip, phase=seg.phase))
        elif isinstance(seg, ShapePulseSegment):
            rf = np.asarray(seg.pulse.calibrated_rf())
            out.append(_item("shaped", seg.pulse.shape.duration, ss.channel(seg.pulse.Gamma),
                             env=rf, flip=seg.pulse.flip,
                             name=getattr(seg.pulse.shape, "name", None)))
        elif isinstance(seg, RawShapedPulseSegment):
            rf = np.asarray(seg.rf)
            out.append(_item("shaped", len(rf) * seg.dt, ss.channel(seg.channel), env=rf))
        else:
            raise TypeError(
                f"draw_sequence does not know how to draw {type(seg).__name__}. "
                f"Add a branch to describe_sequence rather than drawing it by hand."
            )
    return out

def _group_simultaneous(items):
    """Merge consecutive ideal pulses on disjoint channels into one event.
 
    Simultaneous pulses are written as separate segments because the
    propagator applies them one after another -- which is exact, since an
    ideal pulse has zero duration. But drawing them in separate time slots
    says something false: that they do not happen together. Consecutive
    ideal pulses whose channel sets do not overlap are one event.
    """
    out, i =[], 0
    while i < len(items):
        it = items[i]
        if it["kind"] != "ideal":
            out.append(dict(it, pulses=None)); i += 1; continue
        pulses, used = [dict(it)], set(it["channels"])
        j = i + 1
        while j < len(items) and items[j]["kind"] == "ideal" and not (set(items[j]["channels"]) & used):
            pulses.append(dict(items[j])); used |= set(items[j]["channels"]); j += 1
        out.append(dict(_item("ideal", max(p["duration"] for p in pulses), sorted(used)),
                        pulses=pulses))
        i = j

    return out

def _attach_to_shaped(items):
    """Attach an ideal pulse to an adjacent shaped pulse on other channels.

    An ideal pulse has zero duration, so one sitting immediately beside a
    shaped pulse on a different channel happens AT that pulse's edge, not in
    a time slot of its own. Giving it its own slot draws a lie: it pushes the
    two apart on the time axis and says they are sequential events. Attached,
    it is drawn on the shaped pulse's edge, where the propagator applies it.
    """
    out = []
    for it in items:
        if it["kind"] == "ideal" and out and out[-1]["kind"] == "shaped" \
                and not (set(it["channels"]) & set(out[-1]["channels"])) \
                and not out[-1].get("attached"):
            out[-1] = dict(out[-1], attached=it["pulses"], attach_at="end")
            continue
        if it["kind"] == "shaped" and out and out[-1]["kind"] == "ideal" \
                and not (set(it["channels"]) & set(out[-1]["channels"])):
            out.append(dict(it, attached=out.pop()["pulses"], attach_at="start"))
            continue
        out.append(it)
    return out

def _layout(items, to_scale, min_pulse_frac):
    """Assign each item an x position and a drawn width"""
    if to_scale:
        widths = [it["duration"] for it in items]
        total = sum(widths) or 1.0
        floor = min_pulse_frac * total
        # An ideal pulse has NO duration by construction. Drawing it at its
        # true width would make it invisible, so it gets a floor -- and the
        # caller is told, because a diagram that silently fakes a width is
        # worse than one that admits it.
        faked = []
        for i, it in enumerate(items):
            if it["kind"] == "ideal" and widths[i] < floor:
                widths[i] = floor
                faked.append(i)

    else:
        widths = [{"delay": 2.6, "ideal": 0.7, "shaped": 1.4}[it["kind"]] for it in items]
        faked = []

    x, positions = 0.0, []
    for w in widths:
        positions.append((x, w))
        x += w

    return positions, x, faked

H90, H180 = 0.52, 0.52   # bar heights: equal, because both pulses run at the
                         # same RF amplitude -- what differs is duration, and
                         # the widths below already say that
W90, W180 = 0.55, 1.0    # bar widths, as a fraction of the slot

_LABEL_DY = 0.38   # one label height for every element, so labels on a
                   # channel line up instead of stepping with bar height

def _draw_ideal_pulse(ax, xc, w_slot, sub, y, annotate):
    """Draw one ideal pulse centred on xc. Shared by the slot-drawn pulses
    and the ones attached to a shaped pulse, so both look identical."""
    is180 = abs(sub["flip"] - np.pi) < 1e-6
    h = H180 if is180 else H90
    ww = w_slot * (W180 if is180 else W90)
    ax.add_patch(Rectangle((xc - ww / 2, y - h / 2), ww, h,
                           facecolor="0.15" if is180 else "white",
                           edgecolor="0.15", lw=1.1, zorder=3))
    if annotate:
        ax.text(xc, y + _LABEL_DY,
                f"{np.degrees(sub['flip']):.0f}$^\\circ_{{{_phase_label(sub['phase'])}}}$",
                ha="center", va="bottom", fontsize=9)

def draw_sequence(sequence, ax=None, to_scale=True, title=None, min_pulse_frac=0.012, annotate=True):
    """Draw `sequence` (a LiouvilleSequence) as a pulse-sequence diagram.
 
    Returns the matplotlib Axes.
    """
    items = _attach_to_shaped(_group_simultaneous(describe_sequence(sequence)))
    ss = sequence.spin_system
    n = ss.n_spins
    positions, total, faked = _layout(items, to_scale, min_pulse_frac)
 
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 1.5 + 1.15 * n))
 
    ys = [n - 1 - i for i in range(n)]
    for i, y in enumerate(ys):
        ax.plot([0, total], [y, y], color="0.25", lw=1.0, zorder=1)
        ax.text(-0.015 * total, y, ss.nuclei[i], ha="right", va="center", fontsize=11)
  
    for it, (x0, w) in zip(items, positions):
        if it["kind"] == "delay":
            if annotate and it["duration"] > 0:
                ylo = min(ys) - 0.62
                ax.annotate("", xy=(x0 + w, ylo), xytext=(x0, ylo), arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.9))
                lab = f"{it['duration']:.3g} ms" if to_scale else "Δ"
                ax.text(x0 + w / 2, ylo + 0.07, lab, ha="center", va="bottom", fontsize=9, color="0.4")
            continue
 
        if it["kind"] == "ideal":
            for sub in it["pulses"]:
                for ci in sub["channels"]:
                    _draw_ideal_pulse(ax, x0 + w / 2, w, sub, ys[ci], annotate)
            continue
 
        for ci in it["channels"]:
            y = ys[ci]
            if it["kind"] == "ideal":
                continue    # drawn below, per sub-pulse, so a group shares one x
            else:
                env = it["env"]
                a = np.abs(env)
                a = a / (a.max() or 1.0)
                sgn = np.sign(np.real(env))
                sgn[sgn == 0] = 1.0
                t = x0 + np.linspace(0, w, len(env))
                ax.fill_between(t, y, y + 0.5 * H180 * a * sgn, facecolor="0.62", edgecolor="0.15", lw=0.9, zorder=3)
                if annotate:
                    name = it.get("name") or "shaped"
                    lab = name if it["flip"] is None else f"{name}  {np.degrees(it['flip']):.0f}$^\\circ$"
                    ax.text(x0 + w / 2, y + _LABEL_DY, lab,
                            ha="center", va="bottom", fontsize=9)

        for sub in it.get("attached") or ():
            xc = x0 + w if it.get("attach_at") == "end" else x0
            for ci in sub["channels"]:
                _draw_ideal_pulse(ax, xc, min_pulse_frac * total if to_scale else 0.7,
                                  sub, ys[ci], annotate)
                
    # An attached ideal pulse is centred ON the shaped pulse's edge, so half
    # of it lies past the end of the sequence -- the axis has to make room or
    # the last pulse is drawn cut in half.
    tail = any(it.get("attached") and it.get("attach_at") == "end" for it in items)
    edge = 0.5 * (min_pulse_frac * total if to_scale else 0.7) if tail else 0.0

    ax.set_ylim(min(ys) - 1.0, max(ys) + 0.95)
    ax.set_xlim(-0.10 * total, total * 1.02 + edge)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
 
    if to_scale:
        ax.set_xlabel("time (ms)")
        ax.spines["bottom"].set_visible(True)
    else:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
 
    if title:
        ax.set_title(title, fontsize=11)
 
    if to_scale and faked:
        ax.text(total, min(ys) - 0.95,
                "ideal pulses drawn at minimum width (true duration: zero)",
                ha="right", va="bottom", fontsize=8, color="0.55", style="italic")
    return ax
 

