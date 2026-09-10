
"""
visualization.py -- 3D Bloch-sphere views of a magnetization trajectory.
 
Ported from the root-level bloch_pulse_simulation.py, which predates the
OO layer: it took `Ms[i][:, frame]`, an (n_arrows, 3, n_frames) array that
every driver script had to assemble by hand from the sim_* functions. The
functions here take instead exactly what
 
    Pulse.apply(M, df, trajectory=True)
    PulseSequence.run(M, df, trajectory=True)
 
now return -- an (n_frames, 3, n_arrows) array, frame-first, so that
M_traj[k] is a plain (3, n_arrows) magnetization and M_traj[k][:, i] is
the vector belonging to arrow i. Nothing needs transposing on the way in,
and n_arrows and n_frames are read off the array rather than passed
alongside it and trusted.
 
"Arrow" here is just a column of M: one isochromat, i.e. one entry of the
`df` offset array handed to the simulation. Ten arrows means ten offsets
carried through the same pulse in one vectorized run -- not ten runs.
 
matplotlib is an optional extra (`pip install "pulsim[viz]"`), so this
module is deliberately NOT imported by PULSIM/__init__.py; import it
explicitly as `from PULSIM.visualization import ...`.
 
Four deliberate changes from the original while porting:
 
* `cm.get_cmap(name, n)` / `plt.cm.get_cmap` was removed in matplotlib
  3.9; `plt.get_cmap(name, n)` is used instead.
* No function calls plt.show() or plt.savefig() itself any more -- each
  returns its figure (and animation) and lets the caller decide, which is
  what the rest of the tutorials expect.
* An arrow is drawn at length |M| rather than renormalized to 1, so a
  magnetization that has shrunk shows as a shorter arrow.
* A trajectory carries one frame per RF step PLUS the starting frame, so
  it is one longer than seq.time / seq.rf. _rf_panels() reconciles that
  explicitly, and raises if the two do not correspond at all.
"""
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 
 
# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
 
def arrow_colors(n_arrows, color="viridis"):
    """One color per arrow, sampled evenly across a colormap."""
    cmap = plt.get_cmap(color, n_arrows)
    return [cmap(i) for i in range(n_arrows)]
 
 
def draw_cone_arrow(ax, origin, direction, color='r', length=None,
                    cone_length=0.2, cone_radius=0.05, resolution=20):
    """Draw a shaft-plus-cone arrow from `origin` along `direction`.
 
    length : arrow length. Default None means "use |direction|", so a
             magnetization that has shrunk -- through relaxation, or
             through isochromats fanning out and being summed -- draws as
             a shorter arrow instead of being silently renormalized to 1
             as the original helper did. Pass length=1.0 for the old
             always-unit behaviour.
 
    A zero (or non-finite) direction draws nothing rather than raising.
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0.0:
        return
 
    unit = direction / norm
    if length is None:
        length = norm
    if length <= cone_length:                 # keep a very short arrow drawable
        cone_length = 0.5 * length
        cone_radius = 0.25 * cone_length
 
    shaft_end = origin + unit * (length - cone_length)
    ax.plot([origin[0], shaft_end[0]],
            [origin[1], shaft_end[1]],
            [origin[2], shaft_end[2]], color=color)
 
    # cone tip: build any two axes perpendicular to `unit`
    not_z = np.array([1.0, 0.0, 0.0]) if not np.allclose(unit, [1, 0, 0]) else np.array([0.0, 1.0, 0.0])
    x = np.cross(not_z, unit); x /= np.linalg.norm(x)
    y = np.cross(unit, x)
 
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle = np.array([shaft_end + cone_radius * (np.cos(t) * x + np.sin(t) * y) for t in theta])
    tip = shaft_end + unit * cone_length
    verts = [[tip, circle[i], circle[(i + 1) % resolution]] for i in range(resolution)]
    ax.add_collection3d(Poly3DCollection(verts, color=color))
 
 
def draw_bloch_sphere(ax, radius=1.0, num_phi=21, num_theta=21):
    """The faint unit sphere the magnetization moves on."""
    phi, theta = np.meshgrid(np.linspace(0, 2 * np.pi, num_phi),
                             np.linspace(0, np.pi, num_theta))
    ax.plot_surface(radius * np.sin(theta) * np.cos(phi),
                    radius * np.sin(theta) * np.sin(phi),
                    radius * np.cos(theta),
                    color='k', alpha=0.05, edgecolors='k', linewidth=0.1)
 
 
def style_3d_axes(ax, title=None, lim=1.0, font_name=None, labelsize=7):
    """Cubic aspect, equal limits, x/y/z labels -- applied after every cla().
 
    Three ticks per axis, not matplotlib's default seven: at the size these
    panels are actually printed, a full tick set is unreadable overlap and
    the z labels clip off the edge.
    """
    font = {} if font_name is None else {"fontname": font_name}
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([-lim, 0, lim])
    ax.set_yticks([-lim, 0, lim])
    ax.set_zticks([-lim, 0, lim])
    ax.tick_params(labelsize=labelsize, pad=-1)
    ax.set_xlabel('X', labelpad=-4, **font)
    ax.set_ylabel('Y', labelpad=-4, **font)
    ax.set_zlabel('Z', labelpad=-4, **font)
    if title is not None:
        ax.set_title(title, **font)
 
 
# --------------------------------------------------------------------------
# shared trajectory / RF handling
# --------------------------------------------------------------------------
 
def _check_traj(M_traj):
    """Validate the (n_frames, 3, n_arrows) layout and report its shape."""
    M_traj = np.asarray(M_traj, dtype=float)
    if M_traj.ndim != 3 or M_traj.shape[1] != 3:
        raise ValueError(
            f"M_traj must be (n_frames, 3, n_arrows) -- the layout returned by "
            f"Pulse.apply(..., trajectory=True) -- got {M_traj.shape}. If you have "
            f"the old (n_arrows, 3, n_frames) layout, pass M_traj.transpose(2, 1, 0)."
        )
    return M_traj, M_traj.shape[0], M_traj.shape[2]
 
 
def _rf_panels(time, rf, phase, n_frames):
    """Line up the RF traces with the trajectory.
 
    A trajectory has one frame per RF step PLUS the starting frame, so the
    RF arrays are one shorter than M_traj. Prepending the first sample at
    t = 0 keeps every index meaning the same instant in both.
    """
    amp = np.abs(np.asarray(rf))
    pha = np.asarray(phase, dtype=float)
    t = np.asarray(time, dtype=float)
 
    if not (len(t) == len(amp) == len(pha)):
        raise ValueError(
            f"time, rf and phase must be the same length -- got {len(t)}, "
            f"{len(amp)}, {len(pha)}. seq.time, seq.rf and seq.phase come from "
            f"the same PulseSequence, so they should already agree."
        )
 
    if len(t) == n_frames - 1:
        t = np.concatenate([[0.0], t])
        amp = np.concatenate([[amp[0]], amp])
        pha = np.concatenate([[pha[0]], pha])
    elif len(t) != n_frames:
        raise ValueError(
            f"time has {len(t)} points but the trajectory has {n_frames} frames "
            f"(expected {n_frames} or {n_frames - 1}). A trajectory carries one "
            f"frame per RF step plus the starting frame, so seq.time should be "
            f"exactly one shorter than seq.run(..., trajectory=True)."
        )
    return t, amp, pha
 
 
def _amp_limits(amp):
    lo, hi = float(np.min(amp)), float(np.max(amp))
    if lo == hi == 0.0:
        return -0.1, 0.1
    if lo == 0.0:
        return -0.1 * hi, hi * 1.4
    if hi == 0.0:
        return lo * 1.4, -0.1 * lo
    return lo * 1.4, hi * 1.4
 
 
# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------
 
def plot_3D_arrow_with_pulse(M_traj, time, rf, phase, color="viridis",
                             interval=1, stride=1, time_unit="ms",
                             font_name=None, arrow_length=None):
    """Animate the magnetization on the Bloch sphere beside the RF it feels.
 
    M_traj : (n_frames, 3, n_arrows) -- from *.run(..., trajectory=True)
    time   : (n_frames,) or (n_frames - 1,) time axis, e.g. seq.time
    rf     : complex RF envelope, e.g. seq.rf (only |rf| is drawn)
    phase  : RF phase in degrees, e.g. seq.phase
    stride : draw every `stride`-th frame. The 3D panel is redrawn from
             scratch each frame, so a full shaped pulse at 1 us steps is
             thousands of near-identical frames; stride is what makes a
             saveable GIF out of it.
 
    Returns (fig, ani). The caller shows, saves, or discards it.
    """
    M_traj, n_frames, n_arrows = _check_traj(M_traj)
    t, amp, pha = _rf_panels(time, rf, phase, n_frames)
    colors = arrow_colors(n_arrows, color)
    frames = range(0, n_frames, max(int(stride), 1))
 
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1],
                           wspace=0.3, hspace=0.45)
    ax3d = fig.add_subplot(gs[:, 0], projection='3d')
    ax_amp = fig.add_subplot(gs[0, 1])
    ax_pha = fig.add_subplot(gs[1, 1])
 
    ax_amp.plot(t, amp, color='gray', linewidth=1)
    red_line, = ax_amp.plot([], [], color='red', linewidth=2)
    ax_pha.plot(t, pha, color='gray', linewidth=1)
    blue_line, = ax_pha.plot([], [], color='blue', linewidth=2)
 
    font = {} if font_name is None else {"fontname": font_name}
    for ax, label in ((ax_amp, "Amplitude (mT)"), (ax_pha, "Phase (deg)")):
        ax.set_xlim(t[0], t[-1])
        ax.set_ylabel(label, **font)
        ax.set_xlabel(f"Time ({time_unit})", **font)
    ax_amp.set_ylim(*_amp_limits(amp))
    ax_pha.set_yticks(np.arange(0, 361, 90))
 
    def update(frame):
        ax3d.cla()
        draw_bloch_sphere(ax3d)
        for i in range(n_arrows):
            draw_cone_arrow(ax3d, origin=np.zeros(3), direction=M_traj[frame][:, i],
                            color=colors[i], length=arrow_length)
        style_3d_axes(ax3d, title=f"t = {t[frame]:.3f} {time_unit}", font_name=font_name)
        red_line.set_data(t[:frame + 1], amp[:frame + 1])
        blue_line.set_data(t[:frame + 1], pha[:frame + 1])
        return ()
 
    update(0)
    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    return fig, ani
 
 
def save_animation_to_gif(ani, file_name, fps=25, dpi=120):
    """Write a FuncAnimation out as a GIF (needs pillow)."""
    ani.save(file_name, writer='pillow', fps=fps, dpi=dpi)
    return file_name
 
 
# --------------------------------------------------------------------------
# still figure
# --------------------------------------------------------------------------
 
def plot_3D_arrow_snapshots(M_traj, time, rf, phase, frames, labels=None,
                            color="viridis", time_unit="ms", font_name=None,
                            arrow_length=None, figsize=None):
    """The animation's information as one still figure, for print.
 
    frames : the frame indices to draw as Bloch spheres, left to right
             (negative indices allowed, so -1 is the end of the sequence).
    labels : optional titles for those spheres; defaults to the timestamp.
 
    The bottom row shows |RF| and its phase across the whole sequence with
    a dashed marker at each sampled instant, so a reader can see which
    moment of the pulse each sphere belongs to.
 
    Returns the figure.
    """
    M_traj, n_frames, n_arrows = _check_traj(M_traj)
    t, amp, pha = _rf_panels(time, rf, phase, n_frames)
    colors = arrow_colors(n_arrows, color)
    frames = [int(f) % n_frames for f in frames]
    n_snap = len(frames)
 
    if figsize is None:
        figsize = (3.1 * n_snap, 6.0)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, n_snap, height_ratios=[2.1, 1], hspace=0.30, wspace=0.05)
 
    for k, frame in enumerate(frames):
        ax = fig.add_subplot(gs[0, k], projection='3d')
        draw_bloch_sphere(ax)
        for i in range(n_arrows):
            draw_cone_arrow(ax, origin=np.zeros(3), direction=M_traj[frame][:, i],
                            color=colors[i], length=arrow_length)
        title = labels[k] if labels is not None else f"t = {t[frame]:.3f} {time_unit}"
        style_3d_axes(ax, title=f"{k + 1}.  {title}", font_name=font_name)
 
    font = {} if font_name is None else {"fontname": font_name}
    ax_amp = fig.add_subplot(gs[1, :])
    ax_amp.plot(t, amp, color='0.3', linewidth=1)
    ax_amp.set_xlim(t[0], t[-1])
    ax_amp.set_ylim(*_amp_limits(amp))
    ax_amp.margins(x=0)
    ax_amp.set_xlabel(f"Time ({time_unit})", **font)
    ax_amp.set_ylabel("|RF| (mT)", **font)
 
    ax_pha = ax_amp.twinx()
    ax_pha.plot(t, pha, color='tab:blue', linewidth=0.9, alpha=0.55)
    ax_pha.set_ylabel("Phase (deg)", color='tab:blue', **font)
    ax_pha.tick_params(axis='y', labelcolor='tab:blue')
    ax_pha.set_yticks(np.arange(0, 361, 90))
 
    # Markers carry only the snapshot's NUMBER, matched by the "1." .. "n."
    # prefix on each sphere's title. Writing the full label here instead
    # collides the moment two snapshots sit close together in time -- which
    # they always do, since the interesting instants cluster at the pulse
    # boundaries.
    for k, frame in enumerate(frames):
        ax_amp.axvline(t[frame], color='crimson', linestyle='--', linewidth=1)
        ax_amp.annotate(f"{k + 1}", xy=(t[frame], 1.0), xycoords=('data', 'axes fraction'),
                        xytext=(0, 3), textcoords='offset points', color='crimson',
                        fontsize=9, fontweight='bold', ha='center', va='bottom',
                        annotation_clip=False)
 
    return fig
 

