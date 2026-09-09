# The magnetization-rotation method in `pulse.py` / `bloch.py`

This documents exactly how PULSIM computes the rotation of a magnetization
vector through an RF pulse: the physical model, the rotation-matrix
construction, the per-sample time integration, and the flip-angle
calibration. All formulas below match the code as it currently stands
(`bloch.py`, `bloch_rotate` and `torch_bloch_rotate`, x-axis branch — the
only branch verified and used by any caller; see the note at the end).

## 1. Physical model

A single, isolated nuclear spin's magnetization is treated classically as a
3-vector **M** = (Mx, My, Mz) in the rotating frame. No relaxation is
included in this part of the code (`bloch_rotate` has no T1/T2 — that's a
separate function, `bloch_relax`). During a pulse, the spin experiences an
effective field

**B**(t) = ( B1x(t), B1y(t), Δf/γ )

where (B1x, B1y) is the applied RF field — the real and imaginary parts of
the complex envelope RF(t) = B1x(t) + i·B1y(t) — and Δf/γ̄ is the resonance
offset expressed as an equivalent static field (γ̄ = γ/2π, the reduced
gyromagnetic ratio in kHz/mT — `Gamma` in the code; see
docs/PHYSICS_SPECIFICATION.md §1.3). The magnetization precesses about
**B**(t) at the instantaneous Larmor rate 2πγ̄|**B**(t)|.

## 2. Rotating a vector about an arbitrary axis, using only Rx, Ry, Rz

The code has no general axis–angle rotation primitive — only the three
elementary rotations, defined in `mat_operator.py`:

```
Rx(φ) = [[1,    0,     0   ],      Ry(φ) = [[cos φ, 0, sin φ],     Rz(φ) = [[ cos φ, sin φ, 0],
         [0, cos φ, sin φ],                 [0,     1,    0  ],             [-sin φ, cos φ, 0],
         [0,-sin φ, cos φ]]                 [-sin φ,0, cos φ]]              [ 0,      0,    1]]
```

(Note: these correspond to rotation by **−φ** in the usual right-hand-rule
convention — `Rx(φ)` here equals the textbook `Rx(−φ)`. This is a
self-consistent convention used throughout, not an error, and it's the
reason the composition below has the signs it has.)

To rotate **M** by angle β about the direction of **B**, the code:

**a. Finds the polar/azimuthal angle of B.**

```
η = arccos( Bz / |B| )     (angle of B from +z)
θ = atan2( By, Bx )        (azimuthal angle of B in the xy-plane)
```

**b. Builds the "align, rotate, un-align" operator.** `U = Ry(−η)·Rz(θ)`
rotates **B̂** onto +z (this can be checked directly: applying `Rz(θ)` first
brings B into the xz-plane with positive x-component, then `Ry(−η)` brings
that into alignment with +z). Once **B** is aligned with z, rotating by β is
just `Rz(β)`. Undoing the alignment with `U⁻¹` gives the full operator:

```
M' = U⁻¹ · Rz(β) · U · M  =  Rz(−θ)·Ry(η)·Rz(β)·Ry(−η)·Rz(θ) · M
```

This is exactly what `bloch_rotate`'s `angle == "x"` branch computes, and
`torch_bloch_rotate`'s does too (batched over many offsets at once via
`torch.bmm`, but the same formula per offset). It is mathematically
equivalent to a direct Rodrigues rotation of **M** about **B̂** by angle −β —
checked numerically against the closed-form Rodrigues formula to machine
precision (max error ~1e-14) as part of verifying this code.

**c. The flip angle for one time step** is the Larmor precession angle
accumulated over that step:

```
β = 2π · γ · |B| · Δt
```

## 3. Stepping through a shaped pulse

A pulse shape is sampled at N points; sample *n* has envelope value
`RF[n] = B1x[n] + i·B1y[n]` (already calibrated — see §4) and duration
`Δt = t_max / N`. For a fixed resonance offset `df`, the field at step *n* is

```
B(n) = [ Re(RF[n]),  Im(RF[n]),  df/γ ]
```

and the magnetization is updated by applying the rotation from §2 with this
`B(n)` and `Δt`, once per sample, in time order:

```
M_final = R(B(N-1), β(N-1)) · ... · R(B(1), β(1)) · R(B(0), β(0)) · M_initial
```

This is *not* a linearized (Euler) approximation — each individual step is
an **exact** rotation for a field held constant over that Δt. The only
approximation is treating the true, continuously-varying B(t) as piecewise
constant, sampled once per point of the shape. Error comes from how much
B(t) changes *within* one sample, not from the rotation itself — this is a
zeroth-order Magnus / piecewise-exponential integrator, and it converges to
the exact time-ordered evolution as N → ∞.

A hard pulse is the special case where `RF[n]` is constant across all N
samples (see §4); an imported pulse (`import_shaped_pulse`) builds `RF[n]`
from an [amplitude, phase] table read from a file instead of a closed-form
shape function, but is stepped through identically.

## 4. Calibrating RF amplitude to a target flip angle

The raw shape (from `pulse_shape_list.py`, or the amplitude column of an
imported file) is normalized so that applying the *whole* pulse produces the
requested `flip` (radians), assuming on-resonance (`df = 0`):

```
RF[n] = flip · shape[n] / Σ(shape) / (2π · γ · Δt)
```

The intuition: if every sample rotated by the same small increment (valid in
the on-resonance, small-Δt limit), the total rotation would be
`Σ(2π·γ·|RF[n]|·Δt) = 2π·γ·Δt·Σ(RF[n]) = 2π·γ·Δt · flip/(2π·γ·Δt) = flip` —
so this normalization is exactly the one that makes the sum of per-sample
flip angles equal the target, on resonance.

**Adiabatic/complex-phase pulses get an extra ×2.** When the imported
pulse's phase column reaches ≥350° (`import_shaped_pulse`) or the shape's
envelope is complex-valued rather than purely real (`torch_shaped_pulse`),
the calibration above is doubled. This is applied as a special case in the
code; it is not derived from first principles here, and the two trigger
conditions (phase threshold vs. complex-valued check) are not quite the same
test — worth knowing if you're calibrating an unusual shape, since which
branch fires depends on which function you call.

## 5. Sweeping many offsets (the excitation profile)

Everything above is for one resonance offset `df`. To produce an excitation
profile, the identical calculation (§2–4) is repeated independently for each
`df` in a grid (`np.linspace(-BW/2, BW/2, num=1000)` in the numpy path)
— `df` only ever enters through the `Bz` component of the field, so each
offset's trajectory is completely independent of every other's. The numpy
path (`cpu_pulse`) loops over offsets in Python (or via `joblib` when
`MULTI=True`); the torch path (`torch_pulse`) batches all offsets into one
tensor and applies the rotation to all of them at once per time step
(`torch_bloch_rotate`), which is why it needs `torch.bmm` rather than a
single 3×3 matrix multiply.

## 6. Provenance note

`bloch_rotate` and `torch_bloch_rotate` were found, during a separate
verification pass, to each have a sign error in the η term (numpy) / θ term
(torch) of the composition in §2b — traced by comparing against an
independently-derived Rodrigues rotation. Both are now fixed to the formula
given above and verified against that ground truth (numpy: ~1e-14 max error
over 5000 random cases; torch: ~1e-6, limited by its internal float32
precision). This document describes the corrected, current formula. The
`y`/`z` branches of both functions use a different pair of elementary
matrices and were *not* touched or re-verified — nothing in the codebase
currently calls them (`angle` is always `"x"`).

## 7. Cross-reference to the newer object-oriented layer

If you're working with the `RFShape`/`Backend`/`Pulse` classes built
alongside this document: `RFShape.envelope()` supplies `shape[n]` from §3–4;
`Backend.rotate()` implements §2–3 exactly (`NumpyBackend` loops per offset,
`TorchBackend` batches, matching §5); `Pulse.calibrated_rf()` implements §4,
including the `is_adiabatic` ×2 rule. Nothing in this document changes by
using one interface or the other — they compute the same math.
