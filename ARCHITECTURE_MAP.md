# PULSIM — architecture map

**As of:** 2026-09-30 (commit `560d081`). Derived from the source: module
docstrings, public names and relative imports of every file in `PULSIM/`.
The physics each layer implements is specified in
[`docs/PHYSICS_SPECIFICATION.md`](docs/PHYSICS_SPECIFICATION.md); this file
only says *where* things live and *what calls what*.

---

## The idea in one paragraph

A pulse is **a shape, a calibration, and a propagator**, kept separate.
`RFShape` knows only the normalized waveform. `calibration.py` turns it into
a physical RF amplitude — from the pulse area for an ordinary shape, from Q
and the sweep rate for an adiabatic one. `Pulse` combines a shape with a
target (flip angle, Q, or an explicit `nu1_max`) and hands the calibrated RF
to a `Backend`, which steps a magnetization through it. The same `Pulse`
also drives the density-matrix engine through `ShapePulseSegment`, so both
engines always see identical RF.

```
                 bruker.py ── file_import.py
                      │            │
calibration.py ── rf_shape.py (RFShape: 39 registered shapes, FileShape)
                      │
                  pulse_oo.py (Pulse) ── metrics.py
                   │        │
         backend.py │        │ pulse_sequence.py (PulseSequence)
  (Numpy/Torch)     │        │
       │            │        └──────────────► visualization.py (3D views)
   bloch.py         │
   mat_operator.py  │
                    ▼
   liouville.py (Segments, LiouvilleSequence) ── relaxation.py
        │   spin_system.py, spin_operators.py
        └──► sequence_figure.py (sequence diagrams)

   pulse_diagnostics.py  gradients.py  parallel.py  simulate.py   (stand-alone helpers)
```

---

## Layers

### Shapes — `rf_shape.py`, `bruker.py`, `file_import.py`

| Module | Role | Public names |
|---|---|---|
| `rf_shape.py` | the normalized waveform, and the only place a shape is defined | `RFShape` (registry: `RFShape.create(name, ...)`, `RFShape.available()`), `AnalyticShape` and its 33 subclasses, `SincShape`, `CosShape`, `Sinc2PiShape`, `HardShape`, `FileShape`, `CompositeCSVShape` |
| `bruker.py` | read a Bruker/TopSpin shape-file header; standard library only | `parse_header`, `BrukerHeader` (`exmode`, `intent`, `totrot`, `design`, …), `read_bruker_header` |
| `file_import.py` | read amplitude/phase columns from a shape file | `import_file`, `read_xy_points`, `equ2shape` (interactive; the only user of matplotlib/pandas here, imported lazily) |

Every shape is built by `_build()` and read through `envelope()`, which also
applies `time_reversed=True`. The 39 registered names:

| Calibrated by | Shapes |
|---|---|
| pulse area (flip angle) | `eburp1/2`, `iburp1/2`, `uburp`, `reburp`, `gausscasg3/g4/q3/q5`, `hermite`, `seduce1`, `sneeze`, `qsneeze`, `esnob`, `i2snob`, `i3snob`, `rsnob`, `dsnob`, `swrl11/12/17`, `sinc`, `sinc2p`, `cos`, `hard` |
| Q, closed form | `hypsec` (full and half passage) |
| its own sweep | `sincos` (Bendall–Pegg) |
| Q, measured from the waveform | `wurst`, `smoothedchirp`, `tanhtan`, `cawurst`, `casmoothedchirp`, `cagauss`, `calorentz`, `capowhsec` |
| refused (no single Q) | `compositesmoothedchirp` — give `nu1_max` |
| declared by the file | `file` (`FileShape`: intent from `SHAPE_EXMODE`; adiabatic files need `q_mid` or `nu1_max`), `composite` (`CompositeCSVShape`) |

### Calibration — `calibration.py`

No internal imports. `AreaCalibration` (flip angle → amplitude from the signed
integral, Bruker's INTEGFAC); `AdiabaticCalibration` (HypSec closed form,
`half_width` = 1 full / ½ half passage); `resonance_crossing` +
`NumericAdiabaticCalibration` (sweep rate measured at resonance, for every
other adiabatic family); `beta_from_truncation`, `mu_from_sweep_width`.
Spec: Section 6.

### Pulse — `pulse_oo.py`, `pulse_sequence.py`, `metrics.py`

| Module | Role | Public names |
|---|---|---|
| `pulse_oo.py` | shape + target + backend; the single calibration entry point | `Pulse` (`nu1_max`, `b1_max`, `realized_q`, `calibrated_rf()`, `apply(M, df)`) |
| `pulse_sequence.py` | several Pulses in order, with concatenated RF/phase/time for plotting | `PulseSequence` |
| `metrics.py` | figures of merit of a calibrated pulse | `inversion_fidelity`, `realized_q`, `fraction_above` |

### Classical engine — `backend.py`, `bloch.py`, `mat_operator.py`

| Module | Role | Public names |
|---|---|---|
| `backend.py` | the swappable propagator behind `Pulse.apply` | `Backend`, `NumpyBackend` (default; optional T1/T2), `TorchBackend` (experimental) |
| `bloch.py` | Bloch rotation and relaxation kernels | `bloch_rotate`, `bloch_rotate_batch`, `bloch_relax`, `bloch_relax_batch`, `bloch_relax_rotate_batch` (Strang split, production path), `affine_propagate` (exact reference), `bloch_delay`, `relaxation_matrix`, `torch_bloch_rotate` |
| `mat_operator.py` | 3×3 rotation matrices, NumPy and torch | `cpu_rot`, `cpu_rot_batch`, `torch_rot`, `require_torch`, `spin_half`, `boltzmann_factor` |

Spec: Sections 1 and 3.

### Density-matrix engine — `liouville.py`, `relaxation.py`, `spin_system.py`, `spin_operators.py`

`liouville.py` solves the Liouville–von Neumann equation as
ρ → UρU† on 2ⁿ × 2ⁿ matrices (Hilbert space, weak coupling; not
Liouville-space superoperators — see its module docstring).

| Module | Role | Public names |
|---|---|---|
| `liouville.py` | sequence segments and their propagation | `Segment`, `Delay`, `IdealPulse`, `ShapePulseSegment` (wraps a `Pulse`), `RawShapedPulseSegment` (RF array in rad/ms), `LiouvilleSequence` |
| `relaxation.py` | phenomenological T1/T2 per product operator | `Relaxation` |
| `spin_system.py` | nuclei, offsets (rad/ms), couplings (Hz) | `SpinSystem`, `gyro_ratio` |
| `spin_operators.py` | spin-½ operators and products | `Ix`, `Iy`, `Iz`, `E`, `embed`, `product_operator`, `SpinOperators` (readout) |

### Stand-alone helpers

| Module | Role | Public names |
|---|---|---|
| `pulse_diagnostics.py` | first-order J evolution during a pulse (toggling frame), exact delay optimum | `effective_coupling_generator(rf, dt, side=)`, `optimize_delay` |
| `gradients.py` | gradients as position-dependent offsets; spec Section 8 | `gradient_offsets`, `uniform_positions`, `ensemble_average`, `ideal_spoil` |
| `parallel.py` | many independent whole simulations across cores | `parallel_map`, `require_joblib` |
| `simulate.py` | the legacy `sim_*` call signatures, now thin wrappers over `Pulse` | `sim_hard_pulse`, `sim_shaped_pulse`, `sim_import_shaped_pulse`, `sim_own_shaped_pulse` |
| `visualization.py` | 3D Bloch-sphere snapshots and animations of a trajectory | `plot_3D_arrow_snapshots`, `plot_3D_arrow_with_pulse`, `save_animation_to_gif`, … |
| `sequence_figure.py` | sequence diagrams drawn from the segments that actually ran | `describe_sequence`, `draw_sequence` |

`__init__.py` re-exports the everyday names (`RFShape`, `Pulse`,
`PulseSequence`, the backends, the Bloch kernels, the segments,
`LiouvilleSequence`, `SpinSystem`, `Relaxation`, metrics, `parallel_map`,
`sim_*`). Everything else is imported from its submodule.

---

## Dependencies

| Install | Adds | Unlocks |
|---|---|---|
| `pip install .` (base) | numpy, scipy | the whole physics: both engines, all shapes, calibration, gradients, diagnostics |
| `.[viz]` | matplotlib, ipywidgets | `visualization.py`, `sequence_figure.py`, the tutorials' figures |
| `.[file]` | pandas | `CompositeCSVShape` |
| `.[parallel]` | joblib | `parallel_map` |
| `.[torch]` | torch | `TorchBackend`, `torch_bloch_rotate` — experimental |
| `.[all]` | all of the above | |

CI checks the base install on its own (numpy + scipy only, the Pyodide
target), the full install, and a fresh clone running every notebook and
tutorial.

---

## Repository layout

| Path | What it is |
|---|---|
| `PULSIM/` | the package (22 modules) |
| `tests/` | the pytest suite (27 test files); `golden/` baselines, `fixtures/waveforms/` a synthetic shape file |
| `tutorials/` | 12 scripts, all runnable from a clean clone (no vendor files) |
| `PULSIM_colab_oo.ipynb`, `PULSIM_density_colab.ipynb` | the Colab notebooks: Bloch engine and density-matrix engine |
| `docs/PHYSICS_SPECIFICATION.md` | the physics specification |
| `ROTATION_METHOD.md` | notes on the rotation kernels |
| `tools/` | `check_reproducible.py` (CI: run every notebook and tutorial), `generate_test_waveforms.py` (the synthetic fixture) |
| `benchmarks/` | NumPy vs torch timing, `parallel_map` timing |
| `wave/` | vendor shape files — **not distributed** (gitignored); tests that need them skip |

**Legacy, outside the package.** The repository root and `test/` still hold
the scripts from before the object-oriented rewrite:
`pulse_shape_list.py`, `bloch_pulse_simulation.py`, `visualization.py`,
`input_parameter.py`, `automation_input.py`, `color_map.py`,
`check_equivalence.py`, the `*_test.py` and `RF_pulse_*` drivers, and the 21
scripts in `test/`. They are not imported by the package, not run by CI, and
may be stale; use `PULSIM/` and `tutorials/` instead.

---

## What changed since the previous version of this map

The previous map described the pre-rewrite code and is obsolete in full:
`pulse.py` with `cpu_pulse`/`torch_pulse` is replaced by
`RFShape` → `Pulse` → `Backend`; the nmrsim-derived frequency-domain stack
("stack B") is no longer part of the package; and `__init__.py` imports
cleanly from any working directory.
