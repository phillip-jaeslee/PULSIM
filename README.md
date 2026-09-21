# PULSIM: pulse simulator
> Bloch equation calculator for multiple pulses

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/>

| Notebook | What it does | |
|---|---|---|
| **Shaped pulses** | One pulse at a time — excitation profiles, pulse trains, the shape catalogue | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phillip-jaeslee/PULSIM/blob/main/PULSIM_colab_oo.ipynb) |
| **Pulse sequences** | Coupled spins and delays — spin echo, INEPT, BIRD, in the density matrix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phillip-jaeslee/PULSIM/blob/main/PULSIM_density_colab.ipynb) |

![](header.png)

## Installation

OS X & Linux:

```sh
git clone https://www.github.com/phillip-jaeslee/PULSIM
```

## Relaxation

By default PULSIM propagates pure rotations. Relaxation is opt-in, through the
backend:

```python
import numpy as np
import PULSIM

GAMMA = 42.577                      # kHz/mT, reduced gyromagnetic ratio (gamma/2pi)
shape = PULSIM.RFShape.create("gausscasq5", duration=2.0, points=1000)

backend = PULSIM.NumpyBackend(Gamma=GAMMA, T1=900.0, T2=100.0)     # ms
pulse   = PULSIM.Pulse(shape, flip=np.pi / 2, axis="x", backend=backend)

M = pulse.apply(np.array([[0.0], [0.0], [1.0]]), np.array([0.0]))

```

For this 2 ms 90 degree pulse, `|Mxy|` comes out 0.9874 instead of 1.0000: about
1.3% of transverse signal lost to T2 during the pulse itself.

* `T1`, `T2` are in **ms**. `None` (the default) means infinite, i.e. no
  relaxation. Zero or negative raises `ValueError` — zero is not shorthand for
  instantaneous.
* `M0` is the equilibrium `Mz`, default 1.0.
* Leaving both `T1` and `T2` unset takes an early return into the pure-rotation
  path, so results computed before relaxation existed do not shift by a single
  bit.
* During a pulse, propagation is Strang-split: second-order accurate in the
  time step, not exact. `PULSIM.affine_propagate` is the exact constant-field
  propagator, kept as the reference implementation and used to test the
  splitting's convergence.
* `TorchBackend` does **not** implement relaxation. Passing `T1` or `T2` to it
  raises `NotImplementedError` rather than discarding them silently.

Relaxation is available on the density-matrix side too, through a `Relaxation`
object rather than the backend:

```python
seq = PULSIM.LiouvilleSequence(segments, spin_system,
                               relaxation=PULSIM.Relaxation(T1=500.0, T2=60.0))
```

* `T1`, `T2` are in **ms**, one value for every spin or one per spin. `None`
  means infinite. With no `Relaxation` attached, propagation is bit for bit
  what it was before relaxation existed.
* Each product operator decays at a rate built from its factors — 1/T2 per
  transverse factor, 1/T1 per longitudinal one — so `Ix` relaxes at 1/T2 and
  `2IzSy` at 1/T1(I) + 1/T2(S). Longitudinal terms return to `sum_i M0_i Iz_i`.
* `max_step` (default 0.05 ms) subdivides long propagation steps. This is not
  optional detail: relaxation and coherent evolution do **not** commute once
  spins are coupled, because evolution moves amplitude between operators of
  different rate. A 30 ms `Delay` at J = 140 Hz taken in one step is wrong by
  ~9e-3. Equal `T1` and `T2` does not help — the two-spin rate is a sum either
  way.
* No relaxation superoperator, so no cross-relaxation and no NOE: rates are
  attached to operators, not derived from a mechanism.

Equations, sign conventions and the full list of validation requirements are in
[`docs/PHYSICS_SPECIFICATION.md`](docs/PHYSICS_SPECIFICATION.md) §3.

## Gradients

A gradient is a position-dependent offset, so it needs no special propagator —
just positions in, and an average out:

```python
import numpy as np
import PULSIM
from PULSIM.gradients import gradient_offsets, uniform_positions, ensemble_average

GAMMA = 42.577                                   # kHz/mT

r  = uniform_positions(length=0.01, n_positions=64)          # 10 mm slab
df = gradient_offsets(G=[0.0, 0.0, 10.0], r=r, Gamma=GAMMA)  # 10 mT/m along z

M = np.zeros((3, 64))
M[0] = 1.0                                       # transverse, uniform

t = 1.0 / (GAMMA * 10.0 * 0.01)                  # exactly one phase twist
M = PULSIM.bloch_delay(M, t, df, GAMMA)

print(ensemble_average(M))                       # ~ [0, 0, 0]: signal gone
```

Reverse the gradient for the same duration and the signal comes back exactly —
the magnetization was never destroyed, only made invisible to the receiver.
That is the difference between physical dephasing and `ideal_spoil`, which
zeroes `Mx` and `My` outright and cannot be undone.

* `G` in **mT/m**, `r` in **m**, offsets in **kHz**. No conversion factors.
* Positions are sampled at cell midpoints, so complete dephasing cancels
  exactly rather than approximately.
* Static or piecewise-constant gradients only: one call per constant segment.
  Gradient waveforms that vary *within* a single pulse are not supported.
* No diffusion, flow, concomitant fields, or gradient nonlinearity.

See [`docs/PHYSICS_SPECIFICATION.md`](docs/PHYSICS_SPECIFICATION.md) §8.

## Vendor shape files

PULSIM reads the header of a Bruker/TopSpin shape file, not just its numbers:

```python
from PULSIM.bruker import read_bruker_header

h = read_bruker_header("wave/HypSec")
print(h.exmode, h.shape_type, h.intent)   # Adiabatic Inversion adiabatic
print(h.totrot, h.bwfac)                  # 180.0 18.014
print(h.design["mu"], h.design["beta"])   # 5.92944374678314 5.29829236561048
```

`FileShape` uses this automatically: a file that declares
`##$SHAPE_EXMODE= Adiabatic` is treated as adiabatic, so the vendor decides
rather than a heuristic.

```python
from PULSIM.rf_shape import FileShape

FileShape(path="wave/HypSec",       duration=1.0).intent   # 'adiabatic'
FileShape(path="wave/Burbop-180.1", duration=0.5).intent   # None  (EXMODE=BOP)
```

* Pass `intent=` to contradict the file — including `intent=None` to clear its
  claim and force area calibration.
* An adiabatic file **cannot** be calibrated from a flip angle; a chirp has no
  meaningful pulse area. Supply the amplitude the way a spectrometer does:
  `Pulse(shape, nu1_max=<kHz>)`. `realized_q` is then `None`, because the
  adiabaticity factor is genuinely unknown for an imported file.
* `SHAPE_INTEGFAC` is exposed as `header.integfac` for inspection but is **not**
  used for calibration. Across the 203 files in `wave/` it agrees with PULSIM's
  own integral for fewer than half, in ways neither plausible normalization
  convention explains. See the specification before trusting it.
* 18 files carry the full ShapeTool `SHL_*` design block, reachable as
  `header.design`. For those, the file's own `beta` and `mu` agree with
  PULSIM's design relations to one part in 10^15.

See [`docs/PHYSICS_SPECIFICATION.md`](docs/PHYSICS_SPECIFICATION.md) §9.

## Release History

* 0.0.1
    * Work in progress
* 0.0.2
    * Updated numpy into torch for better performance
* 0.1.0
    * First version of PULSIM

## LICENSE

Distributed under the MIT license. See ``LICENSE`` for more information.
