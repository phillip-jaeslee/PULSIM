# PULSIM: pulse simulator
> Bloch equation calculator for multiple pulses

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Google Colab-F9AB00?style=flat-square&logo=Google Colab&logoColor=white"/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phillip-jaeslee/PULSIM/blob/main/PULSIM_colab_oo.ipynb)



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

Equations, sign conventions and the full list of validation requirements are in
[`docs/PHYSICS_SPECIFICATION.md`](docs/PHYSICS_SPECIFICATION.md) §3.

## Release History

* 0.0.1
    * Work in progress
* 0.0.2
    * Updated numpy into torch for better performance
* 0.1.0
    * First version of PULSIM

## LICENSE

Distributed under the MIT license. See ``LICENSE`` for more information.
