# tests/ — safety net for the OO refactor

Not to be confused with `test/` (singular), which holds the driver scripts.

## Workflow

```bash
cd PULSIM
python tests/make_golden.py      # ONCE, before refactoring. Commit the .npz.
python -m pytest tests/ -v       # after every change
```

`make_golden.py` runs the **pre-refactor** code — the `shape_funcs` dict from
`pulse.py`, the `RF_angle` branches from `torch_shaped_pulse`, the `ones((1,N))`
from `hard_pulse`, the `cpu_rot.Rot` phasor loop from `import_shaped_pulse` —
and freezes the resulting arrays into `golden/shapes.npz`.

`test_rf_shape_golden.py` then asserts `rf_shape.py` reproduces those arrays to
1e-13. If it fails, the refactor changed the physics. **Fix the code, don't
regenerate the fixture.** Regenerate only when you've decided a numerical change
is correct — and say so in the commit message.

## Fixtures, and why each one is there

The four imported waveforms are not arbitrary. Each was added because a
deliberately-broken version of `rf_shape.py` passed the suite without it:

| fixture | peak phase | what it catches |
|---|---|---|
| `sine.jhl` | 180° | the everyday case the driver scripts use |
| `Burbop-180.1` | 360.00° | sign of the phasor exponent — with 0/180 phases only, `exp(-iθ)` and `exp(+iθ)` are identical |
| `Bip720,50,20.1` | 331.41° | the `>= 350` adiabatic threshold, from below |
| `BadCop1` | 359.90° | the same threshold, from above |

Two meta-tests (`test_file_fixtures_can_police_the_phasor_sign`,
`test_file_fixtures_straddle_the_adiabatic_threshold`) assert these properties
still hold, so swapping a fixture for a blander one fails loudly instead of
quietly reducing coverage.

## Verifying the suite still has teeth

The suite was checked by mutation testing — introduce a bug, confirm something
fails. All ten of these are currently caught:

| mutation | tests failed |
|---|---|
| phase sign flipped | 4 |
| `FileShape` phasor sign flipped | 3 |
| `sample_times` grid changed to `arange(0, N)` | 8 |
| adiabatic threshold 350 → 250 | 1 |
| adiabatic threshold 350 → 359.99 | 1 |
| `HardShape` amplitude dropped | 1 |
| `points` setter stops invalidating the cache | 1 |
| `duration > 0` validation removed | 2 |
| registry stops lowercasing names | 13 |
| `FileShape` ignores `points=` instead of raising | 1 |

Worth repeating when you add the `Backend` layer.

## Known wart

Importing `file_import` executes `equ2shape(...)` at module level
(`file_import.py:121-124`) and writes `untitled.csv` into the working
directory. Every run of these tests re-creates that file. Left alone
deliberately — it is pre-existing behaviour and fixing it is a separate change.
