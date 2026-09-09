"""
Freeze the CURRENT behaviour of the pulse engine as golden reference data.

Run this ONCE, from the PULSIM directory, BEFORE refactoring anything:

    python tests/make_golden.py

It writes tests/golden/shapes.npz. Commit that file. From then on
tests/test_rf_shape_golden.py proves the new code reproduces the old numbers
exactly, so a refactor that silently changes the physics fails loudly.

Everything below is copied verbatim from the existing implementation — the
shape_funcs dict from pulse.cpu_pulse.shaped_pulse, the RF_angle branches from
pulse.torch_pulse.torch_shaped_pulse, the ones() from hard_pulse, and the
phasor loop from cpu_pulse.import_shaped_pulse. It is deliberately NOT
refactored or tidied: its only job is to say what the code did on the day the
refactor started.

Regenerate only when you have decided a numerical change is correct, and say
so in the commit message.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.file_import import import_file                   # noqa: E402
from pulse_shape_list import *                        # noqa: E402,F403

try:
    from mat_operator import cpu_rot                  # noqa: E402
except ImportError:                                   # pragma: no cover
    # mat_operator imports torch at module level even though cpu_rot is pure
    # numpy, so this fixture cannot be generated on a torch-less machine.
    # Fallback is copied verbatim from mat_operator.cpu_rot.Rot; if that
    # function ever changes, this copy must change with it.
    class cpu_rot:
        def Rot(flip):
            Rot = np.array([[np.cos(flip), np.sin(flip)],
                            [-np.sin(flip), np.cos(flip)]])
            return Rot

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "golden")

# Representative of the real driver scripts (see test/RF_pulse_*.py):
# t_max = 0.6 ms, N = 1000 points.
T_MAX = 0.6
N = 1000

# Two imported waveforms, chosen deliberately:
#   sine.jhl      - what the real driver scripts use; phase column is 0/180 only
#   Burbop-180.1  - phase sweeps the full 0..360. Without this one the fixture
#                   cannot tell exp(-i.theta) from exp(+i.theta), because for
#                   phases of 0 and 180 they are identical. (Found by
#                   mutation-testing the suite.)
#   Bip720,50,20.1 - max phase 331.4 deg: a third, independent phase profile.
#                   It was originally chosen to pin the ">= 350 deg -> adiabatic"
#                   threshold from below; that rule has since been deleted (intent
#                   is declared, not inferred), so it now earns its place only as
#                   phase-profile variety.
WAVE_FILES = {
    "sine": "sine.jhl",
    "burbop180": "Burbop-180.1",
    "bip720": "Bip720,50,20.1",
    "badcop1": "BadCop1",   # max phase 359.90 deg — pins the threshold from above
}


def legacy_shape_funcs(t_max, N):
    """
    The shape_funcs dict, character-for-character as it appears in
    pulse.cpu_pulse.shaped_pulse (and three other places).
    """
    dt = t_max / N
    init = -N / 2
    final = N / 2
    t = np.arange(init, final, 1) * dt

    return {
        "sinc":         lambda: np.hamming(N).T * np.sinc(t),
        "cos":          lambda: np.hamming(N).T * np.cos(t),
        "sinc2p":       lambda: np.sinc(2 * np.pi * t),
        "eburp1":       lambda: E_BURP_1_pulse(duration=t_max, points=N),
        "eburp2":       lambda: E_BURP_2_pulse(duration=t_max, points=N),
        "iburp1":       lambda: I_BURP_1_pulse(duration=t_max, points=N),
        "iburp2":       lambda: I_BURP_2_pulse(duration=t_max, points=N),
        "uburp":        lambda: U_BURP_pulse(duration=t_max, points=N),
        "reburp":       lambda: RE_BURP_pulse(duration=t_max, points=N),
        "gausscasG3":   lambda: GAUSSCASCADE_G3_pulse(duration=t_max, points=N),
        "gausscasG4":   lambda: GAUSSCASCADE_G4_pulse(duration=t_max, points=N),
        "gausscasQ3":   lambda: GAUSSCASCADE_Q3_pulse(duration=t_max, points=N),
        "gausscasQ5":   lambda: GAUSSCASCADE_Q5_pulse(duration=t_max, points=N),
        "hermite":      lambda: HERMITE_pulse(duration=t_max, points=N),
        "seduce1":      lambda: SEDUCE_1_pulse(duration=t_max, points=N),
        "sneeze":       lambda: SNEEZE_pulse(duration=t_max, points=N),
        "qsneeze":      lambda: QSNEEZE_pulse(duration=t_max, points=N),
        "esnob":        lambda: eSNOB_pulse(duration=t_max, points=N),
        "i2snob":       lambda: i2SNOB_pulse(duration=t_max, points=N),
        "i3snob":       lambda: i3SNOB_pulse(duration=t_max, points=N),
        "rsnob":        lambda: rSNOB_pulse(duration=t_max, points=N),
        "dsnob":        lambda: dSNOB_pulse(duration=t_max, points=N),
        "hypsec":       lambda: HYPSEC_pulse(duration=t_max, points=N),
        "swrl11":       lambda: SWIRL11_pulse(duration=t_max, points=N),
        "swrl12":       lambda: SWIRL12_pulse(duration=t_max, points=N),
        "swrl17":       lambda: SWIRL17_pulse(duration=t_max, points=N),
    }


def legacy_rf_angle(RF_org):
    """
    The RF_angle branches from pulse.torch_pulse.torch_shaped_pulse, verbatim.
    Two code paths depending on whether the waveform is real or complex.
    """
    if np.isrealobj(RF_org):
        return np.where(RF_org >= 0, 0.0, 180.0)
    phase_rad = np.angle(RF_org)
    return (-np.degrees(phase_rad)) % 360


def legacy_file_envelope(file_path):
    """
    The phasor loop from pulse.cpu_pulse.import_shaped_pulse, verbatim.
    Returns the complex envelope.

    The old ">= 350 deg -> adiabatic" classification that used to live here is
    gone: it called Burbop-180.1 and BadCop1 adiabatic, and both are
    optimal-control pulses rather than frequency sweeps. Intent is declared by
    the shape now, never inferred from the waveform.
    """
    xy_array = import_file(file_path)
    RF_array = np.zeros(np.shape(xy_array), dtype=np.complex128)
    for k in range(len(xy_array)):
        xy_temp = cpu_rot.Rot(xy_array[k, 1] * np.pi / 180) @ np.array([1, 0]).T
        RF_array[k, 1] = complex(xy_temp[0], xy_temp[1])
    return xy_array[:, 0] * RF_array[:, 1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {"_t_max": np.array(T_MAX), "_N": np.array(N)}

    for name, func in legacy_shape_funcs(T_MAX, N).items():
        RF_org = func()
        data[f"shape/{name}"] = np.asarray(RF_org)
        data[f"phase/{name}"] = np.asarray(legacy_rf_angle(RF_org))
        print(f"  {name:12s} n={np.size(RF_org):5d}  "
              f"dtype={np.asarray(RF_org).dtype}")

    # hard pulse: legacy builds a (1, N) row vector
    data["shape/hard"] = np.ones((1, int(N)))
    print(f"  {'hard':12s} n={N:5d}  dtype=float64  (legacy shape (1, N))")

    # imported waveforms
    for key, filename in WAVE_FILES.items():
        path = os.path.join(os.path.dirname(HERE), "wave", filename)
        if not os.path.exists(path):
            print(f"  SKIPPED file import: {path} not found")
            continue
        env = legacy_file_envelope(path)
        data[f"shape/file_{key}"] = env
        data[f"phase/file_{key}"] = legacy_rf_angle(env)
        data[f"filename/file_{key}"] = np.array(filename)
        print(f"  {'file:' + key:12s} n={env.size:5d}")

    out = os.path.join(OUT_DIR, "shapes.npz")
    np.savez_compressed(out, **data)
    print(f"\nwrote {out}  ({len(data)} arrays, "
          f"{os.path.getsize(out) / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
