"""
characterize_calibration.py --  record what PULSIM's calibration does

This is NOT a correctness test. It is a snapshot taken before the calibration
redesign, so that every later change can be checked against it: which numbers
moved, and which did not.

Run from the repo root: python tests/characterize_calibration.py
"""

import os
import sys
import json
import datetime

import numpy as np
import scipy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PULSIM

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden", "calibration_baseline.json")

# Fixed so the snapshot is reproducible on my machine
DURATION = 2.0          # ms
POINTS = 1000       
GAMMA = 42.577478518    # kHz/mT, 1H
FLIPS_DEG = (90, 180)

def measure(name):
    """Everything worth remembering about one shape's calibration today."""
    record = {}

    try:
        shape = PULSIM.RFShape.create(name, duration=DURATION, points=POINTS)
    except Exception as e:
        return {"build_error": f"{type(e).__name__}: {e}"}

    env = shape.envelope()
    # NOTE: the committed calibration_baseline.json was captured BEFORE the
    # calibration redesign and records the behaviour of that older code. It is a
    # historical record -- do not regenerate it. This script is kept runnable so
    # a fresh snapshot can be taken at a later checkpoint if one is wanted.
    record["modulation"] = "complex" if np.any(env.imag != 0) else "real"
    record["intent"] = shape.intent
    record["calibration_mode"] = shape.calibration_mode
    record["signed_integral"] = float(np.real(env.sum()) / len(env) / np.abs(env).max())
    record["cancellation_ratio"] = float(np.abs(env.sum()) / np.abs(env).sum())

    for flip_deg in FLIPS_DEG:
        key = f"flip_{flip_deg}"
        try:
            pulse = PULSIM.Pulse(
                shape,
                flip = np.deg2rad(flip_deg),
                axis = "x",
                backend=PULSIM.NumpyBackend(Gamma=GAMMA)
            )
            rf = pulse.calibrated_rf()

            M = np.zeros((3, 1))
            M[2] = 1.0
            out = pulse.apply(M, np.array([0.0]))

            record[key] = {
                "peak_rf_mT": float(np.abs(rf).max()),
                "achieved_flip_deg": float(
                    np.degrees(np.arctan2(np.hypot(out[0, 0], out[1, 0]), out[2, 0]))
                ),
                "mz": float(out[2, 0]),
            }
        except Exception as e:
            record[key] = {"error": f"{type(e).__name__}: {e}"}

    return record

def main():
    names = sorted(n for n in PULSIM.RFShape.available() if n not in ("file", "composite"))

    results = {name: measure(name) for name in names}

    payload = {
        "_note": ("Snapshot of PULSIM calibration behaviour BEFORE the "
                  "calibration redesign. This records what the code does, "
                  "not what it should do. Not a correctness assertion."),
        "_generated": datetime.date.today().isoformat(),
        "_settings": {
            "duration_ms": DURATION,
            "points": POINTS,
            "gamma_kHz_per_mT": GAMMA,
            "flips_deg": list(FLIPS_DEG),
        },
        "_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "shapes": results,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"wrote {OUT}")
    print(f"{len(results)} shapes recorded")

if __name__ == "__main__":
    main()