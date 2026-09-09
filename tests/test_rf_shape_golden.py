"""
Golden-file equivalence test for the RFShape layer.

Proves that rf_shape.RFShape reproduces, to floating-point exactness, what
pulse.py's shape_funcs dispatch produced before the refactor started.

    python -m pytest tests/ -v

The reference arrays come from tests/golden/shapes.npz, generated once by
tests/make_golden.py against the pre-refactor code. If a test here fails, the
refactor changed the physics — fix the code, do not regenerate the fixture.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PULSIM.rf_shape import RFShape, FileShape, HardShape, AnalyticShape  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden", "shapes.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(GOLDEN),
    reason="run `python tests/make_golden.py` first",
)


@pytest.fixture(scope="module")
def golden():
    with np.load(GOLDEN) as f:
        return {k: f[k] for k in f.files}


@pytest.fixture(scope="module")
def params(golden):
    return float(golden["_t_max"]), int(golden["_N"])


def shape_names(golden):
    return sorted(k.split("/", 1)[1] for k in golden if k.startswith("shape/"))


# --------------------------------------------------------------------------
# the actual equivalence claim
# --------------------------------------------------------------------------

ANALYTIC_NAMES = [
    "sinc", "cos", "sinc2p",
    "eburp1", "eburp2", "iburp1", "iburp2", "uburp", "reburp",
    "gausscasG3", "gausscasG4", "gausscasQ3", "gausscasQ5",
    "hermite", "seduce1", "sneeze", "qsneeze",
    "esnob", "i2snob", "i3snob", "rsnob", "dsnob",
    "hypsec", "swrl11", "swrl12", "swrl17",
]


@pytest.mark.parametrize("name", ANALYTIC_NAMES)
def test_envelope_matches_legacy(golden, params, name):
    """RFShape.create(name).envelope() == the old shape_funcs[name]()."""
    t_max, N = params
    expected = golden[f"shape/{name}"]
    actual = RFShape.create(name, duration=t_max, points=N).envelope()

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.real, np.real(expected), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(actual.imag, np.imag(expected), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("name", ANALYTIC_NAMES)
def test_phase_matches_legacy(golden, params, name):
    """
    The unified phase formula reproduces BOTH legacy branches.

    torch_shaped_pulse special-cased real waveforms as
    np.where(RF_org >= 0, 0.0, 180.0) and complex ones as
    (-degrees(angle))%360. rf_shape uses only the second. This asserts they
    were the same function all along.
    """
    t_max, N = params
    expected = golden[f"phase/{name}"]
    actual = RFShape.create(name, duration=t_max, points=N).phase_profile
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("name", ANALYTIC_NAMES)
def test_amplitude_profile_is_abs_of_envelope(golden, params, name):
    t_max, N = params
    shape = RFShape.create(name, duration=t_max, points=N)
    np.testing.assert_allclose(shape.amplitude_profile,
                               np.abs(golden[f"shape/{name}"]),
                               rtol=1e-13, atol=1e-15)


def test_hard_pulse_matches_legacy(golden, params):
    """
    Legacy hard_pulse built np.ones((1, N)) and indexed RF[0, n].
    HardShape is 1-D like every other shape; the values must still match.
    """
    t_max, N = params
    expected = golden["shape/hard"]
    actual = HardShape(duration=t_max, points=N).envelope()

    assert actual.ndim == 1, "HardShape should drop the legacy leading axis"
    np.testing.assert_allclose(actual.real, expected.ravel(), rtol=0, atol=0)
    np.testing.assert_allclose(actual.imag, 0.0, rtol=0, atol=0)


#: imported-waveform fixtures, see tests/make_golden.py for why each is here
FILE_KEYS = ["sine", "burbop180", "bip720", "badcop1"]


@pytest.mark.parametrize("key", FILE_KEYS)
def test_file_shape_matches_legacy(golden, params, key):
    """
    FileShape's exp(-i*theta) must equal the legacy cpu_rot.Rot() phasor loop
    (and the torch path's cos(-a) + i sin(-a), which is the same thing).

    Both fixtures matter: sine.jhl has a 0/180 phase column, for which
    exp(-i.theta) and exp(+i.theta) are indistinguishable, so it cannot police
    the sign convention on its own. Burbop-180.1 sweeps the full circle and can.
    """
    if f"shape/file_{key}" not in golden:
        pytest.skip(f"wave fixture {key} missing")

    t_max, _ = params
    filename = str(golden[f"filename/file_{key}"])
    wave = os.path.join(os.path.dirname(HERE), "wave", filename)
    if not os.path.exists(wave):
        pytest.skip(
            f"wave/{filename} is not distributed with the repository "
            f"(.gitignore excludes wave/ -- see audit item 0-1). "
            f"This test runs only where the waveform library is present."
        )

    expected = golden[f"shape/file_{key}"]
    shape = FileShape(path=wave, duration=t_max)

    assert shape.points == expected.size, "point count comes from the file"
    np.testing.assert_allclose(shape.envelope(), expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(shape.phase_profile, golden[f"phase/file_{key}"],
                               rtol=1e-13, atol=1e-12)
    # NOTE: golden's "adiabatic/file_*" key recorded the old ">= 350 deg phase"
    # verdict. That rule is gone -- it called Burbop-180.1 and BadCop1
    # adiabatic, and neither is a frequency sweep. Intent is declared now, so
    # the key is vestigial and will disappear the next time make_golden runs.

def test_file_fixtures_can_police_the_phasor_sign(golden):
    """
    Guard on the guard: at least one file fixture must contain a phase that is
    neither 0 nor 180, otherwise test_file_shape_matches_legacy silently stops
    testing the sign of the exponent.
    """
    interesting = False
    for key in FILE_KEYS:
        if f"phase/file_{key}" not in golden:
            continue
        ph = golden[f"phase/file_{key}"]
        if np.any(np.abs(np.sin(np.deg2rad(ph))) > 1e-6):
            interesting = True
    assert interesting, "no file fixture has an off-axis phase; sign is untested"


def test_sample_times_matches_legacy(params):
    """The arange(-N/2, N/2)*dt grid built identically in all six pulse fns."""
    t_max, N = params
    dt = t_max / N
    expected = np.arange(-N / 2, N / 2, 1) * dt
    actual = RFShape.create("eburp1", duration=t_max, points=N).sample_times()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


# --------------------------------------------------------------------------
# the registry replaces the dispatch dict
# --------------------------------------------------------------------------

def test_registry_covers_every_legacy_name():
    """Nothing that shape_funcs could build has been lost."""
    available = set(RFShape.available())
    missing = {n.lower() for n in ANALYTIC_NAMES} - available
    assert not missing, f"shapes dropped in the refactor: {sorted(missing)}"


def test_registry_is_case_insensitive(params):
    t_max, N = params
    a = RFShape.create("gausscasQ5", duration=t_max, points=N).envelope()
    b = RFShape.create("gausscasq5", duration=t_max, points=N).envelope()
    np.testing.assert_array_equal(a, b)


def test_unknown_shape_lists_the_alternatives():
    with pytest.raises(ValueError, match="Unknown shape"):
        RFShape.create("definitely_not_a_pulse")
    try:
        RFShape.create("definitely_not_a_pulse")
    except ValueError as exc:
        assert "eburp1" in str(exc), "error should list what IS available"


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        class Clash(AnalyticShape):
            name = "eburp1"


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        RFShape(duration=1.0, points=100)


# --------------------------------------------------------------------------
# validation the old positional-argument functions had no room for
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"duration": 0},
    {"duration": -1.0},
    {"duration": "0.6"},
    {"points": 1},
    {"points": 10.5},
    {"points": "1000"},
    {"amplitude": None},
])
def test_invalid_construction_is_rejected(kwargs):
    base = {"duration": 0.6, "points": 1000}
    base.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        RFShape.create("eburp1", **base)


def test_mutating_a_parameter_invalidates_the_cache(params):
    t_max, N = params
    shape = RFShape.create("eburp1", duration=t_max, points=N)
    first = shape.envelope().copy()
    shape.points = N // 2
    assert shape.envelope().size == N // 2
    assert first.size == N


def test_envelope_is_cached(params):
    t_max, N = params
    shape = RFShape.create("uburp", duration=t_max, points=N)
    assert shape.envelope() is shape.envelope()


def test_file_shape_rejects_explicit_points():
    wave = os.path.join(os.path.dirname(HERE), "wave", "sine.jhl")
    if not os.path.exists(wave):
        pytest.skip("wave/sine.jhl missing")
    with pytest.raises(TypeError, match="points"):
        FileShape(path=wave, duration=0.6, points=1000)


def test_adiabatic_intent_is_declared_not_guessed(params):
    """Replaces test_is_adiabatic_flags_complex_waveforms.

    The old rule inferred "adiabatic" from the waveform: a complex envelope for
    analytic shapes, peak phase >= 350 deg for files. Both were wrong on real
    data -- sincos sweeps only 459 deg non-monotonically and is not a chirp,
    and the 350 deg rule flagged Burbop-180.1 and BadCop1, which are
    optimal-control pulses. Intent is declared by the shape now.
    """
    t_max, N = params
    hypsec = RFShape.create("hypsec", duration=t_max, points=N)
    eburp1 = RFShape.create("eburp1", duration=t_max, points=N)

    assert hypsec.intent == "adiabatic"
    assert eburp1.intent is None
    assert hypsec.calibration_mode == "adiabatic"
    assert eburp1.calibration_mode == "area"


def test_file_shape_does_not_guess_intent():
    """A file says nothing about its intent unless told -- and a caller can tell
    it. Burbop-180.1 is the case that matters: the old 350 deg rule called it
    adiabatic, and it is an optimal-control pulse, not a sweep."""
    wave = os.path.join(os.path.dirname(HERE), "wave", "Burbop-180.1")
    if not os.path.exists(wave):
        pytest.skip("wave/Burbop-180.1 is not distributed with the repository (item 0-1)")

    assert FileShape(path=wave, duration=0.6).intent is None
    assert FileShape(path=wave, duration=0.6).calibration_mode == "area"
    assert FileShape(path=wave, duration=0.6, intent="adiabatic").calibration_mode == "adiabatic"