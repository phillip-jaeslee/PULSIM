"""
Test for Bruker/TopSpin shape-file header parsing.

Written against wave/, which is a real corpus rather than a clean one: it
contains user-modified files, composites, a subdirectory, and one file that
is not a shape at all.
"""

import os
import sys

import pytest
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PULSIM
from PULSIM.bruker import parse_header, _normalize, read_bruker_header
from PULSIM.calibration import beta_from_truncation, mu_from_sweep_width
from PULSIM.rf_shape import FileShape

GAMMA = 42.577

WAVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wave")

@pytest.mark.parametrize("raw,want", [
    ("Excitation",      "Excitation"),
    ("<Excitation>",    "Excitation"),
    ("Excitation  ",    "Excitation"),
    ("Excitation\\",    "Excitation"),
    ("<>",              ""),
    ("0\\",             "0"),
    ("None",            "None"),        # a real Bruker value, not absence
])
def test_normalization_of_the_spellings_that_actually_occur(raw, want):
    assert _normalize(raw) == want

def test_scalar_and_array_fields_are_separated():
    text = (
        "##TITLE= Parameter file\n"
        "##$SHAPE_EXMODE= <Adiabatic>\n"
        "##$SHL_MU= (0..15)\n"
        "5.92944374678314 0 0 0\n"
        "##$SHL_TYPE= (0..15)\n"
        "<Inversion> <> <>\n"
        "##MINX= 0.0\n"
        "##$SHAPE_TOTROT= 180\n"
    )
    scalars, arrays = parse_header(text)

    assert scalars["SHAPE_EXMODE"] == "Adiabatic"
    assert scalars["SHAPE_TOTROT"] == "180"
    assert "SHL_MU" not in scalars                  # arrays do not leak into scalars
    assert arrays["SHL_MU"][0] == "5.92944374678314"
    assert arrays["SHL_TYPE"] == ["Inversion", "", ""]

def test_array_terminates_on_a_plain_double_hash_line():
    """##MINX= ends an array just as ##$ does -- otherwise MINX's value
    would be swallowed into the preceding array.
    """
    text = "##$SHL_SW= (0..15)\n20 0 0\n##MINX= 0.0\n##$SHAPE_MODE= 0\n"
    scalars, arrays = parse_header(text)
    assert arrays["SHL_SW"] == ["20", "0", "0"]
    assert scalars["SHAPE_MODE"] == "0"

# -- Reading real files

def test_hypsec_header_fields():
    h = read_bruker_header(os.path.join(WAVE, "HypSec"))
    assert h.exmode == "Adiabatic"        # written <Adiabatic> in the file
    assert h.shape_type == "Inversion"
    assert h.intent == "adiabatic"
    assert h.totrot == 180.0
    assert h.bwfac == pytest.approx(18.014)
    assert h.integfac == pytest.approx(0.05439915)

def test_hypsec_design_round_trips_brukers_own_constants():
    """The file carries the design INPUTS and Bruker's DERIVED constants,
    so it is both the input and the oracle.

    PULSIM computes beta = arccosh(100/trunclev) and mu = pi*sw/(2*beta)
    from trunclev and sw read out of the file, and must land on the beta
    and mu that same file stores. Measured agreement is 8e-16 and 2e-16
    relative -- one ulp -- so the 1e-12 below is not a loose tolerance
    hiding a discrepancy.
    """
    design = read_bruker_header(os.path.join(WAVE, "HypSec")).design

    assert design["trunclev"] == 1.0
    assert design["sw"] == 20.0

    beta = beta_from_truncation(design["trunclev"])
    mu = mu_from_sweep_width(design["sw"], beta)

    assert beta == pytest.approx(design["beta"], rel=1e-12)
    assert mu == pytest.approx(design["mu"], rel=1e-12)

def test_exmode_none_is_a_string_not_an_intent():
    """`##$SHAPE_EXMODE= None` is a real Bruker value meaning "not
    declared". It must not be read as adiabatic, and must not be confused
    with the field being missing."""
    headers = [read_bruker_header(p) for p in _corpus()]
    none_valued = [h for h in headers if h.exmode == "None"]
    assert none_valued, "expected files with a literal None exmode"
    assert all(h.intent is None for h in none_valued)

def test_number_returns_none_rather_than_raising():
    h = read_bruker_header(os.path.join(WAVE, "HypSec"))
    assert h.number("SHAPE_TYPE") is None         # present but not numeric
    assert h.number("NO_SUCH_FIELD") is None      # absent
    assert h.text("NO_SUCH_FIELD") is None

def test_tier_one_files_have_no_design_block():
    h = read_bruker_header(os.path.join(WAVE, "Crp60,0.5,20.1"))
    assert h.design is None
    assert h.exmode is not None                   # tier 1 still works


# -- The corpus

def _corpus():
    return sorted(p for p in glob.glob(os.path.join(WAVE, "*")) if os.path.isfile(p))

def test_every_shape_file_parses_and_declares_an_exmode():
    """Not one of the 203 may raise, and all but the non-shape file must
    carry SHAPE_EXMODE."""
    headers = [(os.path.basename(p), read_bruker_header(p)) for p in _corpus()]
    empty = [name for name, h in headers if not h.scalars and not h.arrays]
    assert empty == ["Update_wave.info"], empty
    missing = [name for name, h in headers if h.exmode is None and name not in empty]
    assert missing == [], missing

def test_corpus_census():
    """A deliberate census, not an invariant.

    If wave/ changes -- waveforms added, removed, or relicensed -- this
    fails and someone re-confirms the numbers rather than discovering the
    drift in a lesson. Update it knowingly.
    """
    headers = [read_bruker_header(p) for p in _corpus()]
    assert len(headers) == 204                                   # incl. Update_wave.info
    assert sum(1 for h in headers if h.exmode) == 203
    assert sum(1 for h in headers if h.intent == "adiabatic") == 47
    assert sum(1 for h in headers if h.design) == 18

def test_every_design_block_carries_the_adiabatic_inputs():
    """Tier 2 is only worth having if the keys we need are actually there."""
    for path in _corpus():
        design = read_bruker_header(path).design
        if design is None:
            continue
        for key in ("mu", "beta", "sw", "trunclev", "length", "npoints"):
            assert key in design, f"{os.path.basename(path)} missing {key}"

# -- FileShape takes the file's word

@pytest.mark.parametrize("filename,exmode,intent", [
    ("HypSec",          "Adiabatic", "adiabatic"),
    ("Crp100,0.5,20.1", "Adiabatic", "adiabatic"),
    ("Burbop-180.1",    "BOP",       None),        # optimal control, not adiabatic
    ("BadCop1",         "Inversion", None),
])
def test_intent_comes_from_the_file(filename, exmode, intent):
    """The replacement for the 350-degree heuristic.

    Burbop-180.1 and BadCop1 are the two the heuristic got wrong -- it
    called them adiabatic and gave them a spurious factor of two. Bruker
    has a dedicated BOP mode for broadband optimal-control pulses and calls
    BadCop1 an Inversion, and neither is adiabatic.
    """
    shape = FileShape(path=os.path.join(WAVE, filename), duration=0.5)
    assert shape.header.exmode == exmode
    assert shape.intent == intent
    assert shape.calibration_mode == ("adiabatic" if intent else "area")

def test_an_explicit_intent_overrides_the_file_in_both_directions():
    """intent=None must mean "I say it has none", not "I didn't say".

    Those are different, which is why the default is a sentinel: a caller
    simulating a modified or mislabelled waveform has to be able to
    contradict the header, including by clearing it.
    """
    hypsec = os.path.join(WAVE, "HypSec")
    burbop = os.path.join(WAVE, "Burbop-180.1")

    assert FileShape(path=hypsec, duration=0.5).calibration_mode == "adiabatic"
    assert FileShape(path=hypsec, duration=0.5, intent=None).calibration_mode == "area"
    assert FileShape(path=burbop, duration=0.5).calibration_mode == "area"
    assert FileShape(path=burbop, duration=0.5,
                     intent="adiabatic").calibration_mode == "adiabatic"

def test_area_calibration_of_an_adiabatic_file_is_refused_usefully():
    """Refusing is correct -- a chirp has no meaningful pulse-area flip
    angle -- but the message has to name the way out, or the user is stuck."""
    shape = FileShape(path=os.path.join(WAVE, "Crp100,0.5,20.1"), duration=0.5)
    with pytest.raises(NotImplementedError, match="nu1_max"):
        shape.calibration

def test_supplying_the_amplitude_directly_still_works():
    """The escape hatch: the spectrometer sets the power, so the user can
    too. Q stays None because it is genuinely unknown, not because
    something failed."""
    shape = FileShape(path=os.path.join(WAVE, "HypSec"), duration=1.0)
    pulse = PULSIM.Pulse(shape, nu1_max=5.0,
                         backend=PULSIM.NumpyBackend(Gamma=GAMMA))

    assert pulse.nu1_max == 5.0
    assert pulse.b1_max == pytest.approx(5.0 / GAMMA)
    assert pulse.realized_q is None

def test_a_non_adiabatic_file_still_calibrates_by_area():
    """The 156 files that are not adiabatic must be entirely unaffected."""
    shape = FileShape(path=os.path.join(WAVE, "Burbop-180.1"), duration=0.5)
    pulse = PULSIM.Pulse(shape, flip=np.pi, axis="x",
                         backend=PULSIM.NumpyBackend(Gamma=GAMMA))
    assert pulse.nu1_max > 0
    assert pulse.realized_q is None