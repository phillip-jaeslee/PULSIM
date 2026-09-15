"""
bruker.py -- reading the header of a Bruker/TopSpin shape file.

file_import.py reads the ##XYPOINTS data block and discards everything above
it. That header is not decoration: it carries the vendor's own statement of 
what the pulse is for (SHAPE_EXMODE), what rotation it performs (SHAPE_TYPE),
its bandwidth factor, and -- in files written by ShapeTool -- the full design
parameter set generated the waveform.

Two tiers, and the split is in the data rather than in our imagination. All
203 shape files in wave/ carry the scalar SHAPE_* fields; 18 of them also
carry the SHL_* design block.

Written against the corpus, not against the format specification. Real files
spell the same value four different ways --

    ##$SHAPE_EXMODE= Excitation
    ##$SHAPE_EXMODE= <Excitation>
    ##$SHAPE_EXMODE= Excitation
    ##$SHAPE_EXMODE= Excitation\\

-- with angle brackets, trailing whitespace and line-continuation
backslashes all appearing in practice. Normalization is mandatory.

Note that `##$SHAPE_EXMODE= None` is the literal string "None", a real
Bruker value meaning "no excitation mode declared". It must not be confused
with a field being absent.

numpy only; no new dependencies, and safe under Pyodide.
"""

import os
import re

from dataclasses import dataclass

_FIELD = re.compile(r"^##\$([A-Za-z0-9_]+)=\s*(.*?)\s*$")
_ARRAY_HEADER = re.compile(r"^\(\d+\.\.\d+\)$")

def _normalize(value):
    """Strip the three decorations real files carry: continuation backslash,
    surrounding whitespace, surrounding angle brackets. `<>` becomes ''.
    """
    value = value.strip()
    if value.endswith("\\"):
        value = value[:-1].strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value

def parse_header(text):
    """Split a shape file's text into scalar and array header fields.

    Returns (scalars, arrays):
        scalars : {name: normalized string}
        arrays  : {name: [normalized string, ...]}
    
    A field whose value is "(0..15)" is an array declaration; its elements
    are on the following lines, whitespace-separted, until the next line
    beginning with '##'. That is how every SHL_* field is stored.
    """
    scalars, arrays = {}, {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _FIELD.match(lines[i])
        if match is None:
            i += 1
            continue

        name, raw = match.group(1), match.group(2)
        if _ARRAY_HEADER.match(raw.strip()):
            tokens, i = [], i + 1
            while i < len(lines) and not lines[i].startswith("##"):
                tokens.extend(lines[i].split())
                i += 1
            arrays[name] = [_normalize(t) for t in tokens]
        else:
            scalars[name] = _normalize(raw)
            i += 1

    return scalars, arrays

_ADIABATIC_EXMODES = {"Adiabatic", "CompositeAdiabatic"}

def _maybe_number(text):
    """float(text), or None if it isn't a number. Bruker fields are
    untyped strings, and plenty of them hold words.
    """
    try:
        return float(text)
    except (TypeError, ValueError):
        return None

@dataclass(frozen=True)
class BrukerHeader:
    """A parsed shape-file header. Immutable; every accessor returns None
    rather than raising when a field is absent, because absence is normal --
    only 155 of 203 files carry SHAPE_TYPE, only 18 carry the SHL_ block.
    """

    scalars: dict
    arrays: dict
    path: str = ""

    def text(self, name):
        """Field as a string, or None if absent or empty."""
        value = self.scalars.get(name)
        return value if value not in (None, "") else None

    def number(self, name):
        """Field as a float, or None if absent or non-numeric"""
        return _maybe_number(self.scalars.get(name))

    @property
    def exmode(self):
        """SHAPE_EXMODE -- how the pulse works: Excitation, Adiabatic,
        Universal, Decoupling, BOP, or the literal string "None".
        """
        return self.text("SHAPE_EXMODE")

    @property
    def shape_type(self):
        return self.text("SHAPE_TYPE")

    @property
    def totrot(self):
        return self.number("SHAPE_TOTROT")

    @property
    def bwfac(self):
        return self.number("SHAPE_BWFAC")

    @property
    def integfac(self):
        """Bruker's stored signed integral.

        DIAGNOSTIC ONLY -- do not calibrate from this. Measured across
        wave/, it agrees with signed_integral_of(envelope) for well under
        half the corpus, and the two normalization conventions that would
        explain it (by the waveform's own peak, by 100) each fit some files
        and contradict others. The likeliest reading is that the value
        reflects the original design and is not recomputed when a file is
        later rescaled, so for user-modified files it describes a waveform
        that is no longer in the file. See PHYSICS_SPECIFICATION.md.
        """
        return self.number("SHAPE_INTEGFAC")

    @property
    def intent(self):
        return "adiabatic" if self.exmode in _ADIABATIC_EXMODES else None

    @property
    def design(self):
        """Tier 2: the ShapeTool design parameters, or None.

        The SHL_ fields are 16-slot arrays -- ShapeTool supports up to 16
        concatenated segments -- of which slot 0 is the active one and the
        rest are padding. Keys are lowercased with the SHL_ prefix removed,
        so SHL_MU becomes "mu". Numeric values are floats, the rest strings.

        For an adiabatic pulse this carries the design inputs (trunclev, sw)
        AND Bruker's derived constants (beta, mu), which is enough to build
        an AdiabaticCalibration from the file itself rather than from
        PULSIM's defaults.
        """
        shl = {k: v for k, v in self.arrays.items() if k.startswith("SHL_")}
        if not shl:
            return None

        out = {}
        for name, values in shl.items():
            if not values or values[0] == "":
                continue
            slot0 = values[0]
            number = _maybe_number(slot0)
            out[name[len("SHL_"):].lower()] = slot0 if number is None else number

        return out

def read_bruker_header(path):
    """Parse the header of a Bruker shape file. Never raises on a file that
    is not a shape file -- it simply comes back with nothing in it."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        scalars, arrays = parse_header(handle.read())
    return BrukerHeader(scalars=scalars, arrays=arrays, path=str(path))
