"""
rf_shape.py — the RFShape layer.

Step 1 of the object-oriented refactor of the Bloch pulse engine.

WHAT THIS REPLACES
------------------
Today the 26-entry ``shape_funcs`` dict is copy-pasted in four places:

    pulse.cpu_pulse.shaped_pulse
    pulse.torch_pulse.torch_shaped_pulse
    bloch_pulse_simulation.sim_shaped_pulse
    (and the file-import path is a *fifth*, separate code path)

Here they collapse into one registry::

    shape = RFShape.create("gausscasQ5", duration=0.6, points=1000)
    shape = RFShape.create("hard",       duration=0.02, points=1000)
    shape = RFShape.create("file",       duration=0.6, path="wave/sine.jhl")

    shape.envelope()    # complex ndarray, length == points
    shape.amplitude_profile   # |envelope|
    shape.phase_profile       # degrees, 0..360, existing sign convention
    shape.is_adiabatic        # drives the historical "x2" scaling

DELIBERATE NON-GOAL
-------------------
This module does **not** change any numbers.  The analytic subclasses delegate
to the existing functions in ``pulse_shape_list.py`` so that the golden test in
``tests/`` can prove bit-for-bit equivalence with the current engine.  Moving
each function body into its subclass is a later, separately-verifiable step —
each class carries a TODO marking where its math belongs.

NOT YET WIRED IN
----------------
Nothing imports this module yet.  ``pulse.py`` and ``bloch_pulse_simulation.py``
are untouched and keep working exactly as before.  They get migrated in step 3,
once ``Backend`` exists.
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import CubicSpline
from PULSIM.file_import import import_file


__all__ = ["RFShape", "AnalyticShape", "HardShape", "FileShape"]


# --------------------------------------------------------------------------
# base
# --------------------------------------------------------------------------

class RFShape(ABC):
    """
    An RF pulse envelope, normalised to unit peak amplitude.

    A shape knows its own waveform and nothing else.  It does not know the flip
    angle, the gyromagnetic ratio, the offset grid, or how to rotate a
    magnetization vector — those belong to ``Pulse`` and ``Backend``.

    Subclasses register themselves by declaring a ``name`` in the class body::

        class MyShape(AnalyticShape):
            name = "myshape"
            _func = staticmethod(my_pulse_function)

    (``name`` is a class attribute rather than a ``class Foo(Base, name=...)``
    keyword because ABCMeta.__new__ already takes a positional ``name``, and
    the two collide.)
    """

    #: registry key; declaring one in a subclass body registers that subclass
    name: str | None = None

    #: per-shape defaults, overridable by subclasses (HYPSEC wants duration=4.0)
    default_duration: float = 1.0
    default_points: int = 1000

    _registry: dict[str, type["RFShape"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # cls.__dict__, not cls.name: an inherited name must not re-register
        declared = cls.__dict__.get("name")
        if declared is None:
            return  # intermediate/abstract subclass, not user-selectable
        key = declared.lower()
        if key in RFShape._registry:
            raise ValueError(
                f"shape name {declared!r} already registered by "
                f"{RFShape._registry[key].__name__}"
            )
        cls.name = key
        RFShape._registry[key] = cls

    def __init__(self, duration=None, points=None, amplitude=1.0, **params):
        self.duration = self.default_duration if duration is None else duration
        self.points = self.default_points if points is None else points
        self.amplitude = amplitude
        self.params = params
        self._envelope = None  # lazily computed, then cached

    # -- validated attributes ------------------------------------------------
    # (kept as explicit properties for now; once Pulse and Magnetization exist
    #  these collapse onto the reusable descriptors in nmrsim/_descriptors.py)

    @property
    def duration(self):
        """Pulse length, in the same time unit used throughout (ms)."""
        return self._duration

    @duration.setter
    def duration(self, value):
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"duration must be a real number, not {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"duration must be positive, got {value}")
        self._duration = float(value)
        self._envelope = None

    @property
    def points(self):
        """Number of samples in the waveform."""
        return self._points

    @points.setter
    def points(self, value):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            if not (isinstance(value, numbers.Real) and float(value).is_integer()):
                raise TypeError(f"points must be an integer, not {type(value).__name__}")
        value = int(value)
        if value < 2:
            raise ValueError(f"points must be >= 2, got {value}")
        self._points = value
        self._envelope = None

    @property
    def amplitude(self):
        """Peak amplitude the normalised waveform is scaled to."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"amplitude must be a real number, not {type(value).__name__}")
        self._amplitude = float(value)
        self._envelope = None

    # -- the one thing subclasses must supply --------------------------------

    @abstractmethod
    def _build(self) -> np.ndarray:
        """Return the raw waveform.  May be real or complex, length == points."""

    # -- derived, shared by every shape --------------------------------------

    def envelope(self) -> np.ndarray:
        """
        The waveform as a complex array, cached.

        Real-valued shapes are returned as complex with zero imaginary part so
        that every downstream consumer sees one dtype and the amplitude/phase
        split below needs no special-casing.
        """
        if self._envelope is None:
            raw = np.asarray(self._build())
            self._envelope = raw.astype(np.complex128)
        return self._envelope

    @property
    def amplitude_profile(self) -> np.ndarray:
        """|envelope| — what gets plotted as "RF (mT)"."""
        return np.abs(self.envelope())

    @property
    def phase_profile(self) -> np.ndarray:
        """
        Instantaneous phase in degrees, wrapped to [0, 360).

        This is the single formula that replaces the two branches in
        ``torch_pulse.torch_shaped_pulse``.  Those branches are in fact the
        same function: for a real waveform ``np.angle`` returns 0 or pi, so
        ``(-degrees(angle)) % 360`` yields exactly the 0.0 / 180.0 that
        ``np.where(RF_org >= 0, 0.0, 180.0)`` produced.  The golden test
        asserts this equality rather than taking it on trust.
        """
        return (-np.degrees(np.angle(self.envelope()))) % 360.0

    @property
    def is_adiabatic(self) -> bool:
        """
        Whether this shape sweeps phase through a full turn.

        Names the condition that currently drives the unexplained ``* 2``
        scaling in three different places.  For analytic shapes that is
        "the waveform is complex"; ``FileShape`` overrides it with the
        historical ">= 350 degrees somewhere in the phase column" rule.
        """
        return bool(np.any(np.abs(self.envelope().imag) > 0))

    def sample_times(self) -> np.ndarray:
        """
        Time grid the engine steps over: ``arange(-N/2, N/2) * dt``.

        Reproduces the grid built identically in all six pulse functions.
        Note it is centred on zero, not starting at zero — the analytic shape
        functions in pulse_shape_list build their own ``linspace(0, duration)``
        internally, so the two grids coexist today.  Preserved as-is.
        """
        dt = self.duration / self.points
        return np.arange(-self.points / 2, self.points / 2, 1) * dt

    @property
    def dt(self) -> float:
        return self.duration / self.points

    # -- registry ------------------------------------------------------------

    @classmethod
    def create(cls, name: str, **kwargs) -> "RFShape":
        """Build a shape by registry name. Replaces the shape_funcs dict."""
        try:
            subclass = RFShape._registry[name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown shape {name!r}. Available shapes: {cls.available()}"
            ) from None
        return subclass(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(RFShape._registry)

    def __repr__(self):
        return (f"{type(self).__name__}(name={self.name!r}, "
                f"duration={self.duration}, points={self.points})")

    def __eq__(self, other):
        if not isinstance(other, RFShape):
            return NotImplemented
        return (type(self) is type(other)
                and self.duration == other.duration
                and self.points == other.points
                and self.amplitude == other.amplitude
                and self.params == other.params)


# --------------------------------------------------------------------------
# analytic shapes — thin wrappers over pulse_shape_list, for now
# --------------------------------------------------------------------------

class AnalyticShape(RFShape):
    """
    A shape defined by a closed-form expression of (duration, points).

    ``_func`` points at the existing implementation in ``pulse_shape_list.py``.
    Step 2 of the refactor moves each function body into its own ``_build``
    and drops ``_func``; doing it that way keeps the golden test green at
    every intermediate commit.
    """

    _func = None

    def _build(self):
        if self._func is None:
            raise NotImplementedError(f"{type(self).__name__} defines no _func")
        return self._func(
            duration=self.duration,
            points=self.points,
            amplitude=self.amplitude,
            **self.params,
        )


# --- BURP family ----------------------------------------------------------

class EBurp1(AnalyticShape):
    name = "eburp1"
    """E-BURP-1 excitation pulse (Geen & Freeman, JMR 93, 93 (1991))."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
        A = np.array([0.88, -1.04, -0.24, 0.14, 0.03, 0.04, -0.03, 0.00])
        B = np.array([-0.40, -1.42, 0.77, 0.06, 0.03, -0.04, -0.02, 0.01])

        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.23

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude # Scale to desired amplitude

        return amp

class EBurp2(AnalyticShape):
    name = "eburp2"
    """E-BURP-2 excitation pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
        A = np.array([0.91, 0.29, -1.28, -0.05, 0.04, 0.02, 0.06, 0.00, -0.02, 0.00])
        B = np.array([-0.16, -1.82, 0.18, 0.42, 0.07, 0.07, -0.01, -0.04, 0.00, 0.00])

        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.26

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude # Scale to desired amplitude

        return amp



class IBurp1(AnalyticShape):
    name = "iburp1"
    """I-BURP-1 inversion pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
        A = np.array([0.70, -0.15, -0.94, 0.11, -0.02, -0.04, 0.01, -0.02, -0.01])
        B = np.array([-1.54, 1.01, -0.24, -0.04, 0.08, -0.04, -0.01, 0.01, -0.01])

        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.50

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude # Scale to desired amplitude

        return amp


class IBurp2(AnalyticShape):
    name = "iburp2"
    """I-BURP-2 inversion pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
        A = np.array([0.81, 0.07, -1.25, -0.24, 0.07, 0.11, 0.05, -0.02, -0.03, -0.02, 0.00])
        B = np.array([-0.68, -1.38, 0.20, 0.45, 0.23, 0.05, -0.04, -0.04, 0.00, 0.01, 0.01])

        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.50

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude # Scale to desired amplitude

        return amp


class UBurp(AnalyticShape):
    name = "uburp"
    """U-BURP universal rotation pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
        A = np.array([-1.42, -0.37, -1.84, 4.40, -1.19, 0.00, -0.37, 0.50, -0.31, 0.18, -0.21, 0.23, -0.12, 0.07, -0.06, 0.06, -0.04, 0.03, -0.02, 0.02])


        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.27

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude # Scale to desired amplitude

        return amp


class ReBurp(AnalyticShape):
    name = "reburp"
    """RE-BURP refocusing pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (e.g., Geen & Freeman, 1991)
        A = np.array([-1.02, 1.11, -1.57, 0.83, -0.42, 0.26, -0.16, 0.10, -0.07, 0.04, -0.03, 0.01, -0.02, 0.00, -0.01])


        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.49

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude # Scale to desired amplitude

        return amp


# --- Gaussian cascades ----------------------------------------------------

class GaussCascadeG3(AnalyticShape):
    name = "gausscasg3"
    """G3 Gaussian cascade (Emsley & Bodenhausen, CPL 165, 469 (1990))."""
    default_points = 100  # matches GAUSSCASCADE_G3_pulse's own default
    def _build(self):
        t = np.linspace(0, self.duration, self.points)

        # G(-270)G(270)G(180) from L. Emsley & G. Bodenhausen, Chem. Phys. Lett. 165, 469 (1990).
        t_half = np.array([18.9, 18.3, 24.3]) / 100 * self.duration / 2
        t_max = np.array([28.7, 50.8, 79.5]) / 100 * self.duration
        omega_max = np.array([-1.00, 1.37, 0.49])

        a = np.log(2) / (t_half **2)

        amp = np.zeros_like(t)
        for n in range(len(t_half)):
            amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

        amp /= np.max(np.abs(amp))  # Normalize to max 1
        amp *= self.amplitude       # Apply desired scaling

        return amp


class GaussCascadeG4(AnalyticShape):
    name = "gausscasg4"
    """G4 Gaussian cascade."""
    def _build(self):
        t = np.linspace(0, self.duration, self.points)

        # G(-270)G(270)G(180)G(90) from L. Emsley & G. Bodenhausen, Chem. Phys. Lett. 165, 469 (1990).
        t_half = np.array([17.2, 12.9, 11.9, 13.9]) / 100 * self.duration / 2
        t_max = np.array([17.7, 49.2, 65.3, 89.2]) / 100 * self.duration
        omega_max = np.array([0.62, 0.72, -0.91, -0.33])

        a = np.log(2) / (t_half **2)

        amp = np.zeros_like(t)
        for n in range(len(t_half)):
            amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

        amp /= np.max(np.abs(amp))  # Normalize to max 1
        amp *= self.amplitude       # Apply desired scaling
        
        return amp

class GaussCascadeQ3(AnalyticShape):
    name = "gausscasq3"
    """Q3 Gaussian cascade refocusing pulse."""
    def _build(self):
        t = np.linspace(0, self.duration, self.points)

        # Parameters from literature (L. Emsley & G. Bodenhausen, J. Magn. Reson. 97, 135-148 (1992).)
        t_half = np.array([18.0, 18.3, 24.5]) / 100 * self.duration / 2
        t_max = np.array([30.6, 54.5, 80.4]) / 100 * self.duration
        omega_max = np.array([-4.39, 4.57, 2.60])

        a = np.log(2) / (t_half **2)

        amp = np.zeros_like(t)
        for n in range(len(t_half)):
            amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

        amp /= np.max(np.abs(amp))  # Normalize to max 1
        amp *= self.amplitude       # Apply desired scaling
        
        return amp


class GaussCascadeQ5(AnalyticShape):
    name = "gausscasq5"
    """Q5 Gaussian cascade excitation pulse."""
    def _build(self):
        t = np.linspace(0, self.duration, self.points)

        # Parameters from literature (L. Emsley & G. Bodenhausen, J. Magn. Reson. 97, 135-148 (1992).)
        t_half = np.array([18.6, 13.9, 14.3, 29.0, 13.7]) / 100 * self.duration / 2
        t_max = np.array([16.2, 30.7, 49.7, 52.5, 80.3]) / 100 * self.duration
        omega_max = np.array([-1.48, -4.34, 7.33, -2.30, 5.66])

        a = np.log(2) / (t_half **2)

        amp = np.zeros_like(t)
        for n in range(len(t_half)):
            amp += omega_max[n] * np.exp(-a[n] * (t - t_max[n])**2)

        amp /= np.max(np.abs(amp))  # Normalize to max 1
        amp *= self.amplitude       # Apply desired scaling
        
        return amp


# --- misc selective -------------------------------------------------------

class Hermite(AnalyticShape):
    name = "hermite"
    """Hermite-shaped pulse. Extra params: coefficient, truncate."""
    def _build(self):
        coefficient = self.params.get('coefficient', 1.0)   # get parameters
        truncate = self.params.get('truncate', 1)           # get parameters

        t_original = np.linspace(-3, 3, 1000)   # High-res for accurate thresholding
        truncate /=100                          # change threshold into percentage

        # Pulse definition from literature
        T = 1 / coefficient
        amp = (1 - (1.782 * (t_original / T) ** 2)) * np.exp(-(t_original / T) ** 2)

        amp /= np.max(np.abs(amp))  # Normalize
        amp *= self.amplitude

        # Find where amp ≥ threshold
        mask = abs(amp) >= truncate
        t_start = t_original[mask][0]
        t_end = t_original[mask][-1]

        # Regenerate t from t_start to t_end
        t = np.linspace(t_start, t_end, self.points)
        amp = (1 - (0.956 * (t / T) ** 2)) * np.exp(-(t / T) ** 2)
        amp /= np.max(np.abs(amp))
        amp *= self.amplitude
        return amp


class Seduce1(AnalyticShape):
    name = "seduce1"
    """SEDUCE-1 decoupling pulse."""
    def _build(self):
        t = np.linspace(-0.5, 0.5, self.points)

        # Parameters from literature (M.A. McCoy & L. Mueller, J. Magn. Reson. A 101, 122-130 (1993).)
        c = 10 * np.tanh(0.08 * abs(t))**2
        amp = np.sin(np.pi* (t+0.5))**2 / np.cosh(500 * t * c)
        return amp        


class Sneeze(AnalyticShape):
    name = "sneeze"
    """SNEEZE excitation pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (J. M. Nuzillard & R. Freeman, J. Magn. Reson. A 110, 252-256 (1994).)
        A = np.array([0.730, 1.091, -0.975, -1.038, -0.047, 0.083, -0.001, 0.036, 0.061, 0.005, -0.026, -0.013])
        B = np.array([0.001, -0.927, -1.706, 0.399, 0.454, 0.089, 0.036, 0.052, -0.017, -0.052, -0.020, 0.001])
        
        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.248

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


class QSneeze(AnalyticShape):
    name = "qsneeze"
    """Q-SNEEZE pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (Kupce, E. & Freeman, R. (1995) J. Magn. Reson., Ser. A112, 134−137.)
        A = np.array([0.934, 0.180, -1.527, 0.003, 0.143, 0.050, 0.072, -0.015, -0.040, -0.005])
        B = np.array([-0.197, -1.772, 0.204, 0.619, 0.076, 0.039, -0.025, -0.060, 0.005, 0.017])
        
        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.250

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


# --- SNOB family ----------------------------------------------------------

class ESnob(AnalyticShape):
    name = "esnob"
    """e-SNOB excitation pulse (Kupce, Boyd & Campbell, JMR B 106, 300 (1995))."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
        A = np.array([-0.6176, -0.0373, -0.0005, -0.0182, -0.0058, -0.0036, -0.0051, -0.0031, -0.0017])
        B = np.array([-0.4855, 0.1260, -0.0191, -0.0005, -0.0003, 0.0017, -0.0013, 0.0001, -0.0025])
      
        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.7500

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp



class I2Snob(AnalyticShape):
    name = "i2snob"
    """i2-SNOB inversion pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
        A = np.array([-0.2687, -0.2972, 0.0989, -0.0010, -0.0168, 0.0009, -0.0017, -0.0013, -0.0014])
        B = np.array([-1.1461, 0.4016, 0.0736, -0.0307, 0.0079, 0.0062, 0.0003, -0.0002, 0.0009])
      
        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.5000

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


class I3Snob(AnalyticShape):
    name = "i3snob"
    """i3-SNOB inversion pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
        A = np.array([0.2801, -0.9995, 0.1928, 0.0967, -0.0480, -0.0148, 0.0088, -0.0002, -0.0030])
        B = np.array([-1.1990, 0.4893, 0.2439, -0.0816, -0.0409, 0.0234, 0.0036, -0.0042, 0.0001])
     
        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.5000

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


class RSnob(AnalyticShape):
    name = "rsnob"
    """r-SNOB refocusing pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
        A = np.array([-1.1472, 0.5572, -0.0829, 0.0525])
        B = np.array([0.0000, 0.0000, 0.0000, 0.0000])

        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.5000

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


class DSnob(AnalyticShape):
    name = "dsnob"
    """d-SNOB pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (E. Kupce, J. Boyd & I.D. Campbell, J. Magn. Reson. B 106, 300-303 (1995).)
        A = np.array([-1.0662, 0.4466, 0.1673, -0.0049, -0.0753, 0.0001, 0.0144, 0.0041])
        B = np.array([-2.4513, -0.2442, 0.6025, 0.1362, -0.0521, -0.0210, 0.0070, 0.0079])
    
        omega = 2 * np.pi / self.duration
        amp = np.zeros_like(t) + 0.5000

        for n in range(len(A)):
            amp += A[n] * np.cos((n+1) * omega * t) + B[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


# --- SWIRL ----------------------------------------------------------------

class Swirl11(AnalyticShape):
    name = "swrl11"
    """SWIRL-11 pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (H. Geen & J.-M. Boehlen, J. Magn. Reson. 125, 376-382 (1997)..)
        omega_n = np.array([1.58, 1.70, 1.90, 1.86, 1.81, 1.25])
        phi = np.array([-1.00, 0.99, 1.01, -0.94, 0.99, 0.97])  
        omega = 2 * np.pi / self.duration

        A_n = 2 * np.abs(omega_n) * np.cos(phi)
        B_n = -2 * np.abs(omega_n) * np.sin(phi)

        amp = np.zeros_like(t)

        for n in range(len(omega_n)):
            amp += A_n[n] * np.cos((2*n+1) * omega * t) + B_n[n] * np.sin((2*n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


class Swirl12(AnalyticShape):
    name = "swrl12"
    """SWIRL-12 pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (H. Geen & J.-M. Boehlen, J. Magn. Reson. 125, 376-382 (1997)..)
        omega_n = np.array([1.61, 0.01, 1.68, 0.01, 1.95, 0.02, 1.76, 0.05, 2.04, 0.01, 1.41, 0.22])
        phi = np.array([-1.01, -1.86, 0.99, -2.16, 1.04, 1.30, -1.03, 1.10, 1.06, -2.11, 1.04, 1.32])
        omega = 2 * np.pi / self.duration

        A_n = 2 * np.abs(omega_n) * np.cos(phi)
        B_n = -2 * np.abs(omega_n) * np.sin(phi)

        amp = np.zeros_like(t)

        for n in range(len(omega_n)):
            amp += A_n[n] * np.cos((n+1) * omega * t) + B_n[n] * np.sin((n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


class Swirl17(AnalyticShape):
    name = "swrl17"
    """SWIRL-17 pulse."""
    def _build(self):

        t = np.linspace(0, self.duration, self.points) # build space for time array

        # Fourier coefficients from literature (H. Geen & J.-M. Boehlen, J. Magn. Reson. 125, 376-382 (1997)..)
        omega_n = np.array([1.48, 1.59, 1.62, 1.49, 1.58, 1.82, 2.05, 1.87, 1.36])
        phi = np.array([-0.96, 0.98, 1.00, -1.00, 1.00, 1.03, -0.93, 0.92, 1.16])
        omega = 2 * np.pi / self.duration

        A_n = 2 * np.abs(omega_n) * np.cos(phi)
        B_n = -2 * np.abs(omega_n) * np.sin(phi)

        amp = np.zeros_like(t)

        for n in range(len(omega_n)):
            amp += A_n[n] * np.cos((2*n+1) * omega * t) + B_n[n] * np.sin((2*n+1) * omega * t)

        amp = amp / np.max(np.abs(amp)) # Normalize to max 1
        amp *= self.amplitude           # Scale to desired amplitude

        return amp


# --- adiabatic ------------------------------------------------------------

class HyperbolicSecant(AnalyticShape):
    name = "hypsec"
    """
    Adiabatic hyperbolic secant (Silver, Joseph & Hoult, JMR 59, 347 (1984)).

    Complex-valued, so ``is_adiabatic`` is True and the historical ``* 2``
    scaling applies once ``Pulse`` takes over calibration.

    Extra params: truncate, sweepwidth, beta, mu, low_to_high.
    """
    default_duration = 4.0  # matches HYPSEC_pulse's own default
    def _build(self):
        truncate = self.params.get('truncate', 1)
        beta = self.params.get('beta', 5.29829)
        mu = self.params.get('mu', 5.92944)
        low_to_high = self.params.get('low_to_high', True)

        t_original = np.linspace(-3, 3, 10_000)
        truncate /= 100
        if low_to_high == False:
            amp_original = (np.cosh(beta * t_original))**(1 + 1j * mu)  # Complex amplitude
        elif low_to_high == True:
            amp_original = (np.cosh(beta * t_original))**(-1 - 1j * mu)

        real = np.real(amp_original)
        imag = np.imag(amp_original)
        amplitude = np.hypot(real, imag)
        mask = abs(amplitude) > truncate
        t_start = t_original[mask][0]
        t_end = t_original[mask][-1]

        t = np.linspace(t_start, t_end, self.points)
        if low_to_high == False:
            amp = (np.cosh(beta * t))**(1 + 1j * mu)  # Complex amplitude
        elif low_to_high == True:
            amp = (np.cosh(beta * t))**(-1 - 1j * mu)
        amp = amp / np.max(np.abs(amp))

        return amp
    

        


# NOTE on pulse_shape_list.SINCOS_pulse and .sincospulse:
# Neither is registered as-is. SINCOS_pulse's docstring is copy-pasted from
# HypSec (cites Silver/Joseph/Hoult, the *hyperbolic secant* paper) and its
# formula (sin(...)**(1+1j)) doesn't implement the Bruker "SinCos" shape at
# all -- it's a mislabeled leftover, not fixable by a small edit.
# sincospulse is the real SinCos implementation (params: phasefac, fullpass,
# swdir match Bruker's st generate SinCos factor/full-half/sweepDir), but it
# has two bugs: the frequency-modulation line uses sin(beta) even though its
# own comment says "PHASEFAC * cos(t)"; and the complex envelope
# (real + 1j*imag) is computed and then discarded -- the function returns the
# bare real amplitude instead. The full/half-passage sweep ranges are also
# swapped relative to the standard adiabatic definition (half passage should
# sweep the auxiliary angle 0->pi/2, full passage 0->pi; the old code did
# neither correctly). Fixed below as SinCos, following Bendall & Pegg,
# J. Magn. Reson. 67, 376-381 (1986).
class SinCos(AnalyticShape):
    name = "sincos"
    """SinCos adiabatic pulse (M.R. Bendall & D.T. Pegg, J. Magn. Reson. 67,
    376-381 (1986)). Extra params: factor (phase amplitude factor, 0-16,
    default 4.0), full_passage (bool, default True), sweep_dir (+1 = high to
    low field, -1 = low to high field, default 1)."""
    def _build(self):
        factor = self.params.get('factor', 4.0)
        full_passage = self.params.get('full_passage', True)
        sweep_dir = self.params.get('sweep_dir', 1)

        t = np.linspace(0, self.duration, self.points)
        x = t / self.duration                          # 0 -> 1 across the pulse

        # Auxiliary sweep angle: half passage sweeps 0->pi/2 (offset goes
        # from max to 0, amplitude rises 0->max); full passage sweeps 0->pi
        # (offset goes from max through 0 to -max, amplitude rises 0->max->0).
        beta_max = np.pi if full_passage else np.pi / 2
        beta = beta_max * x

        amp = np.sin(beta)                              # amplitude ~ sin(beta)
        dphi_dt = factor * np.cos(beta) * sweep_dir      # frequency ~ cos(beta)
        phase = np.cumsum(dphi_dt) * (2 * np.pi / self.points)  # integrate -> phase

        amp = amp / np.max(np.abs(amp))                 # normalize to max 1
        amp = amp * self.amplitude                       # scale to desired amplitude

        return amp * np.exp(1j * phase)


class Wurst(AnalyticShape):
    name = "wurst"
    """WURST adiabatic pulse (E. Kupce & R. Freeman, J. Magn. Reson. A 115,
    273-276 (1995)). Amplitude envelope 1 - |sin(pi*x)|^n (x centred on the
    pulse, -0.5 -> 0.5), swept linearly in frequency across sweep_width.
    Extra params: sweep_width (Hz, total sweep width, default 40000.0),
    power_index (amplitude power index n, default 20.0), sweep_dir (+1 =
    high to low field, -1 = low to high field, default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        power_index = self.params.get('power_index', 20.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        x = np.linspace(0.0, 1.0, self.points) - 0.5     # -0.5 -> 0.5
        amp = 1.0 - np.abs(np.sin(np.pi * x)) ** power_index
        amp = amp * self.amplitude

        offset_hz = -sweep_width * x * sweep_dir          # +SW/2 -> -SW/2
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt

        return amp * np.exp(1j * phase)


class SmoothedChirp(AnalyticShape):
    name = "smoothedchirp"
    """Smoothed chirp adiabatic pulse (J.-M. Boehlen & G. Bodenhausen,
    J. Magn. Reson. A 102, 293 (1993)). Linear frequency sweep across
    sweep_width, with the amplitude edges tapered by a raised-cosine window
    over the first/last smoothed_percent of the pulse (instead of an abrupt
    rectangular truncation).
    Extra params: sweep_width (Hz, default 40000.0), smoothed_percent (% of
    the pulse tapered at each edge, default 10.0), sweep_dir (+1/-1,
    default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        smoothed_percent = self.params.get('smoothed_percent', 10.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        n = self.points
        x = np.linspace(0.0, 1.0, n)                      # 0 -> 1
        amp = np.ones(n)

        n_taper = max(1, int(round(n * smoothed_percent / 100.0)))
        ramp = np.sin(np.pi / 2 * np.linspace(0, 1, n_taper)) ** 2  # 0 -> 1
        amp[:n_taper] = ramp
        amp[-n_taper:] = ramp[::-1]
        amp = amp * self.amplitude

        offset_hz = sweep_width * (0.5 - x) * sweep_dir    # +SW/2 -> -SW/2
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt

        return amp * np.exp(1j * phase)


class CompositeSmoothedChirp(AnalyticShape):
    name = "compositesmoothedchirp"
    """Composite smoothed-chirp pulse: several smoothed-chirp segments of
    length element_duration concatenated back-to-back, each sweeping the
    full sweep_width, with adjacent segments alternating by 180 deg phase
    (T.L. Hwang, P.C.M. van Zijl & M. Garwood, J. Magn. Reson. 124, 250
    (1997)).
    Extra params: sweep_width (Hz, default 40000.0), element_duration
    (length of one segment, same units as self.duration; default
    self.duration / 4), smoothed_percent (per-segment edge taper %, default
    10.0), sweep_dir (+1/-1, default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        element_duration = self.params.get('element_duration', self.duration / 4)
        smoothed_percent = self.params.get('smoothed_percent', 10.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        n = self.points
        n_segments = max(1, int(round(self.duration / element_duration)))
        pts_per_segment = max(1, n // n_segments)

        env = np.zeros(n, dtype=complex)
        idx = 0
        for seg in range(n_segments):
            is_last = seg == n_segments - 1
            npts = (n - idx) if is_last else pts_per_segment
            if npts <= 0:
                break

            x = np.linspace(0.0, 1.0, npts)
            amp = np.ones(npts)
            n_taper = max(1, int(round(npts * smoothed_percent / 100.0)))
            ramp = np.sin(np.pi / 2 * np.linspace(0, 1, n_taper)) ** 2
            amp[:n_taper] = ramp
            amp[-n_taper:] = ramp[::-1]

            offset_hz = sweep_width * (0.5 - x) * sweep_dir
            seg_dt = element_duration / npts
            phase = 2 * np.pi * np.cumsum(offset_hz) * seg_dt
            phase = phase + (np.pi if (seg % 2 == 1) else 0.0)  # alternate 180 deg

            env[idx:idx + npts] = amp * np.exp(1j * phase)
            idx += npts

        env = env / np.max(np.abs(env))
        env = env * self.amplitude
        return env


class TanhTan(AnalyticShape):
    name = "tanhtan"
    """Tanh/Tan adiabatic pulse (M. Garwood & Y. Ke, J. Magn. Reson. 94, 511
    (1991); R.S. Staewen et al., Invest. Radiol. 25, 559 (1990)). Amplitude
    envelope tanh(zeta*(1-|x|)) and frequency sweep (SW/2)*tan(kappa*x)/
    tan(kappa), x in [-1, 1], where kappa = atan(tan_kappa).
    Extra params: sweep_width (Hz, default 40000.0), zeta (amplitude
    steepness, default 10.0), tan_kappa (frequency-sweep steepness, i.e.
    tan(kappa), default 20.0)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        zeta = self.params.get('zeta', 10.0)
        tan_kappa = self.params.get('tan_kappa', 20.0)

        x = np.linspace(-1.0, 1.0, self.points)
        amp = np.tanh(zeta * (1.0 - np.abs(x)))
        amp = amp * self.amplitude

        kappa = np.arctan(tan_kappa)
        offset_hz = (sweep_width / 2.0) * np.tan(kappa * x) / tan_kappa
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt

        return amp * np.exp(1j * phase)


def _ca_offset_from_amplitude(amp_env: np.ndarray, sweep_width: float, sweep_dir: int = 1) -> np.ndarray:
    """
    Constant-adiabaticity offset-frequency trajectory (Hz) for a given real,
    non-negative amplitude envelope (A. Tannus & M. Garwood, J. Magn. Reson.
    A 120, 133-137 (1996)).

    Keeping the adiabaticity factor Q = omega1(t)^2 / |d(offset)/dt| constant
    across the pulse requires d(offset)/dt to be proportional to omega1(t)^2
    -- i.e. the offset sweep is not linear in time, but proportional to the
    *cumulative integral of the amplitude squared*, renormalized to span the
    full sweep_width.
    """
    power = amp_env ** 2
    cum = np.cumsum(power) - 0.5 * power           # cumulative trapezoid-like integral
    total = cum[-1]
    cum_norm = cum / total if total != 0 else cum  # normalized 0 -> 1
    return sweep_width * (0.5 - cum_norm) * sweep_dir  # +SW/2 -> -SW/2, power-weighted


class CaWurst(AnalyticShape):
    name = "cawurst"
    """Constant-adiabaticity WURST pulse (Tannus & Garwood, J. Magn. Reson.
    A 120, 133-137 (1996)). Same amplitude envelope as Wurst; the frequency
    sweep is reparametrized via _ca_offset_from_amplitude instead of being
    linear in time.
    Extra params: sweep_width (Hz, default 40000.0), power_index (default
    20.0), sweep_dir (+1/-1, default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        power_index = self.params.get('power_index', 20.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        x = np.linspace(0.0, 1.0, self.points) - 0.5
        amp_norm = 1.0 - np.abs(np.sin(np.pi * x)) ** power_index
        amp = amp_norm * self.amplitude

        offset_hz = _ca_offset_from_amplitude(amp_norm, sweep_width, sweep_dir)
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt
        return amp * np.exp(1j * phase)


class CaSmoothedChirp(AnalyticShape):
    name = "casmoothedchirp"
    """Constant-adiabaticity smoothed-chirp pulse (Tannus & Garwood, J.
    Magn. Reson. A 120, 133-137 (1996)). Same amplitude envelope as
    SmoothedChirp (raised-cosine edge taper, flat middle); the frequency
    sweep is reparametrized via _ca_offset_from_amplitude instead of being
    linear in time.
    Extra params: sweep_width (Hz, default 40000.0), smoothed_percent
    (default 10.0), sweep_dir (+1/-1, default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        smoothed_percent = self.params.get('smoothed_percent', 10.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        n = self.points
        amp_norm = np.ones(n)
        n_taper = max(1, int(round(n * smoothed_percent / 100.0)))
        ramp = np.sin(np.pi / 2 * np.linspace(0, 1, n_taper)) ** 2
        amp_norm[:n_taper] = ramp
        amp_norm[-n_taper:] = ramp[::-1]
        amp = amp_norm * self.amplitude

        offset_hz = _ca_offset_from_amplitude(amp_norm, sweep_width, sweep_dir)
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt
        return amp * np.exp(1j * phase)


class CaGauss(AnalyticShape):
    name = "cagauss"
    """Constant-adiabaticity Gaussian pulse (Tannus & Garwood, J. Magn.
    Reson. A 120, 133-137 (1996)). Gaussian amplitude envelope truncated at
    trunclevel%; frequency sweep reparametrized via _ca_offset_from_amplitude
    instead of being linear in time.
    Extra params: sweep_width (Hz, default 40000.0), trunclevel (%, default
    1.0), sweep_dir (+1/-1, default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        trunclevel = self.params.get('trunclevel', 1.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        trunc = trunclevel / 100.0
        a = -np.log(trunc)                       # exp(-a*x^2) = trunc at x = +-1
        x = np.linspace(-1.0, 1.0, self.points)
        amp_norm = np.exp(-a * x ** 2)
        amp = amp_norm * self.amplitude

        offset_hz = _ca_offset_from_amplitude(amp_norm, sweep_width, sweep_dir)
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt
        return amp * np.exp(1j * phase)


class CaLorentz(AnalyticShape):
    name = "calorentz"
    """Constant-adiabaticity Lorentzian pulse (Tannus & Garwood, J. Magn.
    Reson. A 120, 133-137 (1996)). Lorentzian amplitude envelope truncated
    at trunclevel%; frequency sweep reparametrized via
    _ca_offset_from_amplitude instead of being linear in time.
    Extra params: sweep_width (Hz, default 40000.0), trunclevel (%, default
    1.0), sweep_dir (+1/-1, default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        trunclevel = self.params.get('trunclevel', 1.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        trunc = trunclevel / 100.0
        a = 1.0 / trunc - 1.0                    # 1/(1+a*x^2) = trunc at x = +-1
        x = np.linspace(-1.0, 1.0, self.points)
        amp_norm = 1.0 / (1.0 + a * x ** 2)
        amp = amp_norm * self.amplitude

        offset_hz = _ca_offset_from_amplitude(amp_norm, sweep_width, sweep_dir)
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt
        return amp * np.exp(1j * phase)


class CaPowHsec(AnalyticShape):
    name = "capowhsec"
    """Constant-adiabaticity power-hyperbolic-secant pulse (Tannus &
    Garwood, J. Magn. Reson. A 120, 133-137 (1996)). Generalizes the
    Silver-Hoult hyperbolic secant (see HyperbolicSecant/HypSec) to a
    sech^n envelope, with the frequency sweep reparametrized via
    _ca_offset_from_amplitude for constant adiabaticity rather than
    HypSec's fixed mu-based phase modulation.
    Extra params: sweep_width (Hz, default 40000.0), beta (envelope
    steepness, default 5.3), power_index (sech exponent n, default 1.0),
    sweep_dir (+1/-1, default 1)."""
    def _build(self):
        sweep_width = self.params.get('sweep_width', 40000.0)
        beta = self.params.get('beta', 5.3)
        power_index = self.params.get('power_index', 1.0)
        sweep_dir = self.params.get('sweep_dir', 1)

        x = np.linspace(-1.0, 1.0, self.points)
        amp_norm = (1.0 / np.cosh(beta * x)) ** power_index
        amp = amp_norm * self.amplitude

        offset_hz = _ca_offset_from_amplitude(amp_norm, sweep_width, sweep_dir)
        phase = 2 * np.pi * np.cumsum(offset_hz) * self.dt
        return amp * np.exp(1j * phase)


# --------------------------------------------------------------------------
# shapes that live inline in pulse.py today
# --------------------------------------------------------------------------

class SincShape(RFShape):

    name = "sinc"
    """Hamming-windowed sinc. Was an inline lambda in shape_funcs."""

    def _build(self):
        t = self.sample_times()
        return np.hamming(self.points).T * np.sinc(t) * self.amplitude


class CosShape(RFShape):
    name = "cos"
    """Hamming-windowed cosine. Was an inline lambda in shape_funcs."""

    def _build(self):
        t = self.sample_times()
        return np.hamming(self.points).T * np.cos(t) * self.amplitude


class Sinc2PiShape(RFShape):
    name = "sinc2p"
    """sinc(2*pi*t), unwindowed. Was an inline lambda in shape_funcs."""

    def _build(self):
        t = self.sample_times()
        return np.sinc(2 * np.pi * t) * self.amplitude


class HardShape(RFShape):
    name = "hard"
    """
    Rectangular pulse — a flat envelope of ones.

    This is the class that makes ``hard_pulse`` stop being a separate function.
    Note the legacy ``hard_pulse`` builds ``np.ones((1, N))`` (2-D) and then
    indexes ``RF[0, n]``; here the envelope is 1-D like every other shape, and
    the callers lose a special case.

    Phase is 0 throughout. The legacy ``RF_angle = 180`` for a negative flip
    angle is a property of the *pulse*, not the shape, and moves to ``Pulse``.
    """

    def _build(self):
        return np.ones(self.points) * self.amplitude


# --------------------------------------------------------------------------
# imported waveforms
# --------------------------------------------------------------------------

class FileShape(RFShape):
    name = "file"
    """
    A waveform read from a Bruker/JCAMP-style file (``wave/*.jhl`` etc).

    This is the second big consolidation: an imported pulse becomes *just
    another shape*, so ``import_shaped_pulse`` and ``shaped_pulse`` stop being
    separate 50-line functions that differ only in how RF is produced.

    ``points`` is determined by the file, not by the caller — passing it is an
    error rather than a silently ignored argument.

    The phasor convention matches both legacy paths exactly:
    ``cpu_rot.Rot(theta) @ [1, 0]`` gives ``(cos t, -sin t)`` i.e. ``exp(-i t)``,
    and the torch path builds ``cos(-a) + i sin(-a)`` — the same thing.
    """

    def __init__(self, path, duration=None, amplitude=1.0, points=None, resample_to=None, **params):
        if points is not None:
            raise TypeError(
                "FileShape takes its point count from the file; do not pass points="
            )
        self.path = path
        xy = np.asarray(self._read(path), dtype=float)
        if resample_to is not None:
            xy = self._resample(xy, duration, resample_to)
        self.xy = xy
        super().__init__(duration=duration,
                         points=len(self.xy),
                         amplitude=amplitude,
                         **params)

    @staticmethod
    def _read(path):
        """Read [amplitude, phase_deg] pairs off disk. Subclasses reading a
        different file format only need to override this."""
        return import_file(path)        

    @staticmethod
    def _resample(xy, duration, target_N):
        """Resample an imported waveform onto target_N evenly-spaced points
        over the pulse duration, via cubic spline -- for callers whose time
        grid must match the simulation's own step size rather than whatever
        the source file happened to be sampled at."""
        N = len(xy)
        dt = duration / N
        t = np.linspace(-N / 2 * dt, N / 2 * dt, N)
        t_resampled = np.linspace(-duration / 2, duration / 2, target_N)
        xy_resampled = np.zeros((target_N, 2), dtype=np.float64)
        xy_resampled[:, 0] = CubicSpline(t, xy[:, 0])(t_resampled)
        xy_resampled[:, 1] = CubicSpline(t, xy[:, 1])(t_resampled)
        return xy_resampled

    def _build(self):
        magnitude = self.xy[:, 0]
        phase_deg = self.xy[:, 1]
        phasor = np.exp(-1j * np.deg2rad(phase_deg))
        return magnitude * phasor * self.amplitude

    @property
    def is_adiabatic(self) -> bool:
        """Historical rule: a phase column reaching 350 deg means a full sweep."""
        return bool(np.max(self.xy[:, 1]) >= 350)

    def __repr__(self):
        return (f"{type(self).__name__}(path={self.path!r}, points={self.points}, "
                f"duration={self.duration})")

class CompositeCSVShape(FileShape):
    name = "composite"
    """
    A composite pulse exported as a plain CSV: a single amplitude column
    ("1"), no phase column. Was sim_own_shaped_pulse's bespoke pandas
    reader. Everything else -- resampling, phasor construction, the
    "phase >= 350 deg -> adiabatic" rule -- is inherited unchanged from
    FileShape; a CSV with no phase column always has phase 0, so
    is_adiabatic naturally comes out False without needing an override.
    """

    @staticmethod
    def _read(path):
        # pandas is an optional extra (`pip install "pulsim[file]"`) and this
        # one read_csv is the only place the package uses it -- imported here
        # so the rest of PULSIM works without it.
        import pandas as pd

        df = pd.read_csv(path)
        amplitude = np.asarray(df["1"], dtype=float)
        phase = np.zeros_like(amplitude)
        return np.stack([amplitude, phase], axis=1)