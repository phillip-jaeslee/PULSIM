import os
import sys
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi']= 300

home_path = os.path.abspath(os.path.join('..'))
if home_path not in sys.path:
    sys.path.append(home_path)

tests_path = os.path.abspath(os.path.join('..', 'tests'))
if tests_path not in sys.path:
    sys.path.append(tests_path)

def ppm_to_hz(ppm, spec_freq):
    """Given a chemical shift in ppm and spectrometer frequency in MHz, return the corresponding chemical shift in Hz."""
    return [d * spec_freq for d in ppm]

from _classes import SpinSystem

from plt import mplplot, mplplot_lineshape


v_aaxx = ppm_to_hz([7.18, 7.18, 6.89, 6.89], 500)

j_aaxx = [[0, 2, 8.5, 0],
          [2, 0, 0, 8.5],
          [8.5, 0, 0, 2],
          [0, 8.5, 2, 0]]

aaxx = SpinSystem(v_aaxx, j_aaxx)

mplplot(aaxx.peaklist())