import os
import sys
import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.cm as cm
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

from plt import mplplot_one, mplplot_lineshape, mplplot

angles = ["anti", "gn60", "gp60"]

for angle in angles:
    if angle =="anti":
        v_aaxx = ppm_to_hz([1.248, 0.952, 0.953, 1.376, 1.376, 1.377, 1.377, 1.248, 0.853, 0.953], 600)
        """
        j_aaxx = [[0, -12.5552, -12.5546, 3.2126, 3.18655, -0.51039, -0.509805, 0.0152104, -0.197609, -0.197542],
                  [-12.5552, 0, -12.0461, 3.65676, 12.9252, -0.333076, -0.0704038, -0.197538, -0.0422818, -0.0240245],
                  [-12.5546, -12.0461, 0, 12.9255, 3.68505, -0.0708124, -0.333399, -0.197615, -0.0232941, -0.0420837],
                  [3.2126, 3.65676, 12.9255, 0, -12.6021, 11.2968, 3.73442, -0.509969, -0.0706148, -0.333173],
                  [3.18655, 12.9252, 3.68505, -12.6021, 0, 3.73893, 11.2918, -0.510397, -0.333299, -0.0706332],
                  [-0.51039, -0.333076, -0.0708124, 11.2968, 3.73893, 0, -12.5978, 3.20137, 12.9242, 3.66874],
                  [-0.509805, -0.0704038, -0.333399, 3.73442, 11.2918, -12.5978, 0, 3.19786, 3.67255, 12.9258],
                  [0.0152104, -0.197538, -0.197615, -0.509969, -0.510397, 3.20137, 3.19786, 0, -12.5548, -12.5557],
                  [-0.197609, -0.0422818, -0.0232941, -0.0706148, -0.333299, 12.9242, 3.67255, -12.5548, 0, -12.0466],
                  [-0.197542, -0.0240245, -0.0420837, -0.333173, -0.0706332, 3.66874, 12.9258, -12.5557, -12.0466, 0]]
        """
        j_aaxx = [[0, -12.5552, -12.5546, 3.2126, 3.18655, 0, -0.0, 0.0, -0.0, -0.0],
                  [-12.5552, 0, -12.0461, 3.65676, 12.9252, -0.0, -0.0, -0.0, -0.0, -0.0],
                  [-12.5546, -12.0461, 0, 12.9255, 3.68505, -0.0, -0.0, -0.0, -0.0, -0.0],
                  [3.2126, 3.65676, 12.9255, 0, -12.6021, 11.2968, 3.73442, -0.0, -0.0, -0.0],
                  [3.18655, 12.9252, 3.68505, -12.6021, 0, 3.73893, 11.2918, -0.0, -0.0, -0.0],
                  [-0.0, -0.0, -0.0, 11.2968, 3.73893, 0, -12.5978, 3.20137, 12.9242, 3.66874],
                  [-0.0, -0.0, -0.0, 3.73442, 11.2918, -12.5978, 0, 3.19786, 3.67255, 12.9258],
                  [0.0, -0.0, -0.0, -0.0, -0.0, 3.20137, 3.19786, 0, -12.5548, -12.5557],
                  [-0.0, -0.0, -0.0, -0.0, -0.0, 12.9242, 3.67255, -12.5548, 0, -12.0466],
                  [-0.0, -0.0, -0.0, -0.0, -0.0, 3.66874, 12.9258, -12.5557, -12.0466, 0]]
    elif angle == "gn60":
        v_aaxx
    elif angle == "gp60":
        v_aaxx
    aaxx = SpinSystem(v_aaxx, j_aaxx)
    mplplot(aaxx.peaklist())



"""
FWHM = 10
fig, ax = mpl.subplots(figsize=(10, 5))

# Create a colormap
cmap = cm.get_cmap('rainbow')
ps = np.linspace(0.1, 1.1, 50)  # match your range: i from 30 to 79

protein = "DYE"

for i, p in enumerate(ps):

    if protein == "LYV":
        v_aaxx_1 = ppm_to_hz([3.104, 2.696], 600)
        v_aaxx_2 = ppm_to_hz([2.595, 3.050], 600)
        j_aaxx_1 = [[0, -12.45], [-12.45, 0]]
        j_aaxx_2 = [[0, -13.02], [-13.02, 0]]
    elif protein == "AYE":
        v_aaxx_1 = ppm_to_hz([3.138, 2.696], 600)
        v_aaxx_2 = ppm_to_hz([2.608, 3.040], 600)
        j_aaxx_1 = [[0, -12.40], [-12.40, 0]]
        j_aaxx_2 = [[0, -13.12], [-13.12, 0]]   
    elif protein == "DYE":
        v_aaxx_1 = ppm_to_hz([2.909, 3.111], 600)
        v_aaxx_2 = ppm_to_hz([2.767, 2.990], 600)
        j_aaxx_1 = [[0, -12.06], [-12.06, 0]]
        j_aaxx_2 = [[0, -13.23], [-13.23, 0]]


    v_aaxx = [v1*p + v2*(1-p) for v1, v2 in zip(v_aaxx_1, v_aaxx_2)]
    j_aaxx = [[j1_ij * p + j2_ij * (1 - p) for j1_ij, j2_ij in zip(row1, row2)] 
              for row1, row2 in zip(j_aaxx_1, j_aaxx_2)]

    aaxx = SpinSystem(v_aaxx, j_aaxx)
    color = cmap(i / len(ps))  # normalized color from 0 to 1

    mplplot_one(aaxx.peaklist(), ax=ax, w=FWHM, hertz=600, color=color)

# Optionally add a colorbar to indicate p-values
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.Normalize(vmin=0.1, vmax=1.1))
cbar = mpl.colorbar(sm, ax=ax)
cbar.set_label("Relative population on 180˚ confomer")

mpl.title(f"Overlay of {protein} spectra depending on the population")
mpl.xlabel("ppm")
mpl.ylabel("Intensity")
mpl.tight_layout()
mpl.savefig(f"{protein}_spectra_depending_population_FWHM{FWHM}_huge.svg")
#mpl.show()
"""