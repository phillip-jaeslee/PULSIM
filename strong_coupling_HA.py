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
"""
hertzs = [200, 300, 400, 500, 600, 700, 800, 900]

diffs = [0, 10, 20, 30, 40, 50, 60, 70]

j_couplings = [-12, -13, -14, -15, -16, -17]

for j_coupling in j_couplings:
    for hertz in hertzs:
        for diff in diffs:
            ppm1 = ((2.95 * hertz) - (diff / 2)) / hertz
            ppm2 = ((2.95 * hertz) + (diff / 2)) / hertz
            v_aaxx = ppm_to_hz([ppm1, ppm2], hertz)

            j_aaxx = [[0, j_coupling],
                    [j_coupling, 0]]

            aaxx = SpinSystem(v_aaxx, j_aaxx)

            mplplot(aaxx.peaklist(), hertz=hertz, diff=diff, j_coupling=j_coupling)

"""

#v_aaxx = ppm_to_hz([2.483, 2.912], 600) # 261
#v_aaxx = ppm_to_hz([2.971, 2.580], 600) # 234
#v_aaxx = ppm_to_hz([2.850, 3.122], 600) # 163
#v_aaxx = ppm_to_hz([2.987, 2.564], 600) # 253
#v_aaxx = ppm_to_hz([2.791, 3.301], 600) # 305
#v_aaxx = ppm_to_hz([2.881, 2.959], 600) # 218

#j_aaxx = [[0, -13.02],
#        [-13.02, 0]]

#j_aaxx = [[0, -13.14],
#        [-13.14, 0]]

#j_aaxx = [[0, -13.65],
#        [-13.65, 0]]

#j_aaxx = [[0, -12.41], [-12.41, 0]]
#j_aaxx = [[0, -13.50], [-13.50, 0]]
#j_aaxx = [[0, -12.94], [-12.94, 0]]


#v_aaxx_1 = ppm_to_hz([2.909, 3.111], 600)
#v_aaxx_2 = ppm_to_hz([2.767, 2.990], 600)
#v_aaxx_3 = ppm_to_hz([2.925, 3.074], 600) 
#j_aaxx_1 = [[0, -12.06], [-12.06, 0]]
#j_aaxx_2 = [[0, -13.23], [-13.23, 0]]
#j_aaxx_3 = [[0, -13.42], [-13.42, 0]] 

#FWHM = 1
#angle = "p60"

#aaxx = SpinSystem(v_aaxx_3, j_aaxx_3)

#mplplot(aaxx.peaklist(), w=FWHM, hertz=600, angle=angle)

#angle = "weight_avg"
"""
FWHM = 1

for i in range(30, 80):
    p = i / 100
    v_aaxx_1 = ppm_to_hz([3.138, 2.696], 600)
    v_aaxx_2 = ppm_to_hz([2.608, 3.040], 600)
    j_aaxx_1 = [[0, -12.40], [-12.40, 0]]
    j_aaxx_2 = [[0, -13.12], [-13.12, 0]]

    v_aaxx = [v1*p + v2*(1-p) for v1, v2 in zip(v_aaxx_1, v_aaxx_2)]
    j_aaxx = [[j1_ij * p + j2_ij * (1 - p) for j1_ij, j2_ij in zip(row1, row2)] 
              for row1, row2 in zip(j_aaxx_1, j_aaxx_2)]

    print(v_aaxx)
    print(j_aaxx)

    aaxx = SpinSystem(v_aaxx, j_aaxx)

    mplplot(aaxx.peaklist(), w=FWHM, hertz=600, angle=str(p))
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
