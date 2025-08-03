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

from plt import mplplot_one, mplplot_lineshape, mplplot, mplplot_three_angles
#peptides = ["QLG"]
#peptides = ["AYE", "DYE", "GFV", "LYV", "QLG"]
#peptides = ["AYE", "GFV"]
peptides = ["AYE", "GFV", "QLG"]
angle_labels = ["180", "p60", "n60"]
temperatures = ["273", "283", "293", "303", "313"]
peaklist_dict = {}

for peptide in peptides:
    if peptide == "AYE":
        CS_H2s = [3.343, 2.538, 3.220]
        CS_H3s = [2.631, 3.099, 3.596]
        CS_HAs = [4.192, 4.440, 4.505]

        p180s = [0.10994540,	0.116728473,    0.123320043,    0.129710566,    0.13589401]
        pn60s = [0.036952353,	0.040772616,	0.044648909,	0.04856345, 	0.052500283]
        pp60s = [0.853102238,	0.842498911,	0.832031049,	0.821725983,	0.811605707]

        j_couplings = [-12.2407, -13.1184, -13.3902]
        j_couplings_2As = [4.18853, 10.7296, 2.12404]
        j_couplings_3As = [10.0806, 2.98416, 6.21764]
    elif peptide == "DYE":
        CS_H2s = [2.534, 2.855, 3.183]
        CS_H3s = [3.887, 2.981, 3.259]
        CS_HAs = [4.092, 3.760, 4.372]

        j_couplings = [-12.5666, -13.3048, -14.0276]
        j_couplings_2As = [1.02873, 10.4335, 6.96964]
        j_couplings_3As = [9.94953, 1.76111, 2.11193]        
    elif peptide == "GFV":
        CS_H2s = [3.423, 2.575, 3.124]
        CS_H3s = [2.711, 3.039, 3.654]
        CS_HAs = [4.458, 4.739, 4.748]
        
        p180s = [0.709396464,	0.698243805,	0.687603776,	0.677454117,	0.667771982]
        pn60s = [0.09037399,	0.095668804,	0.100821571,	0.105828749,	0.11068872]
        pp60s = [0.200229546,	0.206087392,	0.211574653,	0.216717134,	0.221539298]

        j_couplings = [-12.0485, -12.7775, -13.6561]
        j_couplings_2As = [4.19896, 10.8563, 3.53624]
        j_couplings_3As = [10.1587, 2.86465, 4.52995]    
    elif peptide == "LYV":
        CS_H2s = [3.104, 2.595, 2.979]
        CS_H3s = [2.696, 3.050, 3.263]
        CS_HAs = [4.264, 4.436, 4.532]

        j_couplings = [-12.447, -13.0247, -13.645]
        j_couplings_2As = [4.31843, 10.6257, 3.44756]
        j_couplings_3As = [10.2741, 2.74082, 4.50160]    
    elif peptide == "QLG":
        CS_H2s = [1.431, 1.391, 1.752]
        CS_H3s = [1.935, 1.880, 1.957]
        CS_HAs = [4.258, 4.209, 5.166]

        p180s = [0.814303785,	0.804429392,	0.794791280,	0.785394462,	0.776241589]
        pn60s = [0.167043657,	0.174514812,	0.181651962,	0.188461828,	0.194953089]
        pp60s = [0.018652559,	0.021055795,	0.023556759,	0.026143710,	0.028805322]

        j_couplings = [-12.0442, -14.5763, -14.0873]    
        j_couplings_2As = [3.44267, 9.50235, 2.61376]
        j_couplings_3As = [11.1715, 0.980241, 6.27756]    
    else:
        print(f"There is no information about peptide {peptide}")
        continue
    
    for i, temperature in enumerate(temperatures):

        avg_CS_H2 = CS_H2s[0] * p180s[i] + CS_H2s[1] * pp60s[i] + CS_H2s[2] * pn60s[i]
        avg_CS_H3 = CS_H3s[0] * p180s[i] + CS_H3s[1] * pp60s[i] + CS_H3s[2] * pn60s[i]
        avg_j_coupling = j_couplings[0] * p180s[i] + j_couplings[1] * pp60s[i] + j_couplings[2] * pn60s[i]

        v_aaxx = ppm_to_hz([avg_CS_H2, avg_CS_H3], 600)
        j_aaxx = [[0, avg_j_coupling], [avg_j_coupling, 0]]

        aaxx = SpinSystem(v_aaxx, j_aaxx)
        peaklist_dict[temperature] = aaxx.peaklist()
    mplplot_three_angles(peaklist_dict, peptide=peptide, w=5)
    """
    for angle, CS_H2, CS_H3, j_coupling, in zip(angle_labels, CS_H2s, CS_H3s, j_couplings):
        v_aaxx = ppm_to_hz([CS_H2, CS_H3], 600)
        j_aaxx = [[0, j_coupling, ],
                  [j_coupling, 0]]
        aaxx = SpinSystem(v_aaxx, j_aaxx)

        # Save peaklist under angle label
        peaklist_dict[angle] = aaxx.peaklist()
        mplplot_three_angles(peaklist_dict, peptide=peptide, w=5)
    """
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