import matplotlib.pyplot as plt
import ternary
import numpy as np

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

# ----------------------------------
# Set parameters
# ----------------------------------
scale = 100
field = 900
FWHM = 10
heatmap_data = {}

#protein = "LYV"  # for naming
#proteins = ["LYV"]
proteins = ["LYV", "AYE", "DYE"]

for field in range(100, 1000, 100):
    for protein in proteins:
        if protein == "LYV":
            v1 = ppm_to_hz([3.104, 2.696], field) # 180 conformer
            v2 = ppm_to_hz([2.595, 3.050], field) # -60 conformer
            v3 = ppm_to_hz([2.979, 3.269], field)  # 60 conformer
            # Define J-coupling matrices
            j1 = [[0, -12.45], [-12.45, 0]]     # 180 conformer
            j2 = [[0, -13.02], [-13.02, 0]]     # -60 conformer
            j3 = [[0, -13.65], [-13.65, 0]]     # 60 conformer

        elif protein == "AYE":
            v1 = ppm_to_hz([3.138, 2.696], field)
            v2 = ppm_to_hz([2.608, 3.040], field)
            v3 = ppm_to_hz([2.937, 3.326], field)

            # Define J-coupling matrices
            j1 = [[0, -12.40], [-12.40, 0]]
            j2 = [[0, -13.12], [-13.12, 0]]   
            j3 = [[0, -13.61], [-13.61, 0]]

        elif protein == "DYE":
            v1 = ppm_to_hz([2.909, 3.111], field)
            v2 = ppm_to_hz([2.767, 2.990], field)
            v3 = ppm_to_hz([2.925, 3.074], field)

            # Define J-coupling matrices
            j1 = [[0, -12.06], [-12.06, 0]]
            j2 = [[0, -13.23], [-13.23, 0]]
            j3 = [[0, -13.42], [-13.42, 0]]


        # ----------------------------------
        # Loop over ternary combinations
        # ----------------------------------
        for a in range(0, scale + 1, 1):
            for b in range(0, scale + 1 - a, 1):
                c = scale - a - b

                # Normalize populations
                pA, pB, pC = a / scale, b / scale, c / scale

                # Weighted average of v
                v = [pA * v1[i] + pB * v2[i] + pC * v3[i] for i in range(len(v1))]

                # Weighted average of J matrix
                J = [
                    [
                        pA * j1[i][j] + pB * j2[i][j] + pC * j3[i][j]
                        for j in range(len(j1[0]))
                    ]
                    for i in range(len(j1))
                ]

                # Simulate spectrum
                system = SpinSystem(v, J)
                peaks = system.peaklist()
                print(peaks)
                if len(peaks) < 4:
                    peak_diff_hz = 0
                else:
                    # Extract first two peak positions
                    peak_positions_ppm = sorted([p[0] for p in peaks])
                    peak_diff_hz = abs(peak_positions_ppm[1] - peak_positions_ppm[2])

                # Store value in heatmap
                heatmap_data[(a, b, c)] = peak_diff_hz

        import json

        # Convert tuple keys to strings for JSON compatibility
        save_dict = {
            f"{a},{b},{c}": float(value.detach().cpu().item() if hasattr(value, "item") else value)
            for (a, b, c), value in heatmap_data.items()
        }

        with open(f"heatmap_data_{protein}_{field}.json", "w") as f:
            json.dump(save_dict, f)


        heatmap_numpy = {
            key: float(value.detach().cpu().item()) if hasattr(value, "item") else float(value)
            for key, value in heatmap_data.items()
        }

        print(heatmap_data)
        # ----------------------------------
        # Plot ternary heatmap
        # ----------------------------------
        fig, tax = ternary.figure(scale=scale)
        fig.set_size_inches(10, 8)

        tax.heatmap(heatmap_numpy, style="hexagonal", cmap='rainbow', vmin=0)

        tax.set_title(f"Splitting (Hz) depending on conformer — {protein} in field {field}MHz", fontsize=12)
        tax.boundary()
        tax.gridlines(multiple=10)
        tax.ticks(multiple=10)
        tax.bottom_axis_label("180˚ Conformer")
        tax.left_axis_label("-60˚ Conformer")
        tax.right_axis_label("60˚ Conformer")
        tax.get_axes().axis('off')
        tax.clear_matplotlib_ticks()
        tax.savefig(f"ternary_plot_of_{protein}_field_{field}.svg")
        tax.close()
