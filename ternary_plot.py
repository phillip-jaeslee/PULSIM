
import matplotlib
matplotlib.use("TkAgg")  # Ensure correct backend (optional, especially for macOS)

import matplotlib.pyplot as plt
import ternary

import math
import random
"""
def shannon_entropy(p):
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1. * s

scale = 60
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 8)
tax.heatmapf(shannon_entropy, boundary=True, style="triangular")
tax.boundary(linewidth=2.0)
tax.set_title("Shannon Entropy Heatmap")
tax.ticks(axis='lbr', linewidth=1, multiple=5)
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.show()


# Test Permutations in Heatmap

# Colors to plot
aux = dict({(i, j, k): k for i, j, k in ternary.helpers.simplex_iterator(scale)})

def test_permutation(permutation, scale=10, style='h'):
    aux = dict({(i,j,k): k for i, j, k in  ternary.helpers.simplex_iterator(scale)})
    ax = ternary.heatmap(aux, scale, style=style, permutation=permutation)
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    tax.boundary(linewidth=1.5)
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

for permutation in ['012', '120', '210']:
    for style in ['t', 'h']:
        test_permutation(permutation, style=style)
"""

def generate_random_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i,j,k) in simplex_iterator(scale):
        d[(i,j)] = random.random()
    return d

scale = 20
d = generate_random_heatmap_data(scale)
print(d)
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 8)
tax.heatmap(d, style="h")
tax.boundary()
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.set_title("Heatmap Test: Hexagonal")
tax.show()