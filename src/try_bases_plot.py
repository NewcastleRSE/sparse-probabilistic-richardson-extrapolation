import re
import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX fonts
import matplotlib as mpl

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
})



if __name__ == "__main__":

    show_plot = True

    matrix, row_labels, h_values = plot_basis_file("data/mujoco/results/output_two_spheres_bases_300.dat", show_plot)