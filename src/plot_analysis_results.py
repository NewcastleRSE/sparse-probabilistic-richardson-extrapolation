##############################################################################
# Script to plot SPRE analyses results.
#
# From root directory, for example run
# python ./src/plot_analysis_results.py
#
# Richard Howey, July 2025 - April 2026
##############################################################################

# Python modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import itertools
import collections.abc
from pathlib import Path
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullFormatter

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

def choose_h_column(df):
    """
    Mimics your internal logic:
    - Prefer 'h' if present
    - Otherwise use 'h1'
    """
    if "h" in df.columns:
        return "h"
    elif "h1" in df.columns:
        return "h1"
    else:
        raise ValueError("No valid h column found (expected 'h' or 'h1').")


def plot_spre_results(  
    files: list[str],
    h_columns: list[str] = ["h"],
    labels: list[str] = ["SPRE"],
    true_value_filename: str = "",
    true_value: float = None,
    title: str = "Extrapolation Results",
    y_logscale: bool = False,
    x_lims: tuple = None,
    y_lims: tuple = None,
    pos_inset = "lower left",
    zoom_first_n: int = 7,
    *args,
    **kwargs
) -> None:

    # Use current active axes
    ax = plt.gca()

    # Optional true value from file
    if true_value_filename:
        df_true = pd.read_csv(true_value_filename, sep="\t")
        true_value = df_true["true_value"][0]

    inset_data = []

    for file, h_col, label in zip(files, h_columns, labels):

        df = pd.read_csv(file, sep="\t")

        if h_col not in df.columns:
            raise ValueError(
                f"Column '{h_col}' not found in {file}. "
                f"Available columns: {list(df.columns)}"
            )

        if x_lims is not None:
            mask = (df[h_col] >= x_lims[0]) & (df[h_col] <= x_lims[1])
            df = df[mask]

        x_vals = df[h_col].values
        y_vals = df["mu"].values

        # Main plot
        if "var" in df.columns:
            y_err = 2 * np.sqrt(df["var"].values)
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                fmt="o-",
                capsize=5,
                ecolor="black",
                markersize=6,
                label=label,
                *args,
                **kwargs
            )
        else:
            ax.plot(
                x_vals,
                y_vals,
                "o-",
                markersize=6,
                label=label,
                *args,
                **kwargs
            )

        # Store first N smallest h values for inset
        sort_idx = np.argsort(x_vals)

        y_errs = (np.nan,)*zoom_first_n
        if "var" in df.columns:
            y_errs = (2 * np.sqrt(df["var"].values))[sort_idx][:zoom_first_n]
      
        inset_data.append((
            x_vals[sort_idx][:zoom_first_n],
            y_vals[sort_idx][:zoom_first_n],   
            y_errs,         
        ))
 
    # Formatting main axes
    ax.set_xscale("log")
    if y_logscale:
        ax.set_yscale("log")

    ax.set_xlabel(r"$h$")
    ax.set_ylabel("estimate")
    ax.set_title(title, pad=20)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    if y_lims is not None:
        ax.set_ylim(y_lims)

    if true_value is not None:
        ax.axhline(y=true_value, color="red", linestyle="--", linewidth=1)

    if pos_inset is not None:
        # Inset on the SAME axes
        axins = inset_axes(
            ax,
            width="45%",       # slightly smaller
            height="50%",
            loc=pos_inset,
            borderpad=3      # leave margin from parent axes edge
        )

        for x_zoom, y_zoom, y_err in inset_data:

            if np.all(np.isfinite(y_err)):
                axins.errorbar(
                    x_zoom,
                    y_zoom,
                    yerr=y_err,
                    fmt="o-",
                    capsize=4,
                    ecolor="black",
                    markersize=5,
                    *args,
                    **kwargs
                )
            else:
                axins.plot(
                    x_zoom,
                    y_zoom,
                    "o-",
                    markersize=5,
                    *args,
                    **kwargs
                )

        axins.set_xscale("log")
        if y_logscale:
            axins.set_yscale("log")

        if true_value is not None:
            axins.axhline(y=true_value, color="red", linestyle="--", linewidth=1)

        # Add internal padding so ticks/points aren't on the frame
        axins.margins(x=0.08, y=0.10)

        axins.grid(True, linestyle="--", alpha=0.3)

        # Smaller tick labels for clarity
        axins.tick_params(labelsize=8)


def plot_multiple_spre_abs(
    files,
    h_columns,
    labels,
    output_file,
    title = "Absolute Errors of SPRE Estimates",
    show_plot = True, 
    plot_raw_estimates = False,  
    marker=None,
    x_lims: tuple = None,
    y_lims: tuple = None,
    *args,           # Positional arguments for plt.plot (like marker, linestyle)
    **kwargs         # Keyword arguments for plt.plot (like color, alpha, linewidth)
):
    """
    Plot multiple SPRE absolute-error curves on one log-log plot.

    Parameters:
        files : list[str or Path]   Paths to result files   
        h_columns : list[str]       Name of the h column for each file (e.g. ["h", "h1", "h2"]) 
        labels : list[str]          Optional legend labels for each file 
        output_file : str           Output image filename
        title : str                 Plot title
        show_plot : bool            Show the plot as well as writing file 
        plot_raw_estimates : bool   Also plot the raw estimates on the plot   
        *args : tuple               Positional args passed to plt.plot (optional)
        **kwargs : dict             Keyword args passed to plt.plot (optional)

    Returns:
        None

    """

    if len(files) != len(h_columns):        
        raise ValueError("files and h_columns must have the same length")

    if labels is None:
        labels = [Path(f).stem for f in files]

    if len(files) != len(labels):
        raise ValueError("files and labels must have the same length")

    # Check if marker is an iterator (but not a string)
    if isinstance(marker, collections.abc.Iterator):
        marker_iter = marker
    else:
        marker_iter = None

    plt.close("all")
    plt.figure(figsize=(7, 6))

    for file, h_col, label in zip(files, h_columns, labels):
        df = pd.read_csv(file, sep="\t")

        if h_col not in df.columns:
            raise ValueError(
                f"Column '{h_col}' not found in {file}. "
                f"Available columns: {list(df.columns)}"
            )

        if marker_iter is not None:
            m = next(marker_iter)
        else:
            m = marker  # fixed value

        if x_lims is not None:
            mask = (df[h_col] >= x_lims[0]) & (df[h_col] <= x_lims[1])
            df = df[mask]

        x_vals = df[h_col]
        y_vals = df["abs_err_spre_estimate"]

        plt.plot(
            x_vals,
            y_vals, 
            marker = m,          
            *args,
            label=label,
            **kwargs
        )

    # Raw estimates (faded)
    if plot_raw_estimates:
        df = pd.read_csv(files[0], sep="\t")
      
        if x_lims is not None:
            mask = (df[h_col] >= x_lims[0]) & (df[h_col] <= x_lims[1])
            df = df[mask]

        i = 0
        while f"abs_err_estimate_{i+1}" in list(df):
            plt.plot(
                df[h_columns[0]],
                df[f"abs_err_estimate_{i+1}"],
                marker="o",
                linewidth=2,
                markersize=4,
                alpha=0.4,
                zorder = 0,
                label="raw estimates" if i == 0 else None
            )
            i += 1

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$h$")
    plt.ylabel("absolute error")
    plt.title(title)

   
    # Get current active axis
    ax = plt.gca()

    if y_lims is not None:
        ax.set_ylim(y_lims)

    plt.grid(True)
    plt.minorticks_off()
    plt.grid(True, which="major", linestyle="--", alpha=0.4)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    if show_plot:
        plt.show()


def plot_three_images_together(files, output_file, show_plot = True, crop = (0.12, 0, 0.85, 1)):
    """
    Combines 3 image files in one image file.

    Parameters:
        files : list[str or Path]   Filename and paths to files   
        output_filename : str       Filename and path of final file.
        show_plot : bool            Show the plot as well as writing file
        crop : tuple                Crop percentiles for left, upper, right, lower
    Returns:
        None

    """

    cropped_images = []

    for file in files:
        img = Image.open(file)
        w, h = img.size

        # Crop middle of width
        left = int(crop[0] * w)
        upper = int(crop[1] * h)
        right = int(crop[2] * w)
        lower = int(crop[3] * h)

        cropped = img.crop((left, upper, right, lower))
        cropped_images.append(cropped)

    # Height of all images is the same
    height = img.size[1]

    # Compute total width
    total_width = sum(img.size[0] for img in cropped_images)

    # Create blank canvas
    combined = Image.new("RGB", (total_width, height), "white")

    # Paste images side by side
    x_offset = 0
    for img in cropped_images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # Save result
    combined.save(output_file, dpi=(300, 300))

    print(f"Three plots saved to {output_file}")
  
   
def plot_basis_file(filename, output_file, title = "", show_bar = True, show_plot = False, *args, **kwargs):  
    """
    Reads the file and plots a scatter plot of basis elements

    Parameters:
        filename : str              Filename of basis data
        output_file : str           Output image filename
        title : str                 Plot title
        show_plot : bool            Show the plot as well as writing file 
        *args : tuple               Positional args passed to plt.plot (optional)
        **kwargs : dict             Keyword args passed to plt.plot (optional)

    Returns:
        None

    """

    # -----------------------
    # Read file
    # -----------------------
    with open(filename, 'r') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())

    h_values = []
    basis_dict = {}
    col_index = 0

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        h_match = re.search(r'h\s*=\s*([0-9.eE+-]+)', lines[0])
        if not h_match:
            continue

        h_val = float(h_match.group(1))
        h_values.append(h_val)

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("Basis"):
                continue

            nums = tuple(map(int, line.split()))

            if nums not in basis_dict:
                basis_dict[nums] = set()

            basis_dict[nums].add(col_index)

        col_index += 1

    # -----------------------
    # Correct ordering
    # total degree, then descending lex
    # -----------------------
    sorted_basis = sorted(
        basis_dict.keys(),
        key=lambda x: (x[0] + x[1] + x[2], -x[0], -x[1], -x[2])
    )

    row_labels = [f"({a}, {b}, {c})" for (a, b, c) in sorted_basis]

    num_rows = len(sorted_basis)
    num_cols = len(h_values)

    # -----------------------
    # Build matrix
    # -----------------------
    matrix = np.zeros((num_rows, num_cols), dtype=int)

    for i, vec in enumerate(sorted_basis):
        for j in basis_dict[vec]:
            matrix[i, j] = 1

    # -----------------------
    # Fast vectorised scatter
    # -----------------------
    y_indices, x_indices = np.where(matrix == 1)

    # -----------------------
    # Format h labels
    # -----------------------
    def latex_sci(x):
        if x == 0:
            return "$0$"
        exponent = int(np.floor(np.log10(abs(x))))
        mantissa = x / 10**exponent
        return rf"${mantissa:.3g} \times 10^{{{exponent}}}$"

    formatted_h = [latex_sci(h) for h in h_values]

    # -----------------------
    # Plot
    # -----------------------
    if show_bar:
        fig = plt.figure(figsize=(14, 9))
        gs = GridSpec(2, 1, height_ratios=[6, 1], hspace=0.02)

        # Top: scatter
        ax = fig.add_subplot(gs[0])

        # Bottom: bar (shares x)
        ax_bar = fig.add_subplot(gs[1], sharex=ax)

        # Hide x tick labels on scatter
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        # Add space at bottom
        fig.subplots_adjust(bottom=0.25)

    else:
        fig, ax = plt.subplots(figsize=(14, 8))

    # Create list of colours
    cmap = plt.get_cmap("tab20")
    colors = cmap(y_indices)

    # Faint horizontal guide lines
    for y in range(num_rows):
        ax.axhline(y=y, color='gray', linestyle="--", alpha=0.4, linewidth=0.8)

    # Single fast scatter call
    ax.scatter(x_indices, y_indices, s=60, facecolor=colors, edgecolor='dimgray', zorder=2, *args, **kwargs)

    ax.set_xticks(range(num_cols))
    ax.set_xticklabels(formatted_h, rotation=90)
    ax.set_yticks(range(num_rows), row_labels)

    ax.set_xlabel(r"$h$", fontsize=18)
    ax.set_ylabel(r"Elements of Index Set $A$", fontsize=18)
    ax.set_title(title, pad=8)

    ax.invert_yaxis()

    # -----------------------
    # Optional bar plot
    # -----------------------
    if show_bar:
        # Count basis size per h
        basis_counts = matrix.sum(axis=0)
        ax_bar.bar(range(num_cols), basis_counts, width=0.6)
        ax_bar.set_ylabel("Count", fontsize=18)
        #ax_bar.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax_bar.set_xticks(range(num_cols))
        ax_bar.set_xticklabels(formatted_h, rotation=90)
        ax_bar.set_xlim(-0.5, num_cols - 0.5)
        ax_bar.set_xlabel(r"$h$", fontsize=18)

    plt.tight_layout()

    plt.savefig(output_file)
    
    if show_plot:
        plt.show()


# Code below to plot the figures for the paper 
# plus a few more that perhaps didn't make it to the paper or the supplementary
if __name__ == "__main__":

    # Show any plots (except combined flock sim pngs)
    show_plot = False

    # Combine flock plots at different time points  
    files = [
            "data/flock/results/flock1_start.png",
            "data/flock/results/flock1_1_second_with_trail.png",
            "data/flock/results/flock1_5_seconds_no_trail.png"
        ]
    
    output_file = "data/plots/flock_sim_3_timepoints.png"

    plot_three_images_together(files, output_file, show_plot)

    # Combine flock plots at different time points with all with trails  
    files = [
            "data/flock/results/flock1_start.png",
            "data/flock/results/flock1_1_second_with_trail.png",
            "data/flock/results/flock1_5_seconds_with_trail.png"
        ]
    
    output_file = "data/plots/flock_sim_3_timepoints_all_trails.png"

    plot_three_images_together(files, output_file, show_plot)

    # Plot two spheres plot
    crops = (0, 0, 1, 1) # Left, Top, Right, Bottom
    files = [
            "data/mujoco/results/two_spheres_time0.png",
            "data/mujoco/results/two_spheres_time0.3.png",
            "data/mujoco/results/two_spheres_time5.png"
        ]
    
    output_file = "data/plots/two_spheres_sim_3_timepoints.png"

    plot_three_images_together(files, output_file, show_plot, crops)

    # Plot five shapes plot
    files = [
            "data/mujoco/results/many_shapes_time0.png",
            "data/mujoco/results/many_shapes_time0.3.png",
            "data/mujoco/results/many_shapes_time5.png"
        ]
    
    output_file = "data/plots/five_shapes_sim_3_timepoints.png"

    plot_three_images_together(files, output_file, show_plot, crops)

    ###################################
    # Plot error bar plot 
    for seed in [1]:
        plt.close("all")
  
        # Create a subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        seed_str = str(seed)
        if seed == 1:
            seed_str = ""

        labels = ["SPRE White", "SPRE Gaussian",  r"SPRE Mat\'{e}rn-$\frac{1}{2}$", r"SPRE Mat\'{e}rn-$\frac{3}{2}$", "MRE"]
        methods = ["", "_Gaussian", "_Matern12", "_Matern32", "_MRE"]

        for plot_num, method, label in zip(range(1, 5), methods, labels):
            files = [f"data/flock/results/output{seed_str}_flock_321{method}.dat"]
            true_val_file = f"data/flock/results/output{seed_str}_flock_abs_errors_eval_321{method}.dat"
            h_columns = ["h"]           

            plt.subplot(2, 2, plot_num)

            plot_spre_results(files=files,
                            true_value_filename=true_val_file,
                            h_columns=h_columns,
                            #labels=[label],
                            #y_logscale=True,
                            title=label,                            
                            x_lims=(1e-16, 1e-8))

        plt.tight_layout()
        output_file=f"data/plots/spre_3_parameters_error_bars_seed{seed}.png"
        plt.savefig(output_file)

        if show_plot:
            plt.show()

    ###################################
    # Compare different methods for flock scenario 321
    for seed in [1]:
        seed_str = str(seed)
        if seed == 1:
            seed_str = ""

        files = [
            f"data/flock/results/output{seed_str}_flock_abs_errors_eval_321.dat",
            f"data/flock/results/output{seed_str}_flock_abs_errors_eval_321_Gaussian.dat",            
            f"data/flock/results/output{seed_str}_flock_abs_errors_eval_321_Matern12.dat",
            f"data/flock/results/output{seed_str}_flock_abs_errors_eval_321_Matern32.dat",
            f"data/flock/results/output{seed_str}_flock_abs_errors_eval_321_MRE.dat",
            f"data/flock/results/output{seed_str}_flock_abs_errors_eval_321_GRE.dat",
        ]

        h_columns = ["h"] * len(files)

        labels = ["SPRE White", "SPRE Gaussian", r"SPRE Mat\'{e}rn-$\frac{1}{2}$", r"SPRE Mat\'{e}rn-$\frac{3}{2}$", "MRE", "GRE White"]
        title = None #"Absolute Errors of Estimates"
        markers = itertools.cycle(('o', 's', 'v', '^', '+', 'x', '*'))
        plot_multiple_spre_abs(files, h_columns, labels, f"data/plots/spre_3_parameters_seed{seed}.png", plot_raw_estimates=True, title=title,     
                marker=markers,
                linewidth=2,
                markersize=10,
                show_plot=show_plot)
        
    ###################################
    # Compare different methods for Two Spheres scenario 300
    files = [
        f"data/mujoco/results/output_two_spheres_abs_errors_eval_300.dat",
        f"data/mujoco/results/output_two_spheres_abs_errors_eval_300_Gaussian.dat",            
        f"data/mujoco/results/output_two_spheres_abs_errors_eval_300_Matern12.dat",
        f"data/mujoco/results/output_two_spheres_abs_errors_eval_300_Matern32.dat",
        f"data/mujoco/results/output_two_spheres_abs_errors_eval_300_MRE.dat",
        f"data/mujoco/results/output_two_spheres_abs_errors_eval_300_GRE.dat",
    ]

    h_columns = ["h"] * len(files)

    labels = ["SPRE White", "SPRE Gaussian", r"SPRE Mat\'{e}rn-$\frac{1}{2}$", r"SPRE Mat\'{e}rn-$\frac{3}{2}$", "MRE", "GRE White"]
    title = None #"Absolute Errors of Estimates"
    markers = itertools.cycle(('o', 's', 'v', '^', '+', 'x', '*'))
    plot_multiple_spre_abs(files, h_columns, labels, f"data/plots/spre_two_spheres_abs_errors.png", plot_raw_estimates=True, title=title,     
            marker=markers,
            linewidth=2,
            markersize=10,
            show_plot=show_plot,
            x_lims=(1e-16, 1e-8))
    
    ###################################
    # Compare different methods for Many Shapes scenario 203
    files = [
        f"data/mujoco/results/output_many_shapes_abs_errors_eval_203.dat",
        f"data/mujoco/results/output_many_shapes_abs_errors_eval_203_Gaussian.dat",            
        f"data/mujoco/results/output_many_shapes_abs_errors_eval_203_Matern12.dat",
        f"data/mujoco/results/output_many_shapes_abs_errors_eval_203_Matern32.dat",
        f"data/mujoco/results/output_many_shapes_abs_errors_eval_203_MRE.dat",
        f"data/mujoco/results/output_many_shapes_abs_errors_eval_203_GRE.dat",
    ]

    h_columns = ["h"] * len(files)

    labels = ["SPRE White", "SPRE Gaussian", r"SPRE Mat\'{e}rn-$\frac{1}{2}$", r"SPRE Mat\'{e}rn-$\frac{3}{2}$", "MRE", "GRE White"]
    title = None #"Absolute Errors of Estimates"
    markers = itertools.cycle(('o', 's', 'v', '^', '+', 'x', '*'))
    plot_multiple_spre_abs(files, h_columns, labels, f"data/plots/spre_five_shapes_abs_errors.png", plot_raw_estimates=True, title=title,     
            marker=markers,
            linewidth=2,
            markersize=10,
            show_plot=show_plot,
            x_lims=(1e-16, 1e-8))
    
    ###################################
    # Plot error bar plot for 
    for model, scenario, pos_inset in zip(["two_spheres", "many_shapes"], [300, 203], ["center left", "center left"]):
        plt.close("all")

        # Create a subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        labels = ["SPRE White", "SPRE Gaussian",  r"SPRE Mat\'{e}rn-$\frac{1}{2}$", r"SPRE Mat\'{e}rn-$\frac{3}{2}$", "MRE"]
        methods = ["", "_Gaussian", "_Matern12", "_Matern32", "_MRE"]

        for plot_num, method, label in zip(range(1, 5), methods, labels):
            files = [f"data/mujoco/results/output_{model}_{scenario}{method}.dat"]
            true_val_file = f"data/mujoco/results/output_{model}_abs_errors_eval_{scenario}{method}.dat"
            h_columns = ["h"]           

            plt.subplot(2, 2, plot_num)

            plot_spre_results(files=files,
                            true_value_filename=true_val_file,
                            h_columns=h_columns,
                            #labels=[label],
                            #y_logscale=True,
                            title=label,                            
                            x_lims=(1e-16, 1e-11),
                            pos_inset=pos_inset)

        plt.tight_layout()
        model_str = model
        if model == "many_shapes":
            model_str = "five_shapes"
        output_file=f"data/plots/spre_{model_str}_error_bars.png"
        plt.savefig(output_file)

        if show_plot:
            plt.show()

    ###############################################
    # Plot bases plots
    show_bar = False
    plot_basis_file("data/mujoco/results/output_two_spheres_bases_300.dat", "data/plots/spre_two_spheres_bases_plot_white.png", "Two Spheres Model, SPRE White", show_bar, show_plot)
    plot_basis_file("data/mujoco/results/output_two_spheres_bases_300_Gaussian.dat", "data/plots/spre_two_spheres_bases_plot_gaussian.png", "Two Spheres Model, SPRE Gaussian", show_bar, show_plot)
    plot_basis_file("data/mujoco/results/output_two_spheres_bases_300_Matern12.dat", "data/plots/spre_two_spheres_bases_plot_matern12.png", r"Two Spheres Model, SPRE Mat\'{e}rn-$\frac{1}{2}$", show_bar, show_plot)
    plot_basis_file("data/mujoco/results/output_two_spheres_bases_300_Matern32.dat", "data/plots/spre_two_spheres_bases_plot_matern32.png", r"Two Spheres Model, SPRE Mat\'{e}rn-$\frac{3}{2}$", show_bar, show_plot)

    plot_basis_file("data/mujoco/results/output_many_shapes_bases_203.dat", "data/plots/spre_five_shapes_bases_plot_white.png", "Five Shapes Model, SPRE White", show_bar, show_plot)
    plot_basis_file("data/mujoco/results/output_many_shapes_bases_203_Gaussian.dat", "data/plots/spre_five_shapes_bases_plot_gaussian.png", "Five Shapes Model, SPRE Gaussian", show_bar, show_plot)
    plot_basis_file("data/mujoco/results/output_many_shapes_bases_203_Matern12.dat", "data/plots/spre_five_shapes_bases_plot_matern12.png", r"Five Shapes Model, SPRE Mat\'{e}rn-$\frac{1}{2}$", show_bar, show_plot)
    plot_basis_file("data/mujoco/results/output_many_shapes_bases_203_Matern32.dat", "data/plots/spre_five_shapes_bases_plot_matern32.png", r"Five Shapes Model, SPRE Mat\'{e}rn-$\frac{3}{2}$", show_bar, show_plot)

    plot_basis_file("data/flock/results/output_flock_bases_321.dat", "data/plots/spre_flock_bases_plot_white.png", "Flock Model, SPRE White", show_bar, show_plot)
    plot_basis_file("data/flock/results/output_flock_bases_321_Gaussian.dat", "data/plots/spre_flock_bases_plot_gaussian.png", "Flock Model, SPRE Gaussian", show_bar, show_plot)
    plot_basis_file("data/flock/results/output_flock_bases_321_Matern12.dat", "data/plots/spre_flock_bases_plot_matern12.png", r"Flock Model, SPRE Mat\'{e}rn-$\frac{1}{2}$", show_bar, show_plot)
    plot_basis_file("data/flock/results/output_flock_bases_321_Matern32.dat", "data/plots/spre_flock_bases_plot_matern32.png", r"Flock Model, SPRE Mat\'{e}rn-$\frac{3}{2}$", show_bar, show_plot)

