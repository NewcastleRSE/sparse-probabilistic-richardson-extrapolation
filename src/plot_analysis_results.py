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
import itertools
import collections.abc
from pathlib import Path
from PIL import Image
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
    y_logscale : bool = False,
    x_lims : tuple = None,
    y_lims : tuple = None,
    *args,
    **kwargs
) -> None:
    """
    Plot SPRE estimates (mu) vs discretization parameter h from one or more files,
    with optional error bars if the file contains a 'var' column.

    Parameters:
        files : list[str]        Paths to SPRE result files (tab-separated)
        h_columns : str          Name of the column containing h values (default "h")
        labels : list[str]       Optional legend labels for each file
        output_file : str        Filename for saving the plot
        true_value_filename: str Filename containing true value
        true_value : float       Optional horizontal line for the "true" value
        title : str              Plot title
        *args : tuple            Positional arguments passed to plt.plot (marker, linestyle, etc.)
        **kwargs : dict          Keyword arguments passed to plt.plot (color, alpha, linewidth, etc.)

    Returns:
        None
    """

    # Set true value from file if given
    if true_value_filename:
        df = pd.read_csv(true_value_filename, sep="\t")
        true_value = df['true_value'][0]

    for file, h_col, label in zip(files, h_columns, labels):
        # Read tab-separated SPRE results
        df = pd.read_csv(file, sep="\t")

        # Check column exists
        if h_col not in df.columns:
            raise ValueError(f"Column '{h_col}' not found in {file}. Available columns: {list(df.columns)}")

        if x_lims is not None:          
            # Redefine data to within this limit
            x_vals = df[h_col]
            mask = (x_vals >= x_lims[0]) & (x_vals <= x_lims[1])
            df = df[mask]   

        x_vals = df[h_col]
        y_vals = df["mu"]

        # Plot with optional error bars if 'var' exists
        if "var" in df.columns:
            y_err = 2 * np.sqrt(df["var"]) # 2 std devs
            plt.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
                fmt='o-',
                capsize=5,
                ecolor='black',
                markersize=6,
                label=label,                
                *args,
                **kwargs
            )
        else:
            plt.plot(x_vals, y_vals, 'o-', markersize=6, label=label, *args, **kwargs)

    plt.xscale("log")
    if y_logscale:
        plt.yscale("log")
    plt.xlabel(r"$h$")
    plt.ylabel("estimate")
    plt.title(title)
    #plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)

    if y_lims is not None:
        plt.ylim(y_lims)    

    # Optional horizontal line for true value
    if true_value is not None:
        plt.axhline(y=true_value, color='red', linestyle='--', linewidth=1)
        print(f"\nTrue value calculated as f(0) = {true_value}\n")

  

def plot_multiple_spre_abs(
    files,
    h_columns,
    labels,
    output_file,
    title = "Absolute Errors of SPRE Estimates",
    show_plot = True, 
    plot_raw_estimates = False,  
    marker=None,
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
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    if show_plot:
        plt.show()


if __name__ == "__main__":

    # Show any plots (except combined flock sim pngs)
    show_plot = True
       
    # Combine flock model time points
    files = [
            "data/flock/results/flock1_start.png",
            "data/flock/results/flock1_1_second_with_trail.png",
            "data/flock/results/flock1_5_seconds_no_trail.png"
        ]
    
    output_file = "data/plots/flock_sim_3_timepoints.png"

    cropped_images = []

    for file in files:
        img = Image.open(file)
        w, h = img.size

        # Crop middle of width
        left = int(0.12 * w)
        right = int(0.85 * w)

        cropped = img.crop((left, 0, right, h))
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

    print(f"Three flock simulation plots saved to {output_file}")
  

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

    # Compare different methods
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