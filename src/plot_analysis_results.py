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
from pathlib import Path


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


def plot_multiple_spre_abs(
    files,
    h_columns,
    labels,
    output_file,
    title = "Absolute Errors of SPRE Estimates",
    show_plot = True, 
    plot_raw_estimates = False,  
    *args,           # Positional arguments for plt.plot (like marker, linestyle)
    **kwargs         # Keyword arguments for plt.plot (like color, alpha, linewidth)
):
    """
    Plot multiple SPRE absolute-error curves on one log-log plot.

    Parameters:
        files : list[str or Path]  Paths to result files   
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

    plt.close("all")
    plt.figure(figsize=(7, 6))

    for file, h_col, label in zip(files, h_columns, labels):
        df = pd.read_csv(file, sep="\t")

        if h_col not in df.columns:
            raise ValueError(
                f"Column '{h_col}' not found in {file}. "
                f"Available columns: {list(df.columns)}"
            )

        x_vals = df[h_col]
        y_vals = df["abs_err_spre_estimate"]

        plt.plot(
            x_vals,
            y_vals,           
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
                markersize=8,
                alpha=0.4,
                zorder = 0,
                label="raw estimates" if i == 0 else None
            )
            i += 1

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel("absolute error")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    if show_plot:
        plt.show()


if __name__ == "__main__":
    # Example usage
    files = [
        "data/flock/results/output_flock_abs_errors_eval_307.dat",
        "data/flock/results/output_flock_abs_errors_eval_307_GRE.dat"
    ]

    h_columns = ["h", "h"]

    labels = ["SPRE white", "GRE white"]
    title ="Absolute Errors of SPRE Estimates"

    plot_multiple_spre_abs(files, h_columns, labels, "data/plots/spre_122_comparison.png", plot_raw_estimates=True,      
            marker="o",
            linewidth=2,
            markersize=10)

    # 
    files = [
        "data/flock/results/output_flock_abs_errors_eval_321.dat",
        "data/flock/results/output_flock_abs_errors_eval_320.dat"
    ]

    h_columns = ["h", "h"]

    labels = ["SPRE white, from 8 estimates", "SPRE white, from 16 estimates"]
    title ="Absolute Errors of SPRE Estimates"

    plot_multiple_spre_abs(files, h_columns, labels, "data/plots/spre_3_parameters_comparison.png", plot_raw_estimates=True,      
            marker="o",
            linewidth=2,
            markersize=10)
    
    # Compare different methods
    files = [
        "data/flock/results/output_flock_abs_errors_eval_321.dat",
        "data/flock/results/output_flock_abs_errors_eval_321_Gaussian.dat",
        "data/flock/results/output_flock_abs_errors_eval_321_GaussianARD.dat",
        "data/flock/results/output_flock_abs_errors_eval_321_Matern12.dat",
        "data/flock/results/output_flock_abs_errors_eval_321_Matern32.dat",
        "data/flock/results/output_flock_abs_errors_eval_321_MRE.dat",
        "data/flock/results/output_flock_abs_errors_eval_321_GRE.dat",
    ]

    h_columns = ["h"] * 7

    labels = ["SPRE white", "SPRE Gaussian", "SPRE GaussianARD", "SPRE Matern1/2", "SPRE Matern3/2", "MRE", "GRE white"]
    title ="Absolute Errors of Estimates"

    plot_multiple_spre_abs(files, h_columns, labels, "data/plots/spre_3_parameters.png", plot_raw_estimates=True,      
            marker="o",
            linewidth=2,
            markersize=10)