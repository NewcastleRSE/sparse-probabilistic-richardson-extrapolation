##############################################################################
# Script to run SPRE analysis given a model parameter file.
#
# From root directory, for example run
# python ./src/run_model_analysis.py ./data/chem_equil/input_1.json
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


def plot_multiple_spre(
    files,
    h_columns,
    labels=None,
    output_file="spre_comparison.png",
    title="Absolute Errors of SPRE Estimates"
):
    """
    Plot multiple SPRE absolute-error curves on one log-log plot.

    Parameters
    ----------
    files : list[str or Path]
        Paths to result files
    h_columns : list[str]
        Name of the h column for each file (e.g. ["h", "h1", "h2"])
    labels : list[str], optional
        Legend labels for each file
    output_file : str
        Output image filename
    title : str
        Plot title
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
            marker="o",
            linewidth=2,
            markersize=10,
            label=label
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("h")
    plt.ylabel("absolute error")
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


if __name__ == "__main__":
    # Example usage
    files = [
        "data/multi_agent/results/output_multi_agent_122.dat",
        "data/multi_agent/results/output_multi_agent_122_GRE.dat"
    ]

    h_columns = [
        "h",
        "h"
    ]

    labels = [
        "SPRE white",
        "GRE white"  
    ]

    plot_multiple_spre(
        files,
        labels,
        output_file="data/multi_agent/results/spre_122_comparison.png"
    )
