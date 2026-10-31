##############################################################################
# Figure for the toy sequential experimental design example (Figure 4 of the
# paper), drawn from data/toy_design/results/toy_design.json, which is
# produced by src/run_toy_design.py.
##############################################################################

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from paper_figures.style import METHODS, TEXT_WIDTH

RESULTS = Path("data/toy_design/results")


def _format_A(A, per_line=4):
    items = [f"({a[0]},{a[1]})" for a in A]
    lines = [",".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    text = r"$A = \{" + lines[0]
    for line in lines[1:]:
        text += r",$" + "\n" + r"$" + line
    return text + r"\}$"


def _format_sigma(sigma):
    exponent = int(np.floor(np.log10(sigma)))
    mantissa = sigma / 10 ** exponent
    if -2 <= exponent <= 1:
        return rf"$\sigma = {sigma:.2g}$"
    return rf"$\sigma = {mantissa:.1f} \times 10^{{{exponent}}}$"


def plot_toy_design(out: Path) -> None:
    res = json.loads((RESULTS / "toy_design.json").read_text())
    its = res["iterations"]
    colour = METHODS["SPRE_Matern12"]["color"]
    fig, axes = plt.subplots(2, 4, figsize=(TEXT_WIDTH, 0.6 * TEXT_WIDTH), sharex=True, sharey=True)
    t = np.linspace(0.5, 1, 200)
    for ax, it in zip(axes.ravel(), its):
        X = np.array(it["X"])
        Xn = np.array(it["X_new"]) if it["X_new"] else np.zeros((0, 2))
        ax.fill_between(t, 1 / (res["budget"] * t), 1, color="#ecebe7", lw=0, zorder=0)  # c(x) <= C
        ax.scatter(X[:, 0], X[:, 1], s=14, facecolor="white", edgecolor="#0b0b0b", linewidth=0.6, zorder=3,
                   label="previously selected")
        ax.scatter(Xn[:, 0], Xn[:, 1], s=16, color=colour, edgecolor="white", linewidth=0.4, zorder=4,
                   label="new design points")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(True)
        ax.set_title(_format_A(it["A"]), fontsize=7, linespacing=1.1)
        ax.text(0.05, 0.08, f"iteration {it['iteration']}\n" + _format_sigma(it["sigma"]), transform=ax.transAxes,
                fontsize=7.5, va="bottom")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x_1$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$x_2$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor="#ecebe7", edgecolor="none"))
    labels.append(r"$c(\mathbf{x}) \leq C$")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.93, hspace=0.45, wspace=0.12)
    fig.savefig(out)
    plt.close(fig)
