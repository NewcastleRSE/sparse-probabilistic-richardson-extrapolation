##############################################################################
# Figures for the three case studies (two spheres, five shapes, flocking),
# drawn from the stored results files in data/*/results.
##############################################################################

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from paper_figures.style import METHODS, TEXT_WIDTH, NEUTRAL, style_log_axes

# Stored results used in the paper
CASE_STUDIES = {
    "two_spheres": dict(dir="data/mujoco/results", stem="two_spheres", scenario="300",
                        title="Two Spheres 3D Model", h_max_abs=1e-8, h_max_bars=1e-11, inset=True),
    "five_shapes": dict(dir="data/mujoco/results", stem="many_shapes", scenario="203",
                        title="Five Shapes 3D Model", h_max_abs=1e-8, h_max_bars=1e-11, inset=True),
    "flock":       dict(dir="data/flock/results", stem="flock", scenario="321",
                        title="Agent-Based Flocking Model", h_max_abs=1.0, h_max_bars=1e-8, inset=True),
}

# File suffix for each method
SUFFIX = {"SPRE_white": "", "SPRE_Matern12": "_Matern12", "SPRE_Matern32": "_Matern32",
          "SPRE_Gaussian": "_Gaussian", "MRE": "_MRE", "GRE": "_GRE"}
SPRE_KERNELS = ["SPRE_white", "SPRE_Matern12", "SPRE_Matern32", "SPRE_Gaussian"]
KERNEL_NAMES = {"SPRE_white": "white noise kernel", "SPRE_Matern12": r"Mat\'{e}rn-$\frac{1}{2}$ kernel",
                "SPRE_Matern32": r"Mat\'{e}rn-$\frac{3}{2}$ kernel", "SPRE_Gaussian": "Gaussian kernel"}


def _abs_file(cs, method):
    return Path(cs["dir"]) / f"output_{cs['stem']}_abs_errors_eval_{cs['scenario']}{SUFFIX[method]}.dat"

def _est_file(cs, method):
    return Path(cs["dir"]) / f"output_{cs['stem']}_{cs['scenario']}{SUFFIX[method]}.dat"

def _bases_file(cs, method):
    return Path(cs["dir"]) / f"output_{cs['stem']}_bases_{cs['scenario']}{SUFFIX[method]}.dat"


def plot_abs_errors(name: str, out: Path) -> None:
    """Absolute error of each method, together with the raw estimates, as h is varied."""
    cs = CASE_STUDIES[name]
    fig, ax = plt.subplots(figsize=(0.75 * TEXT_WIDTH, 0.42 * TEXT_WIDTH))
    d0 = pd.read_csv(_abs_file(cs, "SPRE_white"), sep="\t")
    d0 = d0[d0.h <= cs["h_max_abs"]]
    i = 1
    while f"abs_err_estimate_{i}" in d0:
        ax.plot(d0.h, d0[f"abs_err_estimate_{i}"], color=NEUTRAL, lw=0.7, alpha=0.6, zorder=1,
                label=METHODS["raw"]["label"] if i == 1 else None)
        i += 1
    methods = list(SUFFIX)
    for k, method in enumerate(methods):
        d = pd.read_csv(_abs_file(cs, method), sep="\t")
        d = d[d.h <= cs["h_max_abs"]]
        m = METHODS[method]
        ax.plot(d.h, d["abs_err_spre_estimate"], color=m["color"], marker=m["marker"], label=m["label"],
                zorder=10 - k, markeredgecolor="white", markeredgewidth=0.4)
    style_log_axes(ax)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel("absolute error")
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(1, len(handles))) + [0]   # raw estimates last
    ax.legend([handles[k] for k in order], [labels[k] for k in order],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.savefig(out)
    plt.close(fig)


def plot_error_bars(name: str, out: Path) -> None:
    """Two standard deviation predictive intervals from SPRE, for each kernel, relative to f(0)."""
    cs = CASE_STUDIES[name]
    true_value = pd.read_csv(_abs_file(cs, "SPRE_white"), sep="\t")["true_value"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 0.62 * TEXT_WIDTH))
    for ax, method in zip(axes.ravel(), SPRE_KERNELS):
        d = pd.read_csv(_est_file(cs, method), sep="\t")
        d = d[d.h <= cs["h_max_bars"]].sort_values("h")
        dev = d["mu"].values - true_value
        err = 2 * np.sqrt(d["var"].values)
        scale = 10.0 ** np.floor(np.log10(np.max(np.abs(dev) + err)))
        m = METHODS[method]
        ax.axhline(0, color="#e34948", ls="--", lw=0.9, zorder=1)
        ax.errorbar(d.h, dev / scale, yerr=err / scale, color=m["color"], marker=m["marker"], ms=3,
                    lw=1.0, elinewidth=0.8, capsize=2, ecolor="#52514e", zorder=3)
        ax.set_xscale("log")
        ax.grid(True, which="major")
        ax.set_title(r"SPRE, " + KERNEL_NAMES[method])
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"estimate $-\, f(\mathbf{0})$")
        ax.text(0.0, 1.01, rf"$\times 10^{{{int(np.log10(scale))}}}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        if cs["inset"]:
            n_zoom = 7
            dz = d.iloc[:n_zoom]
            devz = dz["mu"].values - true_value
            errz = 2 * np.sqrt(dz["var"].values)
            sz = 10.0 ** np.floor(np.log10(np.max(np.abs(devz) + errz)))
            axins = inset_axes(ax, width="44%", height="36%", loc="upper left",
                               bbox_to_anchor=(0.1, 0, 1, 0.97), bbox_transform=ax.transAxes, borderpad=0)
            axins.axhline(0, color="#e34948", ls="--", lw=0.8)
            axins.errorbar(dz.h, devz / sz, yerr=errz / sz, color=m["color"], marker=m["marker"], ms=2.5,
                           lw=0.9, elinewidth=0.7, capsize=1.5, ecolor="#52514e")
            axins.set_xscale("log")
            axins.minorticks_off()
            axins.tick_params(labelsize=6.5, length=2, pad=1)
            axins.tick_params(axis="x", labelbottom=False)
            axins.text(0.98, 0.95, rf"$\times 10^{{{int(np.log10(sz))}}}$", transform=axins.transAxes,
                       ha="right", va="top", fontsize=6.5,
                       bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.9))
            axins.grid(True, which="major")
            for sp in axins.spines.values():
                sp.set_linewidth(0.5)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.9 * (hi - lo))   # head-room for the inset
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.94, hspace=0.42, wspace=0.22)
    fig.savefig(out)
    plt.close(fig)


def read_bases(filename: Path):
    """Parse a bases file into a list of (h, set of index tuples)."""
    blocks = re.split(r"\n\s*\n", Path(filename).read_text().strip())
    out = []
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        mh = re.search(r"h\s*=\s*([0-9.eE+-]+)", lines[0])
        if not mh:
            continue
        A = {tuple(map(int, ln.split())) for ln in lines[1:] if ln.strip() and not ln.startswith("Basis")}
        out.append((float(mh.group(1)), A))
    return out


def plot_index_sets(name: str, method: str, out: Path) -> None:
    """Estimated index set A as h is varied, with its cardinality."""
    cs = CASE_STUDIES[name]
    data = read_bases(_bases_file(cs, method))
    elements = sorted({a for _, A in data for a in A}, key=lambda a: (sum(a), tuple(-x for x in a)))
    hs = np.array([h for h, _ in data])
    m = METHODS[method]
    nrow = len(elements)
    fig = plt.figure(figsize=(TEXT_WIDTH, 0.17 * TEXT_WIDTH + 0.13 * nrow))
    gs = fig.add_gridspec(2, 1, height_ratios=[nrow, 3.2], hspace=0.08)
    ax = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharex=ax)
    for r in range(nrow):
        ax.axhline(r, color="#e6e5e1", lw=0.6, zorder=0)
    for h, A in data:
        rows = [elements.index(a) for a in A]
        ax.scatter([h] * len(rows), rows, s=14, color=m["color"], edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f"$({','.join(map(str, a))})$" for a in elements])
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.grid(False)
    ax.grid(True, axis="x", which="major")
    ax.set_ylabel(r"elements of $A$")
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ax.set_title(f"{cs['title']}, SPRE with {KERNEL_NAMES[method]}")
    axb.bar(hs, [len(A) for _, A in data], width=hs * 0.45, color=m["color"], alpha=0.85)
    axb.set_xscale("log")
    axb.set_ylabel(r"$|A|$")
    from matplotlib.ticker import MaxNLocator
    axb.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    axb.set_xlabel(r"$h$")
    axb.grid(True, axis="x", which="major")
    axb.grid(True, axis="y")
    axb.minorticks_off()
    ax.minorticks_off()
    fig.savefig(out)
    plt.close(fig)


def plot_sequential_design(out: Path, results_dir="data/mujoco/results") -> None:
    """Fixed versus sequential experimental design for the two spheres model, at equal cost."""
    fixed = pd.read_csv(Path(results_dir) / "output_two_spheres_design_fixed.dat", sep="\t")
    seq = pd.read_csv(Path(results_dir) / "output_two_spheres_design_sequential.dat", sep="\t")
    final = seq[seq["iteration"] == seq["iteration"].max()]
    med = final.groupby("h").agg(cost=("cost", "median"), err=("abs_err", "median"))
    fig, ax = plt.subplots(figsize=(0.75 * TEXT_WIDTH, 0.42 * TEXT_WIDTH))
    ax.plot(fixed["cost"], fixed["min_raw_abs_err"], color=NEUTRAL, lw=0.9, marker=".", ms=3, zorder=1,
            label=r"most accurate raw estimate")
    ax.plot(fixed["cost"], fixed["abs_err"], color="#0b0b0b", marker="o", markeredgecolor="white",
            markeredgewidth=0.4, label=r"SPRE, fixed design", zorder=3)
    ax.plot(final["cost"], final["abs_err"], ls="none", marker="s", ms=3.5, color=METHODS["SPRE_Matern12"]["color"],
            alpha=0.45, markeredgewidth=0, label=r"SPRE, sequential design (each seed)", zorder=2)
    ax.plot(med["cost"], med["err"], color=METHODS["SPRE_Matern12"]["color"], marker="s", markeredgecolor="white",
            markeredgewidth=0.4, label=r"SPRE, sequential design (median)", zorder=4)
    style_log_axes(ax)
    ax.set_xlabel(r"total computational cost $\sum_i c(\mathbf{x}_i)$")
    ax.set_ylabel("absolute error")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.savefig(out)
    plt.close(fig)
