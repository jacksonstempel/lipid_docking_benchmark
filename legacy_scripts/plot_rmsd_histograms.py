#!/usr/bin/env python3
"""Generate RMSD histograms and density plots for boltz and vina methods."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


DATA_PATH = Path("analysis/benchmark_20251024_105730.csv")
OUTPUT_DIR = Path("analysis/plots")
TITLE = "Ligand Placement Accuracy: Boltz vs Vina"
X_LABEL = "RMSD (Å)"
Y_LABEL = "Frequency"
METHOD_COLORS: Dict[str, str] = {"boltz": "#D81B60", "vina": "#1E88E5"}
BAR_EDGE_COLOR = "#1A1A1A"


def configure_matplotlib() -> None:
    """Set consistent style parameters for publication-style figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "figure.figsize": (8, 5),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linestyle": "-",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.labelsize": 12,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "DejaVu Sans",
        }
    )


def load_data() -> pd.DataFrame:
    """Load RMSD benchmark data."""
    df = pd.read_csv(DATA_PATH)
    expected = {"method", "rmsd_global"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {', '.join(sorted(missing))}")
    return df


def make_bin_edges(bin_width: float, data_max: float) -> np.ndarray:
    """Generate bin edges with the final bin capturing all RMSD > 10 Å."""
    edges = np.arange(0, 10 + bin_width, bin_width)
    if data_max > 10:
        edges = np.append(edges, data_max + bin_width)
    else:
        edges = np.append(edges, 10 + bin_width)
    return edges


def make_bin_labels(edges: np.ndarray, bin_width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin centers, widths, and human-readable labels."""
    widths = np.diff(edges)
    centers = edges[:-1] + widths / 2
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if left >= 10:
            labels.append("10+")
        else:
            if np.isclose(bin_width, 1.0):
                labels.append(f"{int(left)}–{int(right)}")
            else:
                labels.append(f"{left:.1f}–{right:.1f}")
    return centers, widths, np.array(labels)


def plot_single_histogram(
    series: pd.Series, method: str, bin_width: float, edges: np.ndarray, output_path: Path
) -> None:
    """Plot histogram for a single method."""
    counts, _ = np.histogram(series[np.isfinite(series)], bins=edges)
    counts[-1] += np.isinf(series).sum()
    centers, widths, labels = make_bin_labels(edges, bin_width)

    fig, ax = plt.subplots()
    ax.bar(
        centers,
        counts,
        width=widths * 0.9,
        color=METHOD_COLORS[method],
        alpha=0.85,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
    )
    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(0, max(10 + bin_width, centers[-1] + widths[-1] / 2))
    ax.set_ylim(0, max(counts) * 1.15 if counts.any() else 1)
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_histogram(
    data: pd.DataFrame, bin_width: float, edges: np.ndarray, output_path: Path
) -> None:
    """Plot grouped histogram comparing boltz and vina counts for shared bins."""
    centers, widths, labels = make_bin_labels(edges, bin_width)
    counts = {}
    for method in ("vina", "boltz"):
        series = data.loc[data["method"] == method, "rmsd_global"]
        counts_method, _ = np.histogram(series[np.isfinite(series)], bins=edges)
        counts_method[-1] += np.isinf(series).sum()
        counts[method] = counts_method

    bar_width = widths * 0.4

    fig, ax = plt.subplots()
    ax.bar(
        centers - bar_width / 2,
        counts["vina"],
        width=bar_width,
        color=METHOD_COLORS["vina"],
        alpha=0.85,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
        label="vina",
    )
    ax.bar(
        centers + bar_width / 2,
        counts["boltz"],
        width=bar_width,
        color=METHOD_COLORS["boltz"],
        alpha=0.85,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=0.6,
        label="boltz",
    )
    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(0, max(10 + bin_width, centers[-1] + widths[-1] / 2))
    y_max = max(max(counts["vina"]), max(counts["boltz"]))
    ax.set_ylim(0, y_max * 1.15 if y_max else 1)
    ax.legend(frameon=False, loc="upper right")
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_density(
    boltz: pd.Series,
    vina: pd.Series,
    bin_width: float,
    output_path: Path,
) -> None:
    """Plot overlapping density curves with translucent fills."""
    boltz_finite = boltz[np.isfinite(boltz)]
    vina_finite = vina[np.isfinite(vina)]
    x_max = max(boltz_finite.max(), vina_finite.max(), 10) if not boltz_finite.empty and not vina_finite.empty else 10
    x_grid = np.linspace(0, x_max + 2, 500)

    kde_boltz = gaussian_kde(boltz_finite)
    kde_vina = gaussian_kde(vina_finite)

    density_boltz = kde_boltz(x_grid)
    density_vina = kde_vina(x_grid)

    fig, ax = plt.subplots()
    ax.plot(x_grid, density_vina, color=METHOD_COLORS["vina"], linewidth=2.0, label="vina")
    ax.fill_between(
        x_grid,
        density_vina,
        color=METHOD_COLORS["vina"],
        alpha=0.35,
    )
    ax.plot(x_grid, density_boltz, color=METHOD_COLORS["boltz"], linewidth=2.0, label="boltz")
    ax.fill_between(
        x_grid,
        density_boltz,
        color=METHOD_COLORS["boltz"],
        alpha=0.35,
    )

    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_xlim(0, x_max + 2)
    ax.legend(frameon=False, loc="upper right")
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    filtered = data[data["method"].isin({"boltz", "vina"})].copy()

    if filtered.empty:
        raise ValueError("No data found for methods 'boltz' or 'vina'.")

    boltz = filtered.loc[filtered["method"] == "boltz", "rmsd_global"].dropna()
    vina = filtered.loc[filtered["method"] == "vina", "rmsd_global"].dropna()
    finite_max = pd.concat([boltz[np.isfinite(boltz)], vina[np.isfinite(vina)]])
    if finite_max.empty:
        raise ValueError("No finite RMSD values available for plotting.")
    global_max = finite_max.max()

    for bin_width in (0.5, 1.0):
        suffix = f"{str(bin_width).replace('.', '_')}A"
        edges = make_bin_edges(bin_width, global_max)

        plot_single_histogram(
            boltz,
            "boltz",
            bin_width,
            edges,
            OUTPUT_DIR / f"rmsd_hist_boltz_bins_{suffix}.png",
        )
        plot_single_histogram(
            vina,
            "vina",
            bin_width,
            edges,
            OUTPUT_DIR / f"rmsd_hist_vina_bins_{suffix}.png",
        )
        plot_grouped_histogram(
            filtered,
            bin_width,
            edges,
            OUTPUT_DIR / f"rmsd_hist_boltz_vs_vina_bins_{suffix}.png",
        )
        plot_density(
            boltz,
            vina,
            bin_width,
            OUTPUT_DIR / f"rmsd_density_boltz_vs_vina_bins_{suffix}.png",
        )


if __name__ == "__main__":
    main()
