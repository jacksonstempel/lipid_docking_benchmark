#!/usr/bin/env python3
"""
Make publication-style plots from the benchmark CSV outputs.

Plain-language overview

- The benchmark writes two CSV files:
  - `benchmark_summary.csv` (best-per-target rows)
  - `benchmark_allposes.csv` (one row per evaluated pose)
- This script reads those CSVs and generates figures comparing methods (Boltz vs Vina).
- It writes figures (PDF by default, optional PNG previews) to an output directory.

Run:

`python scripts/plot_results.py --help`
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable

# Ensure Matplotlib's cache/config directory is writable.
#
# We keep this in a repo-local cache folder so it never mixes with human-facing outputs.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MPLCONFIGDIR = _PROJECT_ROOT / ".cache" / "lipid_benchmark" / "matplotlib"
try:
    _MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
    # Force a non-GUI backend so Matplotlib doesn't try to load Qt (which can emit
    # QStandardPaths warnings on some systems).
    os.environ.setdefault("MPLBACKEND", "Agg")
except OSError:
    pass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

try:  # optional
    import scienceplots  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    scienceplots = None

try:  # optional
    import cmocean  # type: ignore
except Exception:  # pragma: no cover
    cmocean = None

_THEME_APPLIED = False


@dataclass(frozen=True)
class SummaryFrames:
    """
    Convenience container for the two “matched” summary tables we compare.

    - `boltz`: rows where `method == "boltz"`
    - `vina_top1`: rows where `method == "vina_top1"`

    These are kept in the same PDBID order so plots can compare the same targets.
    """
    boltz: pd.DataFrame
    vina_top1: pd.DataFrame


def _finite(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas column to a clean numeric array.

    - Coerces non-numeric entries (e.g., "NA") to missing values.
    - Drops NaN/Inf values.
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return x


def _kde_xy(
    x: np.ndarray,
    *,
    xmin: float,
    xmax: float,
    n: int = 256,
    bw_adjust: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a smooth density curve (KDE) for a 1D distribution.

    Returns `(grid, density)` arrays suitable for plotting, or empty arrays when there
    is not enough data to estimate a density (fewer than 3 points).
    """
    if x.size < 3:
        return np.array([]), np.array([])
    grid = np.linspace(float(xmin), float(xmax), int(n))
    # Smaller bw_adjust -> less smoothing (more local detail).
    kde = gaussian_kde(x, bw_method=lambda s: s.scotts_factor() * float(bw_adjust))
    y = kde(grid)
    return grid, y


def _vina_topk_best_label(k: int) -> str:
    return f"Vina top-{int(k)} best"


def _vina_topk_per_target(
    vina_pose_df: pd.DataFrame,
    *,
    metric_col: str,
    k: int,
    prefer: str,
) -> np.ndarray:
    """
    Return one value per target (PDB ID) for a Vina "top-K" evaluation.

    What "top-K" means in these distribution plots:
    - Vina produces a ranked list of poses (pose_index=1 is its top suggestion).
    - For each target and each K, we summarize the *best value found in the first K poses*.

    How "best" is defined depends on the metric:
    - RMSD metrics: lower is better -> take the minimum across pose_index <= K.
    - Overlap/score metrics: higher is better -> take the maximum across pose_index <= K.

    This differs from plotting *all* poses up to K (multiple points per target),
    which will not necessarily improve as K increases.
    """
    if prefer not in {"min", "max"}:
        raise ValueError("prefer must be 'min' or 'max'")

    if metric_col not in vina_pose_df.columns:
        return np.array([], dtype=float)

    df = vina_pose_df[["pdbid", "pose_index", metric_col]].copy()
    df["pose_index"] = pd.to_numeric(df["pose_index"], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["pdbid", "pose_index", metric_col])
    df = df[df["pose_index"] <= int(k)]
    if df.empty:
        return np.array([], dtype=float)

    grouped = df.groupby("pdbid")[metric_col]
    series = grouped.min() if prefer == "min" else grouped.max()
    return _finite(series)


def _vina_topk_series_per_target(
    vina_pose_df: pd.DataFrame,
    *,
    metric_col: str,
    k: int,
    prefer: str,
) -> pd.Series:
    """
    Like `_vina_topk_per_target`, but returns a Series indexed by PDB ID.

    This is useful for plots where each target should be counted once (e.g., top-K success
    rate curves).
    """
    if prefer not in {"min", "max"}:
        raise ValueError("prefer must be 'min' or 'max'")
    if metric_col not in vina_pose_df.columns:
        return pd.Series(dtype=float)

    df = vina_pose_df[["pdbid", "pose_index", metric_col]].copy()
    df["pose_index"] = pd.to_numeric(df["pose_index"], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=["pdbid", "pose_index", metric_col])
    df = df[df["pose_index"] <= int(k)]
    if df.empty:
        return pd.Series(dtype=float)

    grouped = df.groupby("pdbid")[metric_col]
    return grouped.min() if prefer == "min" else grouped.max()


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for an empirical CDF."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def _load_frames(summary_csv: Path) -> SummaryFrames:
    """
    Load `benchmark_summary.csv` and split it into Boltz and Vina-top1 tables.

    We also validate that both tables contain the same set of targets (same PDBIDs),
    so plots compare like-with-like.
    """
    df = pd.read_csv(summary_csv)
    boltz = df[df["method"] == "boltz"].copy()
    vina_top1 = df[df["method"] == "vina_top1"].copy()
    if len(boltz) == 0 or len(vina_top1) == 0:
        raise RuntimeError("Missing boltz or vina_top1 rows in summary CSV.")
    boltz = boltz.sort_values("pdbid").reset_index(drop=True)
    vina_top1 = vina_top1.sort_values("pdbid").reset_index(drop=True)
    if not (boltz["pdbid"].to_numpy() == vina_top1["pdbid"].to_numpy()).all():
        raise RuntimeError("Summary CSV boltz/vina_top1 PDBID sets do not match.")
    return SummaryFrames(boltz=boltz, vina_top1=vina_top1)


def _apply_pub_style() -> None:
    """
    Set Matplotlib defaults for clean, publication-style figures.

    This adjusts fonts, line widths, tick styles, and PDF settings so the output looks
    consistent across machines. We use Matplotlib’s built-in “mathtext” support so
    the script does not depend on a system LaTeX installation.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # Prefer a Unicode-complete font so symbols like "<=" render correctly
            # (Computer Modern encodings can yield odd glyph substitutions like "¡").
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "CMU Serif", "Computer Modern Roman"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "axes.labelpad": 6,
            "legend.fontsize": 9,
            "legend.framealpha": 0.0,
            "legend.edgecolor": "none",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
            "mathtext.fontset": "dejavuserif",
        }
    )


def _apply_theme() -> None:
    """
    Apply a consistent plotting theme (fonts, styles, and optional scienceplots presets).

    This is called once near the start of `main()` so all figures share a coherent style.
    """
    _apply_pub_style()
    if scienceplots is not None:
        plt.style.use(["science", "nature", "no-latex"])
    # Apply our refined settings after any style overrides.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "CMU Serif", "Computer Modern Roman"]
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = False
    # Disable ticks on top and right to prevent artifacts
    plt.rcParams["xtick.top"] = False
    plt.rcParams["ytick.right"] = False
    plt.rcParams["xtick.minor.top"] = False
    plt.rcParams["ytick.minor.right"] = False
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False


def _apply_theme_once() -> None:
    """
    Apply the plotting theme at most once per process.

    This keeps plot functions simple while avoiding repeated global Matplotlib mutation.
    """
    global _THEME_APPLIED
    if _THEME_APPLIED:
        return
    _apply_theme()
    _THEME_APPLIED = True


def _palette3() -> tuple[str, str, str]:
    """Return a refined, publication-quality 3-color palette."""
    # Elegant, distinguishable palette with good contrast
    return ("#2E86AB", "#E94F37", "#41B3A3")  # Teal blue, Vermilion, Sea green


def _palette4() -> tuple[str, str, str, str]:
    """
    Return a colorblind-friendly 4-color palette.

    Used for plots with four curves (Boltz + Vina top-K best curves).
    """
    # Okabe–Ito inspired: colorblind-friendly and less harsh than saturated primaries.
    # Order: Boltz, Vina top-1 best, Vina top-5 best, Vina top-20 best
    return ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _add_colorbar(mappable, *, ax: plt.Axes, label: str) -> None:
    """
    Add a colorbar with typography consistent with the rest of the figures.

    Axis labels are 11pt in this script, and ticks are 9pt; we mirror that here.
    """
    cb = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, aspect=25)
    cb.set_label(label, fontsize=11, labelpad=6)
    cb.ax.tick_params(labelsize=9)
    cb.outline.set_linewidth(0.5)


def _median_iqr_trend(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: np.ndarray,
    min_n: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a binned trend summary (median + interquartile range).

    Returns `(centers, median, q25, q75)` arrays. Bins with fewer than `min_n` points
    are filled with NaN so plotting code can naturally skip them.
    """
    centers = 0.5 * (bins[:-1] + bins[1:])
    meds: list[float] = []
    q25s: list[float] = []
    q75s: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (x >= lo) & (x < hi)
        vals = y[m]
        if vals.size < min_n:
            meds.append(np.nan)
            q25s.append(np.nan)
            q75s.append(np.nan)
            continue
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        meds.append(float(q50))
        q25s.append(float(q25))
        q75s.append(float(q75))
    return centers, np.array(meds, float), np.array(q25s, float), np.array(q75s, float)


@contextlib.contextmanager
def _suppress_stderr_substrings(substrings: tuple[str, ...]):
    """
    Suppress noisy C-level stderr messages that aren't actionable for users.

    Some PDF backends/libraries may emit messages directly to file descriptor 2
    (bypassing Python warnings). We capture them, drop known-noisy lines, and
    re-emit anything else so real issues remain visible.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass

    original_fd = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            os.dup2(tmp.fileno(), 2)
            try:
                yield
            finally:
                try:
                    sys.stderr.flush()
                except Exception:
                    pass
                os.dup2(original_fd, 2)

                tmp.seek(0)
                data = tmp.read().decode(errors="ignore")
                if data:
                    kept: list[str] = []
                    for line in data.splitlines():
                        if any(s in line for s in substrings):
                            continue
                        kept.append(line)
                    if kept:
                        sys.stderr.write("\n".join(kept) + "\n")
                        sys.stderr.flush()
    finally:
        os.close(original_fd)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save a figure to PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with _suppress_stderr_substrings(("timestamp seems very low",)):
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _save_preview_png(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """
    Save a “preview” PNG version of a figure.

    PDFs are preferred for publication, but PNGs are convenient for quick viewing in file
    browsers or chat. These can optionally be pruned after the run.
    """
    save_dir = out_dir / "_preview"
    save_dir.mkdir(parents=True, exist_ok=True)
    with _suppress_stderr_substrings(("timestamp seems very low",)):
        fig.savefig(save_dir / f"{stem}.png", bbox_inches="tight", dpi=300)


def _prune_non_pdf(out_dir: Path) -> None:
    """Remove preview outputs; PDFs are always preserved."""
    path = out_dir / "_preview"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _save_figure(
    fig: plt.Figure,
    out_dir: Path,
    *,
    stem: str,
    preview_png: bool,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
    use_tight_layout: bool = True,
) -> None:
    """
    Save a figure in the standard output structure.

    What it writes:
    - Always writes a PDF named `{stem}.pdf`.
    - Optionally writes PNG previews to `{out_dir}/_preview/`.
    """
    if use_tight_layout:
        if tight_layout_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=tight_layout_rect)
    _save(fig, out_dir, stem)
    if preview_png:
        _save_preview_png(fig, out_dir, stem)


def plot_rmsd_distributions(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    rmsd_cap_a: float = 10.0,
    preview_png: bool = False,
) -> None:
    """
    Plot distributions of RMSD values for Boltz vs Vina (top-K best).

    This figure is meant to answer: “How accurate are the methods overall?” by showing
    smoothed distributions for:
    - ligand RMSD
    - headgroup RMSD

    It also shows how Vina changes as you consider more top-ranked poses, using a
    "top-K best" definition (best value among the first K poses for each target).
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    vina_pose["pose_index"] = pd.to_numeric(vina_pose.get("pose_index"), errors="coerce")
    max_pose_index = int(vina_pose["pose_index"].max()) if vina_pose["pose_index"].notna().any() else 0

    c_boltz, c_top1, c_top5, c_top20 = _palette4()
    labels = {
        1: _vina_topk_best_label(1),
        5: _vina_topk_best_label(5),
        20: _vina_topk_best_label(20),
    }
    colors = {"Boltz": c_boltz, labels[1]: c_top1, labels[5]: c_top5, labels[20]: c_top20}
    # Only plot top-K curves that are actually present in the CSV.
    # If your benchmark run only saved e.g. 5 poses, "top-20" would be identical to "top-5".
    vina_ks = [k for k in (1, 5, 20) if k <= max_pose_index]
    metrics = [
        ("ligand_rmsd", "Ligand RMSD"),
        ("headgroup_rmsd", "Headgroup RMSD"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, title) in zip(axes, metrics):
        xb = _finite(frames.boltz[col])
        # RMSD is "lower is better": summarize Vina as best-within-top-K per target.
        xv_top1 = _vina_topk_per_target(vina_pose, metric_col=col, k=1, prefer="min")
        xv_top5 = _vina_topk_per_target(vina_pose, metric_col=col, k=5, prefer="min")
        xv_top20 = _vina_topk_per_target(vina_pose, metric_col=col, k=20, prefer="min")

        xmin = 0.0
        xmax = float(rmsd_cap_a)

        # Cap to emphasize the bulk; outliers are still present in other plots.
        xb = xb[(xb >= xmin) & (xb <= xmax)]
        xv_top1 = xv_top1[(xv_top1 >= xmin) & (xv_top1 <= xmax)]
        xv_top5 = xv_top5[(xv_top5 >= xmin) & (xv_top5 <= xmax)]
        xv_top20 = xv_top20[(xv_top20 >= xmin) & (xv_top20 <= xmax)]

        # Use SciPy KDE for full control over the support so the filled curves
        # always start at xmin (avoids hard vertical edges at the first sample).
        vina_curves: list[tuple[str, np.ndarray, str]] = []
        if 20 in vina_ks:
            vina_curves.append((labels[20], xv_top20, colors[labels[20]]))
        if 5 in vina_ks:
            vina_curves.append((labels[5], xv_top5, colors[labels[5]]))
        if 1 in vina_ks:
            vina_curves.append((labels[1], xv_top1, colors[labels[1]]))

        # Plot broader Vina sets first (background), then tighter sets, then Boltz on top.
        plot_order = vina_curves + [("Boltz", xb, colors["Boltz"])]
        for label, x, color in plot_order:
            gx, gy = _kde_xy(x, xmin=xmin, xmax=xmax, bw_adjust=0.75)
            if not gx.size:
                continue
            lw = 2.5
            fill_alpha = {
                "Boltz": 0.20,
                labels[1]: 0.16,
                labels[5]: 0.12,
                labels[20]: 0.10,
            }.get(label, 0.06)
            ax.fill_between(gx, 0.0, gy, color=color, alpha=fill_alpha, lw=0.0)
            ax.plot(gx, gy, color=color, lw=lw, label=label)

            # Median reference line (more robust than mean for skewed distributions).
            if x.size:
                med = float(np.median(x))
                if xmin <= med <= xmax:
                    ax.axvline(med, color=color, lw=1.6, ls="--", alpha=0.9, zorder=1)

        ax.set_title(title, fontsize=13, fontweight="medium", pad=10)
        ax.set_xlabel(r"RMSD ($\mathrm{\AA}$)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    legend_handles = [Line2D([0], [0], color=colors["Boltz"], lw=2.5, label="Boltz")]
    if 1 in vina_ks:
        legend_handles.append(Line2D([0], [0], color=colors[labels[1]], lw=2.5, label=labels[1]))
    if 5 in vina_ks:
        legend_handles.append(Line2D([0], [0], color=colors[labels[5]], lw=2.5, label=labels[5]))
    if 20 in vina_ks:
        legend_handles.append(Line2D([0], [0], color=colors[labels[20]], lw=2.5, label=labels[20]))
    for ax in axes:
        ax.legend(
            handles=legend_handles,
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            edgecolor="none",
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            borderaxespad=0.0,
            ncol=1,
            handlelength=2.2,
            fontsize=10,
        )

    stem = "fig_rmsd_distributions"
    _save_figure(
        fig,
        out_dir,
        stem=stem,
        preview_png=preview_png,
    )

    plt.close(fig)


def plot_paired_rmsd(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    preview_png: bool = False,
) -> None:
    """
    Plot paired per-target comparisons: Boltz RMSD vs Vina top-K best RMSD (2×2 panels).

    What "top-K best" means here:
    - Vina produces a ranked list of poses (pose_index 1 is the top suggestion).
    - For each target and each K, we compute the *best* RMSD found among the first K poses
      (min RMSD over pose_index <= K). This is a standard "is a near-native pose present in
      the top K suggestions?" evaluation.

    Each point is one target. Points are colored by the RMSD difference (Vina - Boltz).
    """
    _apply_theme_once()

    boltz_df = frames.boltz[["pdbid", "ligand_rmsd"]].copy()
    boltz_df["ligand_rmsd"] = pd.to_numeric(boltz_df["ligand_rmsd"], errors="coerce")
    boltz_df = boltz_df.dropna(subset=["ligand_rmsd"]).sort_values("pdbid").reset_index(drop=True)

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    vina_pose["pose_index"] = pd.to_numeric(vina_pose["pose_index"], errors="coerce")
    vina_pose["ligand_rmsd"] = pd.to_numeric(vina_pose["ligand_rmsd"], errors="coerce")
    vina_pose = vina_pose.dropna(subset=["pose_index", "ligand_rmsd"])

    ks = [1, 2, 5, 20]
    vina_topk: dict[int, pd.Series] = {}
    for k in ks:
        sub = vina_pose[vina_pose["pose_index"] <= k]
        # Best RMSD among the first K poses for each target.
        vina_topk[k] = sub.groupby("pdbid")["ligand_rmsd"].min()

    # Build paired arrays in the same PDBID order as the Boltz summary.
    x = boltz_df.set_index("pdbid")["ligand_rmsd"]
    paired: dict[int, pd.Series] = {}
    for k in ks:
        yk = vina_topk[k].reindex(x.index)
        paired[k] = yk

    # Compute global limits and color scale from all available deltas.
    deltas = []
    for k in ks:
        yk = paired[k]
        ok = np.isfinite(x.to_numpy()) & np.isfinite(yk.to_numpy())
        if ok.any():
            deltas.append((yk.to_numpy()[ok] - x.to_numpy()[ok]))
    delta_all = np.concatenate(deltas) if deltas else np.array([0.0])

    if cmocean is not None:
        cmap = cmocean.cm.balance
    else:
        cmap = "RdBu_r"
    vmax = float(np.max(np.abs(delta_all))) if delta_all.size else 1.0

    y_all = np.concatenate([paired[k].dropna().to_numpy(dtype=float) for k in ks if paired[k].dropna().size])
    lim = float(max(np.max(x.to_numpy(dtype=float)), np.max(y_all) if y_all.size else 0.0, 1.0)) * 1.05

    fig = plt.figure(figsize=(9.5, 9.2))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.06],
        left=0.08,
        right=0.92,
        bottom=0.08,
        top=0.92,
        wspace=0.25,
        hspace=0.28,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])
    sc = None
    for ax, k in zip(axes, ks):
        yk = paired[k]
        ok = np.isfinite(x.to_numpy()) & np.isfinite(yk.to_numpy())
        xx = x.to_numpy(dtype=float)[ok]
        yy = yk.to_numpy(dtype=float)[ok]
        delta = yy - xx
        sc = ax.scatter(
            xx,
            yy,
            c=delta,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            s=45,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.plot([0, lim], [0, lim], color="#555555", lw=1.2, ls="--", zorder=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(_vina_topk_best_label(k), fontsize=12, fontweight="medium", pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    for ax in axes[::2]:
        ax.set_ylabel(r"Vina top-K best RMSD ($\mathrm{\AA}$)", fontsize=11)
    for ax in axes[2:]:
        ax.set_xlabel(r"Boltz RMSD ($\mathrm{\AA}$)", fontsize=11)

    if sc is not None:
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(r"$\Delta$RMSD (Vina $-$ Boltz)", fontsize=11, labelpad=6)
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_linewidth(0.5)

    fig.suptitle("Per-target RMSD Comparison (Vina top-K best)", fontsize=13, fontweight="medium", y=0.98)
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")

    stem = "fig_paired_ligand_rmsd_topk"
    _save_figure(fig, out_dir, stem=stem, preview_png=preview_png, use_tight_layout=False)
    plt.close(fig)


def plot_contacts_vs_rmsd(
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    log_counts: bool = False,
    preview_png: bool = False,
) -> None:
    """
    Plot how contact overlap changes with RMSD across predicted poses.

    Each point is one prediction (either a Vina pose or a Boltz prediction). This helps
    visualize the relationship between:
    - geometric accuracy (RMSD)
    - interaction accuracy (headgroup contact overlap)

    The hexbin background is computed over *all* predictions (Vina poses + Boltz) to show
    the overall density of points. The trend line is computed over all points as well.
    """
    _apply_theme_once()

    vina_df = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    boltz_df = allposes_df[allposes_df["method"] == "boltz"].copy()

    x_vina = pd.to_numeric(vina_df["ligand_rmsd"], errors="coerce")
    x_boltz = pd.to_numeric(boltz_df["ligand_rmsd"], errors="coerce")

    y_env_vina = pd.to_numeric(vina_df["head_env_jaccard"], errors="coerce")
    y_env_boltz = pd.to_numeric(boltz_df["head_env_jaccard"], errors="coerce")
    y_typed_vina = pd.to_numeric(vina_df["headgroup_typed_jaccard"], errors="coerce")
    y_typed_boltz = pd.to_numeric(boltz_df["headgroup_typed_jaccard"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, y_vina, y_boltz, title in [
        (axes[0], y_env_vina, y_env_boltz, "Headgroup Environment Overlap"),
        (axes[1], y_typed_vina, y_typed_boltz, "Headgroup Typed Interaction Overlap"),
    ]:
        vina_mask = np.isfinite(x_vina.to_numpy(dtype=float)) & np.isfinite(y_vina.to_numpy(dtype=float))
        boltz_mask = np.isfinite(x_boltz.to_numpy(dtype=float)) & np.isfinite(y_boltz.to_numpy(dtype=float))

        xx_vina = x_vina.to_numpy(dtype=float)[vina_mask]
        yy_vina = y_vina.to_numpy(dtype=float)[vina_mask]
        xx_boltz = x_boltz.to_numpy(dtype=float)[boltz_mask]
        yy_boltz = y_boltz.to_numpy(dtype=float)[boltz_mask]

        xx_all = np.concatenate([xx_vina, xx_boltz]) if (xx_vina.size or xx_boltz.size) else np.array([])
        yy_all = np.concatenate([yy_vina, yy_boltz]) if (yy_vina.size or yy_boltz.size) else np.array([])

        if xx_all.size:
            xcap = float(np.percentile(xx_all, 99.0))
            ax.set_xlim(0.0, max(xcap, 1.0))
            # Use a more vibrant, professional colormap with better contrast
            cmap = "YlGnBu"  # Yellow-Green-Blue, good for density plots
            if cmocean is not None:
                cmap = cmocean.cm.dense
            hb = ax.hexbin(
                xx_all,
                yy_all,
                gridsize=40,
                mincnt=1,
                cmap=cmap,
                linewidths=0.2,
                edgecolors="face",
                alpha=0.92,
                bins="log" if log_counts else None,
            )
            _add_colorbar(hb, ax=ax, label="Count" + (" (log)" if log_counts else ""))

        # Bin RMSD and plot median + IQR as a robust trend line (all predictions).
        if xx_all.size:
            xmax = float(np.percentile(xx_all, 99))
            bins = np.linspace(0.0, max(xmax, 1.0), 10)
            centers, meds, q25s, q75s = _median_iqr_trend(xx_all, yy_all, bins=bins, min_n=10)
            ok = np.isfinite(meds)

            trend_color = "#C41E3A"  # Cardinal red
            ax.fill_between(centers[ok], q25s[ok], q75s[ok], color=trend_color, alpha=0.15, zorder=4)
            ax.plot(centers[ok], meds[ok], color=trend_color, lw=2.5, zorder=5)

        ax.set_title(title, fontsize=12, fontweight="medium", pad=10)
        ax.set_xlabel(r"Ligand RMSD ($\mathrm{\AA}$)", fontsize=11)
        ax.set_ylabel("Jaccard Overlap", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    stem = "fig_contacts_vs_rmsd_pose_cloud"
    _save_figure(
        fig,
        out_dir,
        stem=stem,
        preview_png=preview_png,
    )

    plt.close(fig)


def plot_contact_overlap_distributions(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    metric_col: str,
    title: str,
    stem: str,
    preview_png: bool = False,
) -> None:
    """
    Plot distributions of contact-overlap scores (Jaccard) for Boltz vs Vina (top-K best).

    This is the “interaction accuracy” companion to RMSD plots. The `metric_col` selects
    which overlap metric to visualize (e.g., headgroup environment overlap vs typed
    interaction overlap).
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    vina_pose["pose_index"] = pd.to_numeric(vina_pose.get("pose_index"), errors="coerce")
    max_pose_index = int(vina_pose["pose_index"].max()) if vina_pose["pose_index"].notna().any() else 0
    vina_ks = [k for k in (1, 5, 20) if k <= max_pose_index]

    xb = _finite(frames.boltz[metric_col])
    # Overlap is "higher is better": summarize Vina as best-within-top-K per target.
    xv_top1 = _vina_topk_per_target(vina_pose, metric_col=metric_col, k=1, prefer="max")
    xv_top5 = _vina_topk_per_target(vina_pose, metric_col=metric_col, k=5, prefer="max")
    xv_top20 = _vina_topk_per_target(vina_pose, metric_col=metric_col, k=20, prefer="max")

    xmin, xmax = 0.0, 1.0
    xb = xb[(xb >= xmin) & (xb <= xmax)]
    xv_top1 = xv_top1[(xv_top1 >= xmin) & (xv_top1 <= xmax)]
    xv_top5 = xv_top5[(xv_top5 >= xmin) & (xv_top5 <= xmax)]
    xv_top20 = xv_top20[(xv_top20 >= xmin) & (xv_top20 <= xmax)]

    c_boltz, c_top1, c_top5, c_top20 = _palette4()
    labels = {
        1: _vina_topk_best_label(1),
        5: _vina_topk_best_label(5),
        20: _vina_topk_best_label(20),
    }
    colors = {"Boltz": c_boltz, labels[1]: c_top1, labels[5]: c_top5, labels[20]: c_top20}

    # Reserve space on the right for the legend without shrinking the main plot.
    # (We increase the figure width and use a tight-layout rect below.)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    vina_curves: list[tuple[str, np.ndarray, str]] = []
    if 20 in vina_ks:
        vina_curves.append((labels[20], xv_top20, colors[labels[20]]))
    if 5 in vina_ks:
        vina_curves.append((labels[5], xv_top5, colors[labels[5]]))
    if 1 in vina_ks:
        vina_curves.append((labels[1], xv_top1, colors[labels[1]]))

    # Plot broader Vina sets first (background), then tighter sets, then Boltz on top.
    plot_order = vina_curves + [("Boltz", xb, colors["Boltz"])]
    for label, x, color in plot_order:
        gx, gy = _kde_xy(x, xmin=xmin, xmax=xmax, bw_adjust=0.75)
        if not gx.size:
            continue
        lw = 2.5
        fill_alpha = {
            "Boltz": 0.20,
            labels[1]: 0.16,
            labels[5]: 0.12,
            labels[20]: 0.10,
        }.get(label, 0.06)
        ax.fill_between(gx, 0.0, gy, color=color, alpha=fill_alpha, lw=0.0)
        ax.plot(gx, gy, color=color, lw=lw, label=label)

        if x.size:
            med = float(np.median(x))
            if xmin <= med <= xmax:
                ax.axvline(med, color=color, lw=1.6, ls="--", alpha=0.9, zorder=1)

    ax.set_title(title, fontsize=13, fontweight="medium", pad=10)
    ax.set_xlabel("Jaccard Overlap", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)

    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncol=1,
        fontsize=10,
    )

    _save_figure(
        fig,
        out_dir,
        stem=stem,
        preview_png=preview_png,
        tight_layout_rect=(0.0, 0.0, 0.82, 1.0),
    )

    plt.close(fig)


def plot_topk_success_curves(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    preview_png: bool = False,
) -> None:
    """
    Plot "top-K success" curves for Vina.

    This answers: "If I look at the top K suggested poses, how often is at least one
    'good enough'?"

    We show the fraction of targets whose best-within-top-K RMSD is below a threshold.
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    if vina_pose.empty:
        return
    vina_pose["pose_index"] = pd.to_numeric(vina_pose.get("pose_index"), errors="coerce")
    max_pose_index = int(vina_pose["pose_index"].max()) if vina_pose["pose_index"].notna().any() else 0
    if max_pose_index <= 0:
        return

    ks = list(range(1, min(20, max_pose_index) + 1))
    thresholds = [2.0, 5.0]

    c_boltz, c1, c2, _ = _palette4()
    c_thr = {2.0: c1, 5.0: c2}

    metrics = [
        ("ligand_rmsd", "Ligand RMSD"),
        ("headgroup_rmsd", "Headgroup RMSD"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, title) in zip(axes, metrics):
        # Boltz baseline: one value per target from the summary.
        boltz = pd.to_numeric(frames.boltz.set_index("pdbid")[col], errors="coerce").dropna()

        for thr in thresholds:
            # Vina top-K best per target (min RMSD in top-K).
            rates: list[float] = []
            for k in ks:
                s = _vina_topk_series_per_target(vina_pose, metric_col=col, k=k, prefer="min")
                # Align to targets present in Boltz summary so denominators match across methods.
                s = pd.to_numeric(s.reindex(boltz.index), errors="coerce").dropna()
                rate = float((s <= thr).mean()) if len(s) else float("nan")
                rates.append(rate)

            ax.plot(
                ks,
                rates,
                color=c_thr[thr],
                lw=2.5,
                marker="o",
                markersize=3.5,
                label=f"Vina top-K best (<= {thr:g} $\\mathrm{{\\AA}}$)",
            )

            # Boltz horizontal baseline for the same threshold.
            boltz_rate = float((boltz <= thr).mean()) if len(boltz) else float("nan")
            ax.axhline(boltz_rate, color=c_thr[thr], lw=1.2, ls=":", alpha=0.9)

        ax.set_title(title, fontsize=13, fontweight="medium", pad=10)
        ax.set_xlabel("K (number of Vina poses)", fontsize=11)
        ax.set_ylabel("Fraction of targets", fontsize=11)
        ax.set_xlim(min(ks), max(ks))
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    axes[0].legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        fontsize=10,
    )

    _save_figure(fig, out_dir, stem="fig_vina_topk_success_curves", preview_png=preview_png)
    plt.close(fig)


def plot_ecdf_rmsd(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    preview_png: bool = False,
) -> None:
    """
    ECDF view of RMSD distributions (no KDE smoothing choices).

    Shows Boltz vs Vina top-1 best vs Vina top-20 best (best-within-top-K per target).
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    if vina_pose.empty:
        return
    vina_pose["pose_index"] = pd.to_numeric(vina_pose.get("pose_index"), errors="coerce")

    c_boltz, c1, c2, _ = _palette4()
    colors = {"Boltz": c_boltz, _vina_topk_best_label(1): c1, _vina_topk_best_label(20): c2}

    metrics = [
        ("ligand_rmsd", "Ligand RMSD"),
        ("headgroup_rmsd", "Headgroup RMSD"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, title) in zip(axes, metrics):
        boltz = _finite(frames.boltz[col])
        vina1 = _finite(_vina_topk_series_per_target(vina_pose, metric_col=col, k=1, prefer="min"))
        vina20 = _finite(_vina_topk_series_per_target(vina_pose, metric_col=col, k=20, prefer="min"))

        for label, values in [
            ("Boltz", boltz),
            (_vina_topk_best_label(1), vina1),
            (_vina_topk_best_label(20), vina20),
        ]:
            x, y = _ecdf(values)
            if x.size:
                ax.plot(x, y, color=colors[label], lw=2.5, label=label)

        ax.set_title(title + " (ECDF)", fontsize=13, fontweight="medium", pad=10)
        ax.set_xlabel(r"RMSD ($\mathrm{\AA}$)", fontsize=11)
        ax.set_ylabel("Fraction of targets <= x", fontsize=11)
        ax.set_xlim(left=0.0)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    axes[0].legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        fontsize=10,
    )

    _save_figure(fig, out_dir, stem="fig_rmsd_ecdf", preview_png=preview_png)
    plt.close(fig)


def plot_vina_rank_vs_quality(
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    preview_png: bool = False,
) -> None:
    """
    Show whether Vina's rank correlates with correctness.

    For each pose_index (rank), plot the median (with IQR band) RMSD across targets.
    """
    _apply_theme_once()

    vina = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    if vina.empty:
        return
    vina["pose_index"] = pd.to_numeric(vina.get("pose_index"), errors="coerce")
    vina = vina.dropna(subset=["pose_index", "pdbid"])
    if vina.empty:
        return

    max_pose_index = int(vina["pose_index"].max()) if vina["pose_index"].notna().any() else 0
    ranks = np.arange(1, min(20, max_pose_index) + 1, dtype=int)
    if ranks.size == 0:
        return

    c_boltz, c1, _, _ = _palette4()
    color = c1

    metrics = [
        ("ligand_rmsd", "Ligand RMSD"),
        ("headgroup_rmsd", "Headgroup RMSD"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, title) in zip(axes, metrics):
        medians: list[float] = []
        q25s: list[float] = []
        q75s: list[float] = []
        for r in ranks:
            vals = pd.to_numeric(vina.loc[vina["pose_index"] == r, col], errors="coerce").dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                medians.append(np.nan)
                q25s.append(np.nan)
                q75s.append(np.nan)
                continue
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            medians.append(float(q50))
            q25s.append(float(q25))
            q75s.append(float(q75))

        x = ranks.astype(float)
        med = np.array(medians, dtype=float)
        q25 = np.array(q25s, dtype=float)
        q75 = np.array(q75s, dtype=float)
        ok = np.isfinite(med) & np.isfinite(q25) & np.isfinite(q75)
        if ok.any():
            ax.fill_between(x[ok], q25[ok], q75[ok], color=color, alpha=0.18, lw=0.0)
            ax.plot(x[ok], med[ok], color=color, lw=2.5, marker="o", markersize=3.5)

        ax.set_title(title + " vs Vina rank", fontsize=13, fontweight="medium", pad=10)
        ax.set_xlabel("Vina rank (pose_index)", fontsize=11)
        ax.set_ylabel(r"RMSD ($\mathrm{\AA}$)", fontsize=11)
        ax.set_xlim(float(ranks.min()), float(ranks.max()))
        ax.set_ylim(bottom=0.0)
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    _save_figure(fig, out_dir, stem="fig_vina_rank_vs_rmsd", preview_png=preview_png)
    plt.close(fig)

def main(argv: Iterable[str] | None = None) -> int:
    """
    Command-line entry point for plot generation.

    - Reads CSV inputs (summary + all-poses).
    - Generates a standard set of figures.
    - Writes PDFs (and optionally PNG previews) into `--out-dir`.
    """
    p = argparse.ArgumentParser(description="Generate publication-quality plots from benchmark CSVs.")
    p.add_argument("--summary", default="output/benchmark_summary.csv", help="Path to summary CSV.")
    p.add_argument("--allposes", default="output/benchmark_allposes.csv", help="Path to allposes CSV.")
    p.add_argument("--out-dir", default="plots", help="Output directory for figures.")
    p.add_argument("--log-density", action="store_true", help="Use log10 scaling for hexbin pose densities.")
    p.add_argument("--preview-png", action="store_true", help="Also write PNGs for local preview (then prune).")
    p.add_argument("--keep-preview", action="store_true", help="Keep preview PNGs (no pruning).")
    args = p.parse_args(list(argv) if argv is not None else None)

    def _resolve_csv(arg_value: str) -> Path:
        """
        Resolve a CSV path from either an explicit user path or a set of fallbacks.

        Why this exists:
        - Output CSVs are typically gitignored, so after cloning the repo you may not
          have `output/benchmark_*.csv` yet.
        - Older runs of this repo wrote to `analysis/benchmark/`, so we try that too.
        """
        p = Path(arg_value).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (_PROJECT_ROOT / p).resolve()

    summary_csv = _resolve_csv(str(args.summary))
    allposes_csv = _resolve_csv(str(args.allposes))
    out_dir_arg = Path(args.out_dir).expanduser()
    out_dir = (_PROJECT_ROOT / out_dir_arg).resolve() if not out_dir_arg.is_absolute() else out_dir_arg.resolve()

    if not summary_csv.is_file() or not allposes_csv.is_file():
        missing = []
        if not summary_csv.is_file():
            missing.append(f"summary CSV not found: {summary_csv}")
        if not allposes_csv.is_file():
            missing.append(f"allposes CSV not found: {allposes_csv}")
        msg = (
            "Cannot generate plots because benchmark CSVs are missing.\n"
            + "\n".join(missing)
            + "\n\nRun the benchmark first:\n"
            "  python scripts/benchmark.py --out-dir output\n"
            "Then re-run plotting:\n"
            "  python scripts/plot_results.py --out-dir plots\n"
        )
        raise FileNotFoundError(msg)

    frames = _load_frames(summary_csv)
    allposes_df = pd.read_csv(allposes_csv)
    preview_png = bool(args.preview_png)

    # Print a single helpful note if the input file doesn't contain enough Vina poses
    # to support "top-20 best" curves.
    try:
        vina_pose_idx = pd.to_numeric(
            allposes_df.loc[allposes_df["method"] == "vina_pose", "pose_index"], errors="coerce"
        )
        max_pose_index = int(vina_pose_idx.max()) if vina_pose_idx.notna().any() else 0
        if 0 < max_pose_index < 20:
            sys.stderr.write(
                f"[plot_results] Note: {allposes_csv} contains Vina pose_index up to {max_pose_index}; "
                "to generate top-20 best curves, re-run: python scripts/benchmark.py --vina-max-poses 20\n"
            )
    except Exception:
        pass

    tasks: list[tuple[str, callable]] = [
        (
            "RMSD distributions",
            lambda: plot_rmsd_distributions(frames, allposes_df, out_dir=out_dir, preview_png=preview_png),
        ),
        ("Paired RMSD (top-K best)", lambda: plot_paired_rmsd(frames, allposes_df, out_dir=out_dir, preview_png=preview_png)),
        ("RMSD ECDF", lambda: plot_ecdf_rmsd(frames, allposes_df, out_dir=out_dir, preview_png=preview_png)),
        ("Vina rank vs RMSD", lambda: plot_vina_rank_vs_quality(allposes_df, out_dir=out_dir, preview_png=preview_png)),
        ("Vina top-K success", lambda: plot_topk_success_curves(frames, allposes_df, out_dir=out_dir, preview_png=preview_png)),
        (
            "Contacts vs RMSD",
            lambda: plot_contacts_vs_rmsd(
                allposes_df, out_dir=out_dir, log_counts=bool(args.log_density), preview_png=preview_png
            ),
        ),
        (
            "Headgroup environment overlap",
            lambda: plot_contact_overlap_distributions(
                frames,
                allposes_df,
                out_dir=out_dir,
                metric_col="head_env_jaccard",
                title="Headgroup Environment Overlap",
                stem="fig_head_env_overlap_distributions",
                preview_png=preview_png,
            ),
        ),
    ]

    def _render_progress(done: int, total: int, label: str) -> None:
        width = 24
        filled = int(round(width * done / max(total, 1)))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(f"\r[{bar}] {done}/{total} {label:<32}")
        sys.stderr.flush()

    total = len(tasks)
    _render_progress(0, total, "Starting")
    for idx, (label, fn) in enumerate(tasks, start=1):
        _render_progress(idx - 1, total, label)
        fn()
        _render_progress(idx, total, label)
    sys.stderr.write("\n")

    if preview_png and not bool(args.keep_preview):
        _prune_non_pdf(out_dir)
    sys.stderr.write(f"Wrote PDF figures to {out_dir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
