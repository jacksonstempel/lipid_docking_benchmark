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
from scipy.stats import gaussian_kde, linregress

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


@dataclass(frozen=True)
class GninaFrames:
    """
    Convenience container for GNINA + Vina per-target metrics merged with Boltz top-1.

    - `per_target`: per-target GNINA/Vina metrics from analyze_gnina_experiment.py
    - `boltz`: boltz top-1 metrics from benchmark_summary.csv (merged into per_target)
    """
    per_target: pd.DataFrame


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


def _vina_topk_by_ligand_rmsd(
    vina_pose_df: pd.DataFrame,
    *,
    return_col: str,
    k: int,
) -> pd.Series:
    """
    Select the pose with the best ligand RMSD among top-K, return another metric.

    This ensures consistent pose selection across all metrics: we always pick the
    pose that minimizes ligand RMSD within the first K poses, then report that
    pose's value for `return_col`.

    This is more realistic than per-metric oracle selection, since in practice
    you'd select one pose (by RMSD or score) and be stuck with its performance
    on all other metrics.
    """
    if "ligand_rmsd" not in vina_pose_df.columns:
        return pd.Series(dtype=float)
    if return_col not in vina_pose_df.columns:
        return pd.Series(dtype=float)

    df = vina_pose_df[["pdbid", "pose_index", "ligand_rmsd", return_col]].copy()
    df["pose_index"] = pd.to_numeric(df["pose_index"], errors="coerce")
    df["ligand_rmsd"] = pd.to_numeric(df["ligand_rmsd"], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna(subset=["pdbid", "pose_index", "ligand_rmsd"])
    df = df[df["pose_index"] <= int(k)]
    if df.empty:
        return pd.Series(dtype=float)

    # For each target, find the row with the minimum ligand RMSD
    idx_best = df.groupby("pdbid")["ligand_rmsd"].idxmin()
    best_rows = df.loc[idx_best].set_index("pdbid")
    return best_rows[return_col]


def _load_gnina_frames(analysis_dir: Path, *, summary_csv: Path) -> GninaFrames:
    """Load GNINA per-target metrics and merge Boltz top-1 metrics by PDB ID."""
    per_target_path = analysis_dir / "per_target.csv"
    if not per_target_path.is_file():
        raise FileNotFoundError(f"GNINA per_target.csv not found: {per_target_path}")
    per_target = pd.read_csv(per_target_path)
    if "boltz_top1_ligand_rmsd" in per_target.columns:
        return GninaFrames(per_target=per_target)

    boltz = pd.read_csv(summary_csv)
    boltz = boltz[boltz["method"] == "boltz"].copy()
    boltz = boltz[[
        "pdbid",
        "ligand_rmsd",
        "headgroup_rmsd",
        "head_env_jaccard",
        "headgroup_typed_jaccard",
    ]].rename(columns={
        "ligand_rmsd": "boltz_top1_ligand_rmsd",
        "headgroup_rmsd": "boltz_top1_headgroup_rmsd",
        "head_env_jaccard": "boltz_head_env_jaccard",
        "headgroup_typed_jaccard": "boltz_headgroup_typed_jaccard",
    })

    merged = per_target.merge(boltz, on="pdbid", how="inner")
    return GninaFrames(per_target=merged)


def _boxplot(
    ax: plt.Axes,
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    fill_alpha: float = 0.45,
    whis: tuple[float, float] | float = 1.5,
) -> None:
    bp = ax.boxplot(
        data,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        whis=whis,
        medianprops={"color": "#222222", "linewidth": 1.6},
        boxprops={"linewidth": 1.0},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(fill_alpha)
    ax.set_title(title, fontsize=12, fontweight="medium", pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)


def _bootstrap_ci(
    values: np.ndarray,
    *,
    stat_fn=np.median,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Return (stat, lo, hi) via bootstrap resampling."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(vals, size=vals.size, replace=True)
        stats[i] = float(stat_fn(sample))
    alpha = (100.0 - ci) / 2.0
    lo = float(np.percentile(stats, alpha))
    hi = float(np.percentile(stats, 100.0 - alpha))
    stat = float(stat_fn(vals))
    return stat, lo, hi


def plot_top1_rmsd_methods(
    gnina_frames: GninaFrames,
    *,
    out_dir: Path,
    preview_png: bool,
) -> None:
    """Top-1 ligand RMSD distributions across methods."""
    _apply_theme_once()
    df = gnina_frames.per_target
    c_boltz, c_vina, c_gnn, c_gnn_nc = _palette4()

    lig = [
        _finite(df["boltz_top1_ligand_rmsd"]),
        _finite(df["gnina_cnn_top1_ligand_rmsd"]),
        _finite(df["gnina_nocnn_top1_ligand_rmsd"]),
        _finite(df["vina_top1_ligand_rmsd"]),
    ]
    labels = [
        f"Boltz-2 (N={lig[0].size})",
        f"GNINA CNN (N={lig[1].size})",
        f"GNINA no-CNN (N={lig[2].size})",
        f"Vina (N={lig[3].size})",
    ]
    colors = [c_boltz, c_gnn, c_gnn_nc, c_vina]

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.4))
    _boxplot(
        ax,
        lig,
        labels,
        colors,
        title="Top-1 Ligand RMSD",
        ylabel=r"RMSD ($\mathrm{\AA}$)",
    )
    _save_figure(fig, out_dir, stem="fig_top1_rmsd_methods", preview_png=preview_png)
    plt.close(fig)


def plot_sampling_vs_ranking_gnina(
    gnina_frames: GninaFrames,
    *,
    out_dir: Path,
    preview_png: bool,
) -> None:
    """Show sampling (best-of-20) and ranking gap distributions."""
    _apply_theme_once()
    df = gnina_frames.per_target
    _, c_vina, c_gnn, c_gnn_nc = _palette4()

    bestk = [
        _finite(df["gnina_cnn_bestK_ligand_rmsd"]),
        _finite(df["gnina_nocnn_bestK_ligand_rmsd"]),
        _finite(df["vina_bestK_ligand_rmsd"]),
    ]
    gap = [
        _finite(df["gnina_cnn_gap_ligand_rmsd"]),
        _finite(df["gnina_nocnn_gap_ligand_rmsd"]),
        _finite(df["vina_gap_ligand_rmsd"]),
    ]
    labels = [
        f"GNINA CNN (N={bestk[0].size})",
        f"GNINA no-CNN (N={bestk[1].size})",
        f"Vina (N={bestk[2].size})",
    ]
    colors = [c_gnn, c_gnn_nc, c_vina]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    _boxplot(
        axes[0],
        bestk,
        labels,
        colors,
        title="Sampling: Best-of-20 Ligand RMSD",
        ylabel=r"RMSD ($\mathrm{\AA}$)",
    )
    axes[0].set_ylim(bottom=0.0)
    _boxplot(
        axes[1],
        gap,
        labels,
        colors,
        title="Ranking Gap (Top-1 − Best-of-20)",
        ylabel=r"$\Delta$RMSD ($\mathrm{\AA}$)",
    )
    _save_figure(fig, out_dir, stem="fig_sampling_vs_ranking", preview_png=preview_png)
    plt.close(fig)


def plot_per_target_comparison_gnina(
    gnina_frames: GninaFrames,
    *,
    out_dir: Path,
    preview_png: bool,
) -> None:
    """Per-target scatter comparisons: Boltz vs Vina and GNINA vs Vina (top-1)."""
    _apply_theme_once()
    df = gnina_frames.per_target
    c_boltz, c_vina, c_gnn, _ = _palette4()

    x_b = pd.to_numeric(df["boltz_top1_ligand_rmsd"], errors="coerce").to_numpy(float)
    y_v = pd.to_numeric(df["vina_top1_ligand_rmsd"], errors="coerce").to_numpy(float)
    y_g = pd.to_numeric(df["gnina_cnn_top1_ligand_rmsd"], errors="coerce").to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    fixed_max = 12.5
    for ax, x, y, title, color, xlabel, ylabel in [
        (axes[0], x_b, y_v, "Boltz-2 vs Vina", c_vina, "Boltz-2 RMSD (Å)", "Vina RMSD (Å)"),
        (axes[1], x_b, y_g, "Boltz-2 vs GNINA CNN", c_gnn, "Boltz-2 RMSD (Å)", "GNINA CNN RMSD (Å)"),
    ]:
        mask = np.isfinite(x) & np.isfinite(y)
        xx = x[mask]
        yy = y[mask]
        if xx.size == 0:
            continue
        max_val = fixed_max
        ax.scatter(xx, yy, s=26, alpha=0.7, color=color, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], ls="--", lw=1.2, color="#666666", alpha=0.8)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.set_title(title, fontsize=12, fontweight="medium", pad=8)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    _save_figure(fig, out_dir, stem="fig_per_target_comparison_gnina", preview_png=preview_png)
    plt.close(fig)


def plot_contact_overlap_methods(
    gnina_frames: GninaFrames,
    *,
    out_dir: Path,
    preview_png: bool,
) -> None:
    """Headgroup contact overlap distributions across methods (top-1)."""
    _apply_theme_once()
    df = gnina_frames.per_target
    c_boltz, c_vina, c_gnn, c_gnn_nc = _palette4()

    env = [
        _finite(df["boltz_head_env_jaccard"]),
        _finite(df["gnina_cnn_head_env_jaccard"]),
        _finite(df["gnina_nocnn_head_env_jaccard"]),
        _finite(df["vina_head_env_jaccard"]),
    ]
    typed = [
        _finite(df["boltz_headgroup_typed_jaccard"]),
        _finite(df["gnina_cnn_headgroup_typed_jaccard"]),
        _finite(df["gnina_nocnn_headgroup_typed_jaccard"]),
        _finite(df["vina_headgroup_typed_jaccard"]),
    ]
    labels = [
        f"Boltz-2 (N={env[0].size})",
        f"GNINA CNN (N={env[1].size})",
        f"GNINA no-CNN (N={env[2].size})",
        f"Vina (N={env[3].size})",
    ]
    colors = [c_boltz, c_gnn, c_gnn_nc, c_vina]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    _boxplot(
        axes[0],
        env,
        labels,
        colors,
        title="Headgroup Environment Overlap (Jaccard)",
        ylabel="Jaccard",
        ylim=(0.0, 1.0),
        whis=(5, 95),
    )
    ax = axes[1]
    thresholds = [0.5, 0.75]
    offsets = [-0.18, 0.18]
    bar_width = 0.32
    xs = np.arange(1, len(labels) + 1)
    for thr, offset in zip(thresholds, offsets):
        rates = []
        for vals in typed:
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            k = int((vals >= thr).sum())
            rate = float(k / n) if n else float("nan")
            rates.append(rate)
        ax.bar(
            xs + offset,
            rates,
            width=bar_width,
            color=colors,
            alpha=0.75 if thr == 0.5 else 0.45,
            edgecolor="none",
            label=f"Jaccard ≥ {thr:.2f}",
        )
    ax.set_title("Typed Interaction Success Rate", fontsize=12, fontweight="medium", pad=8)
    ax.set_ylabel("Success rate", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save_figure(fig, out_dir, stem="fig_contact_overlap_methods", preview_png=preview_png)
    plt.close(fig)


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


def _plot_overlap_distribution_boltz_vs_vina_topk(
    ax: plt.Axes,
    *,
    xb: np.ndarray,
    xv_top1: np.ndarray,
    xv_top20: np.ndarray,
    title: str,
    xmin: float,
    xmax: float,
    ymax: float = 2.0,
) -> None:
    c_boltz, c_top1, _, c_top20 = _palette4()

    xb = xb[(xb >= xmin) & (xb <= xmax)]
    xv_top1 = xv_top1[(xv_top1 >= xmin) & (xv_top1 <= xmax)]
    xv_top20 = xv_top20[(xv_top20 >= xmin) & (xv_top20 <= xmax)]

    labels = {
        "Boltz": "Boltz",
        "top1": _vina_topk_best_label(1),
        "top20": _vina_topk_best_label(20),
    }

    # Plot broader set first (background), then tighter set, then Boltz on top.
    plot_order = [
        (labels["top20"], xv_top20, c_top20, 0.10),
        (labels["top1"], xv_top1, c_top1, 0.14),
        (labels["Boltz"], xb, c_boltz, 0.20),
    ]

    for label, x, color, fill_alpha in plot_order:
        gx, gy = _kde_xy(x, xmin=xmin, xmax=xmax, bw_adjust=0.75)
        if not gx.size:
            continue
        ax.fill_between(gx, 0.0, gy, color=color, alpha=fill_alpha, lw=0.0, zorder=1)
        ax.plot(gx, gy, color=color, lw=2.5, label=f"{label} (N={int(x.size)})", zorder=2)
        if x.size:
            med = float(np.median(x))
            if xmin <= med <= xmax:
                ax.axvline(med, color=color, lw=1.6, ls="--", alpha=0.9, zorder=3)

    ax.set_title(title, fontsize=13, fontweight="medium", pad=10)
    ax.set_xlabel("Jaccard Overlap", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, float(ymax))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)


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

    # Pre-compute sample sizes (use ligand_rmsd as canonical; counts should be same for both metrics)
    xb_full = _finite(frames.boltz["ligand_rmsd"])
    n_boltz = len(xb_full)
    n_vina_top1 = len(_vina_topk_per_target(vina_pose, metric_col="ligand_rmsd", k=1, prefer="min"))
    n_vina_top5 = len(_vina_topk_per_target(vina_pose, metric_col="ligand_rmsd", k=5, prefer="min"))
    n_vina_top20 = len(_vina_topk_per_target(vina_pose, metric_col="ligand_rmsd", k=20, prefer="min"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (col, title) in zip(axes, metrics):
        xb = _finite(frames.boltz[col])
        # For ligand_rmsd: select by ligand RMSD (standard behavior)
        # For other metrics: select pose by ligand RMSD, report that pose's value
        if col == "ligand_rmsd":
            xv_top1 = _vina_topk_per_target(vina_pose, metric_col=col, k=1, prefer="min")
            xv_top5 = _vina_topk_per_target(vina_pose, metric_col=col, k=5, prefer="min")
            xv_top20 = _vina_topk_per_target(vina_pose, metric_col=col, k=20, prefer="min")
        else:
            # Select by ligand RMSD, return value for this metric
            xv_top1 = _finite(_vina_topk_by_ligand_rmsd(vina_pose, return_col=col, k=1))
            xv_top5 = _finite(_vina_topk_by_ligand_rmsd(vina_pose, return_col=col, k=5))
            xv_top20 = _finite(_vina_topk_by_ligand_rmsd(vina_pose, return_col=col, k=20))

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

    # Build legend with sample sizes
    legend_handles = [Line2D([0], [0], color=colors["Boltz"], lw=2.5, label=f"Boltz (N={n_boltz})")]
    if 1 in vina_ks:
        legend_handles.append(Line2D([0], [0], color=colors[labels[1]], lw=2.5, label=f"{labels[1]} (N={n_vina_top1})"))
    if 5 in vina_ks:
        legend_handles.append(Line2D([0], [0], color=colors[labels[5]], lw=2.5, label=f"{labels[5]} (N={n_vina_top5})"))
    if 20 in vina_ks:
        legend_handles.append(Line2D([0], [0], color=colors[labels[20]], lw=2.5, label=f"{labels[20]} (N={n_vina_top20})"))
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


def plot_ligand_rmsd_distribution_top1(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    rmsd_cap_a: float = 10.0,
    preview_png: bool = False,
) -> None:
    """
    Plot a single-panel ligand RMSD distribution: Boltz vs Vina top-1 best.

    This is a reduced variant of `plot_rmsd_distributions` intended for manuscript layouts
    where only ligand RMSD is shown (no headgroup panel).
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()

    # Counts (pre-cap; consistent with the standard distributions plot)
    n_boltz = len(_finite(frames.boltz["ligand_rmsd"]))
    n_vina_top1 = len(_vina_topk_per_target(vina_pose, metric_col="ligand_rmsd", k=1, prefer="min"))

    xb = _finite(frames.boltz["ligand_rmsd"])
    xv_top1 = _vina_topk_per_target(vina_pose, metric_col="ligand_rmsd", k=1, prefer="min")

    xmin = 0.0
    xmax = float(rmsd_cap_a)
    xb = xb[(xb >= xmin) & (xb <= xmax)]
    xv_top1 = xv_top1[(xv_top1 >= xmin) & (xv_top1 <= xmax)]

    c_boltz, c_top1, _, _ = _palette4()
    colors = {"Boltz": c_boltz, _vina_topk_best_label(1): c_top1}

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.0))

    # Plot Vina first (background), then Boltz on top.
    plot_order = [(_vina_topk_best_label(1), xv_top1, colors[_vina_topk_best_label(1)]), ("Boltz", xb, colors["Boltz"])]
    for label, x, color in plot_order:
        gx, gy = _kde_xy(x, xmin=xmin, xmax=xmax, bw_adjust=0.75)
        if not gx.size:
            continue
        lw = 2.5
        fill_alpha = 0.16 if label != "Boltz" else 0.20
        ax.fill_between(gx, 0.0, gy, color=color, alpha=fill_alpha, lw=0.0)
        ax.plot(gx, gy, color=color, lw=lw, label=label)

        if x.size:
            med = float(np.median(x))
            if xmin <= med <= xmax:
                ax.axvline(med, color=color, lw=1.6, ls="--", alpha=0.9, zorder=1)

    ax.set_title("Ligand RMSD", fontsize=13, fontweight="medium", pad=10)
    ax.set_xlabel(r"RMSD ($\mathrm{\AA}$)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)

    legend_handles = [
        Line2D([0], [0], color=colors["Boltz"], lw=2.5, label=f"Boltz (N={n_boltz})"),
        Line2D([0], [0], color=colors[_vina_topk_best_label(1)], lw=2.5, label=f"{_vina_topk_best_label(1)} (N={n_vina_top1})"),
    ]
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

    _save_figure(fig, out_dir, stem="fig_ligand_rmsd_distribution_top1", preview_png=preview_png)
    plt.close(fig)


def plot_ligand_rmsd_distribution_all_vina_poses(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    rmsd_cap_a: float = 10.0,
    preview_png: bool = False,
) -> None:
    """
    Plot a single-panel ligand RMSD distribution: Boltz vs all Vina poses.

    Unlike top-K summaries, this includes every Vina pose row in `benchmark_allposes.csv`
    (i.e., multiple points per target).
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()

    # Counts (pre-cap; consistent with the standard distributions plot)
    n_boltz = len(_finite(frames.boltz["ligand_rmsd"]))
    n_vina_all = len(_finite(vina_pose["ligand_rmsd"])) if "ligand_rmsd" in vina_pose.columns else 0

    xb = _finite(frames.boltz["ligand_rmsd"])
    xv_all = _finite(vina_pose["ligand_rmsd"]) if "ligand_rmsd" in vina_pose.columns else np.array([], dtype=float)

    xmin = 0.0
    xmax = float(rmsd_cap_a)
    xb = xb[(xb >= xmin) & (xb <= xmax)]
    xv_all = xv_all[(xv_all >= xmin) & (xv_all <= xmax)]

    c_boltz, c_top1, _, _ = _palette4()
    colors = {"Boltz": c_boltz, "Vina (all poses)": c_top1}

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.0))

    # Plot Vina first (background), then Boltz on top.
    plot_order = [("Vina (all poses)", xv_all, colors["Vina (all poses)"]), ("Boltz", xb, colors["Boltz"])]
    for label, x, color in plot_order:
        gx, gy = _kde_xy(x, xmin=xmin, xmax=xmax, bw_adjust=0.75)
        if not gx.size:
            continue
        lw = 2.5
        fill_alpha = 0.16 if label != "Boltz" else 0.20
        ax.fill_between(gx, 0.0, gy, color=color, alpha=fill_alpha, lw=0.0)
        ax.plot(gx, gy, color=color, lw=lw, label=label)

        if x.size:
            med = float(np.median(x))
            if xmin <= med <= xmax:
                ax.axvline(med, color=color, lw=1.6, ls="--", alpha=0.9, zorder=1)

    ax.set_title("Ligand RMSD", fontsize=13, fontweight="medium", pad=10)
    ax.set_xlabel(r"RMSD ($\mathrm{\AA}$)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)

    legend_handles = [
        Line2D([0], [0], color=colors["Boltz"], lw=2.5, label=f"Boltz (N={n_boltz})"),
        Line2D([0], [0], color=colors["Vina (all poses)"], lw=2.5, label=f"Vina (all poses) (N={n_vina_all})"),
    ]
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

    _save_figure(fig, out_dir, stem="fig_ligand_rmsd_distribution_all_vina_poses", preview_png=preview_png)
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
    pdbids = x.index.to_numpy()
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
        ax.set_title(f"{_vina_topk_best_label(k)} (N={int(ok.sum())})", fontsize=12, fontweight="medium", pad=8)
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


def plot_paired_rmsd_vina_top1_vs_top20(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    preview_png: bool = False,
) -> None:
    """
    Reduced 1×2 variant of `plot_paired_rmsd` showing only Vina top-1 best and top-20 best.
    """
    _apply_theme_once()

    boltz_df = frames.boltz[["pdbid", "ligand_rmsd"]].copy()
    boltz_df["ligand_rmsd"] = pd.to_numeric(boltz_df["ligand_rmsd"], errors="coerce")
    boltz_df = boltz_df.dropna(subset=["ligand_rmsd"]).sort_values("pdbid").reset_index(drop=True)

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    vina_pose["pose_index"] = pd.to_numeric(vina_pose["pose_index"], errors="coerce")
    vina_pose["ligand_rmsd"] = pd.to_numeric(vina_pose["ligand_rmsd"], errors="coerce")
    vina_pose = vina_pose.dropna(subset=["pose_index", "ligand_rmsd"])

    max_pose_index = int(vina_pose["pose_index"].max()) if vina_pose["pose_index"].notna().any() else 0
    k_large = 20 if max_pose_index >= 20 else max_pose_index
    if k_large <= 1:
        return
    ks = [1, int(k_large)]

    vina_topk: dict[int, pd.Series] = {}
    for k in ks:
        sub = vina_pose[vina_pose["pose_index"] <= k]
        vina_topk[k] = sub.groupby("pdbid")["ligand_rmsd"].min()

    x = boltz_df.set_index("pdbid")["ligand_rmsd"]
    paired: dict[int, pd.Series] = {}
    for k in ks:
        paired[k] = vina_topk[k].reindex(x.index)

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

    fig = plt.figure(figsize=(9.5, 4.8))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 0.06],
        left=0.08,
        right=0.92,
        bottom=0.14,
        top=0.90,
        wspace=0.25,
        hspace=0.0,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

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
        ax.set_title(f"{_vina_topk_best_label(k)} (N={int(ok.sum())})", fontsize=12, fontweight="medium", pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    axes[0].set_ylabel(r"Vina top-K best RMSD ($\mathrm{\AA}$)", fontsize=11)
    for ax in axes:
        ax.set_xlabel(r"Boltz RMSD ($\mathrm{\AA}$)", fontsize=11)

    if sc is not None:
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(r"$\Delta$RMSD (Vina $-$ Boltz)", fontsize=11, labelpad=6)
        cb.ax.tick_params(labelsize=9)
        cb.outline.set_linewidth(0.5)

    fig.suptitle("Per-target RMSD Comparison (Vina top-K best)", fontsize=13, fontweight="medium", y=0.98)
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")

    _save_figure(fig, out_dir, stem="fig_paired_ligand_rmsd_topk_vina_top1_top20", preview_png=preview_png, use_tight_layout=False)
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

    Empty-set cases (where both ref and pred sets have size <= 1 for typed interactions)
    are filtered out to avoid misleading Jaccard=1.0 artifacts.
    """
    _apply_theme_once()

    vina_df = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    boltz_df = allposes_df[allposes_df["method"] == "boltz"].copy()

    # RMSD columns
    lig_rmsd_vina = pd.to_numeric(vina_df["ligand_rmsd"], errors="coerce")
    lig_rmsd_boltz = pd.to_numeric(boltz_df["ligand_rmsd"], errors="coerce")
    head_rmsd_vina = pd.to_numeric(vina_df.get("headgroup_rmsd"), errors="coerce")
    head_rmsd_boltz = pd.to_numeric(boltz_df.get("headgroup_rmsd"), errors="coerce")

    # Overlap columns
    y_env_vina = pd.to_numeric(vina_df["head_env_jaccard"], errors="coerce")
    y_env_boltz = pd.to_numeric(boltz_df["head_env_jaccard"], errors="coerce")
    y_typed_vina = pd.to_numeric(vina_df["headgroup_typed_jaccard"], errors="coerce")
    y_typed_boltz = pd.to_numeric(boltz_df["headgroup_typed_jaccard"], errors="coerce")

    # Size columns for filtering trivial cases
    env_ref_size_vina = pd.to_numeric(vina_df.get("head_env_ref_size", pd.Series(dtype=float)), errors="coerce")
    env_pred_size_vina = pd.to_numeric(vina_df.get("head_env_pred_size", pd.Series(dtype=float)), errors="coerce")
    env_ref_size_boltz = pd.to_numeric(boltz_df.get("head_env_ref_size", pd.Series(dtype=float)), errors="coerce")
    env_pred_size_boltz = pd.to_numeric(boltz_df.get("head_env_pred_size", pd.Series(dtype=float)), errors="coerce")

    typed_ref_size_vina = pd.to_numeric(vina_df.get("headgroup_typed_ref_size", pd.Series(dtype=float)), errors="coerce")
    typed_pred_size_vina = pd.to_numeric(vina_df.get("headgroup_typed_pred_size", pd.Series(dtype=float)), errors="coerce")
    typed_ref_size_boltz = pd.to_numeric(boltz_df.get("headgroup_typed_ref_size", pd.Series(dtype=float)), errors="coerce")
    typed_pred_size_boltz = pd.to_numeric(boltz_df.get("headgroup_typed_pred_size", pd.Series(dtype=float)), errors="coerce")

    # For typed interactions, filter cases where both ref and pred are trivially small (<=1)
    # This avoids misleading Jaccard=1.0 when e.g. both have exactly 1 matching H-bond by chance
    min_typed_size = 2
    typed_nontrivial_vina = (typed_ref_size_vina >= min_typed_size) | (typed_pred_size_vina >= min_typed_size)
    typed_nontrivial_boltz = (typed_ref_size_boltz >= min_typed_size) | (typed_pred_size_boltz >= min_typed_size)

    # 2x2 grid: rows = overlap type (env, typed), cols = RMSD type (ligand, headgroup)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    plot_configs = [
        # (row, col, x_vina, x_boltz, y_vina, y_boltz, extra_mask_vina, extra_mask_boltz, title, xlabel)
        (0, 0, lig_rmsd_vina, lig_rmsd_boltz, y_env_vina, y_env_boltz, None, None,
         "Headgroup Environment Overlap", r"Ligand RMSD ($\mathrm{\AA}$)"),
        (0, 1, head_rmsd_vina, head_rmsd_boltz, y_env_vina, y_env_boltz, None, None,
         "Headgroup Environment Overlap", r"Headgroup RMSD ($\mathrm{\AA}$)"),
        (1, 0, lig_rmsd_vina, lig_rmsd_boltz, y_typed_vina, y_typed_boltz, typed_nontrivial_vina, typed_nontrivial_boltz,
         "Typed Interaction Overlap", r"Ligand RMSD ($\mathrm{\AA}$)"),
        (1, 1, head_rmsd_vina, head_rmsd_boltz, y_typed_vina, y_typed_boltz, typed_nontrivial_vina, typed_nontrivial_boltz,
         "Typed Interaction Overlap", r"Headgroup RMSD ($\mathrm{\AA}$)"),
    ]

    for row, col, x_vina, x_boltz, y_vina, y_boltz, extra_vina, extra_boltz, title, xlabel in plot_configs:
        ax = axes[row, col]

        # Build masks
        vina_mask = np.isfinite(x_vina.to_numpy(dtype=float)) & np.isfinite(y_vina.to_numpy(dtype=float))
        boltz_mask = np.isfinite(x_boltz.to_numpy(dtype=float)) & np.isfinite(y_boltz.to_numpy(dtype=float))

        if extra_vina is not None:
            vina_mask = vina_mask & extra_vina.to_numpy(dtype=bool)
        if extra_boltz is not None:
            boltz_mask = boltz_mask & extra_boltz.to_numpy(dtype=bool)

        xx_vina = x_vina.to_numpy(dtype=float)[vina_mask]
        yy_vina = y_vina.to_numpy(dtype=float)[vina_mask]
        xx_boltz = x_boltz.to_numpy(dtype=float)[boltz_mask]
        yy_boltz = y_boltz.to_numpy(dtype=float)[boltz_mask]

        xx_all = np.concatenate([xx_vina, xx_boltz]) if (xx_vina.size or xx_boltz.size) else np.array([])
        yy_all = np.concatenate([yy_vina, yy_boltz]) if (yy_vina.size or yy_boltz.size) else np.array([])

        n_poses = len(xx_all)

        if xx_all.size:
            xcap = float(np.percentile(xx_all, 99.0))
            ax.set_xlim(0.0, max(xcap, 1.0))
            cmap = "YlGnBu"
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

        # Bin RMSD and plot median + IQR as a robust trend line
        if xx_all.size:
            xmax = float(np.percentile(xx_all, 99))
            bins = np.linspace(0.0, max(xmax, 1.0), 10)
            centers, meds, q25s, q75s = _median_iqr_trend(xx_all, yy_all, bins=bins, min_n=10)
            ok = np.isfinite(meds)

            trend_color = "#C41E3A"
            ax.fill_between(centers[ok], q25s[ok], q75s[ok], color=trend_color, alpha=0.15, zorder=4)
            ax.plot(centers[ok], meds[ok], color=trend_color, lw=2.5, zorder=5)

        ax.set_title(f"{title} (N={n_poses})", fontsize=12, fontweight="medium", pad=10)
        ax.set_xlabel(xlabel, fontsize=11)
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


def plot_contact_overlap_distributions_side_by_side(
    frames: SummaryFrames,
    allposes_df: pd.DataFrame,
    *,
    out_dir: Path,
    preview_png: bool = False,
) -> None:
    """
    Plot headgroup environment overlap and typed interaction overlap side-by-side.

    This figure combines the two overlap metrics into a single 1×2 panel for easier
    visual comparison.
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    if vina_pose.empty:
        return

    panels = [
        ("head_env_jaccard", "Headgroup Environment Overlap"),
        ("headgroup_typed_jaccard", "Headgroup Typed Interaction Overlap"),
    ]

    # 1x2: Boltz vs Vina top-K best (K=1,20), selecting by ligand RMSD.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    xmin, xmax = 0.0, 1.0
    for ax, (metric_col, panel_title) in zip(axes, panels):
        if metric_col not in vina_pose.columns:
            continue

        boltz_df = frames.boltz
        vina_df_for_topk = vina_pose

        if metric_col == "headgroup_typed_jaccard":
            ref_vina = pd.to_numeric(vina_pose.get("headgroup_typed_ref_size"), errors="coerce")
            pred_vina = pd.to_numeric(vina_pose.get("headgroup_typed_pred_size"), errors="coerce")
            if not ref_vina.empty and not pred_vina.empty:
                vina_df_for_topk = vina_df_for_topk[(ref_vina >= 2) | (pred_vina >= 2)]

            if "headgroup_typed_ref_size" in boltz_df.columns and "headgroup_typed_pred_size" in boltz_df.columns:
                ref_b = pd.to_numeric(boltz_df["headgroup_typed_ref_size"], errors="coerce")
                pred_b = pd.to_numeric(boltz_df["headgroup_typed_pred_size"], errors="coerce")
                boltz_df = boltz_df[(ref_b >= 2) | (pred_b >= 2)]

        xb = _finite(boltz_df[metric_col]) if metric_col in boltz_df.columns else np.array([], dtype=float)
        vina_df_for_topk = vina_df_for_topk.copy()
        vina_df_for_topk["pose_index"] = pd.to_numeric(vina_df_for_topk.get("pose_index"), errors="coerce")
        xv_top1 = _finite(_vina_topk_by_ligand_rmsd(vina_df_for_topk, return_col=metric_col, k=1))
        xv_top20 = _finite(_vina_topk_by_ligand_rmsd(vina_df_for_topk, return_col=metric_col, k=20))
        _plot_overlap_distribution_boltz_vs_vina_topk(
            ax,
            xb=xb,
            xv_top1=xv_top1,
            xv_top20=xv_top20,
            title=panel_title,
            xmin=xmin,
            xmax=xmax,
        )
        ax.legend(
            frameon=True,
            facecolor="white",
            framealpha=0.85,
            edgecolor="none",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            borderaxespad=0.0,
            ncol=1,
            handlelength=2.2,
            fontsize=8,
        )

    stem = "fig_contact_overlap_distributions"
    _save_figure(fig, out_dir, stem=stem, preview_png=preview_png)
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
    n_targets = None
    for ax, (col, title) in zip(axes, metrics):
        # Boltz baseline: one value per target from the summary.
        boltz = pd.to_numeric(frames.boltz.set_index("pdbid")[col], errors="coerce").dropna()
        n_targets = len(boltz)

        for thr in thresholds:
            # Vina top-K best per target: select by ligand RMSD, report this metric.
            rates: list[float] = []
            for k in ks:
                if col == "ligand_rmsd":
                    s = _vina_topk_series_per_target(vina_pose, metric_col=col, k=k, prefer="min")
                else:
                    s = _vina_topk_by_ligand_rmsd(vina_pose, return_col=col, k=k)
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

        ax.set_title(f"{title} (N={n_targets})", fontsize=13, fontweight="medium", pad=10)
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

    Shows Boltz vs Vina top-1/top-5/top-20 best (best-within-top-K per target).
    """
    _apply_theme_once()

    vina_pose = allposes_df[allposes_df["method"] == "vina_pose"].copy()
    if vina_pose.empty:
        return
    vina_pose["pose_index"] = pd.to_numeric(vina_pose.get("pose_index"), errors="coerce")
    max_pose_index = int(vina_pose["pose_index"].max()) if vina_pose["pose_index"].notna().any() else 0

    c_boltz, c1, c5, c20 = _palette4()
    colors = {
        "Boltz": c_boltz,
        _vina_topk_best_label(1): c1,
        _vina_topk_best_label(5): c5,
        _vina_topk_best_label(20): c20,
    }

    metrics = [
        ("ligand_rmsd", "Ligand RMSD"),
        ("headgroup_rmsd", "Headgroup RMSD"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, title) in zip(axes, metrics):
        boltz = _finite(frames.boltz[col])
        # For ligand_rmsd: select by ligand RMSD
        # For other metrics: select pose by ligand RMSD, report that pose's value
        if col == "ligand_rmsd":
            vina1 = _finite(_vina_topk_series_per_target(vina_pose, metric_col=col, k=1, prefer="min"))
            vina5 = _finite(_vina_topk_series_per_target(vina_pose, metric_col=col, k=5, prefer="min"))
            vina20 = _finite(_vina_topk_series_per_target(vina_pose, metric_col=col, k=20, prefer="min"))
        else:
            vina1 = _finite(_vina_topk_by_ligand_rmsd(vina_pose, return_col=col, k=1))
            vina5 = _finite(_vina_topk_by_ligand_rmsd(vina_pose, return_col=col, k=5))
            vina20 = _finite(_vina_topk_by_ligand_rmsd(vina_pose, return_col=col, k=20))

        # Build list of curves to plot (only include if we have enough poses)
        curves = [("Boltz", boltz, len(boltz))]
        curves.append((_vina_topk_best_label(1), vina1, len(vina1)))
        if max_pose_index >= 5:
            curves.append((_vina_topk_best_label(5), vina5, len(vina5)))
        if max_pose_index >= 20:
            curves.append((_vina_topk_best_label(20), vina20, len(vina20)))

        for label, values, n in curves:
            x, y = _ecdf(values)
            if x.size:
                ax.plot(x, y, color=colors[label], lw=2.5, label=f"{label} (N={n})")

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
        fontsize=9,
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

    Plots all individual poses as a scatter plot with a linear regression trendline,
    showing the slope and p-value to quantify the correlation between rank and RMSD.
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
    if max_pose_index <= 0:
        return

    c_boltz, c1, _, _ = _palette4()
    scatter_color = "#4A90A4"  # Muted teal for scatter points
    trend_color = "#C41E3A"  # Cardinal red for trendline

    metrics = [
        ("ligand_rmsd", "Ligand RMSD"),
        ("headgroup_rmsd", "Headgroup RMSD"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (col, title) in zip(axes, metrics):
        # Extract all (rank, RMSD) pairs
        x_col = vina["pose_index"].to_numpy(dtype=float)
        y_col = pd.to_numeric(vina[col], errors="coerce").to_numpy(dtype=float)

        # Filter to finite values and ranks <= 20
        mask = np.isfinite(x_col) & np.isfinite(y_col) & (x_col <= 20)
        xx = x_col[mask]
        yy = y_col[mask]

        if len(xx) < 3:
            continue

        n_poses = len(xx)

        # Scatter plot with transparency for overlapping points
        ax.scatter(
            xx, yy,
            c=scatter_color,
            s=20,
            alpha=0.4,
            edgecolors="none",
            zorder=2,
        )

        # Linear regression
        result = linregress(xx, yy)
        slope = result.slope
        intercept = result.intercept
        pvalue = result.pvalue
        rvalue = result.rvalue

        # Plot trendline
        x_line = np.array([xx.min(), xx.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=trend_color, lw=2.5, ls="-", zorder=3)

        # Format p-value for display
        if pvalue < 0.001:
            p_str = "p < 0.001"
        elif pvalue < 0.01:
            p_str = f"p = {pvalue:.3f}"
        else:
            p_str = f"p = {pvalue:.2f}"

        # Add regression stats as text annotation
        stats_text = f"slope = {slope:.3f} $\\mathrm{{\\AA}}$/rank\n{p_str}\n$r$ = {rvalue:.2f}"
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="none", alpha=0.9),
        )

        ax.set_title(f"{title} vs Vina rank (N={n_poses})", fontsize=13, fontweight="medium", pad=10)
        ax.set_xlabel("Vina rank (pose_index)", fontsize=11)
        ax.set_ylabel(r"RMSD ($\mathrm{\AA}$)", fontsize=11)
        ax.set_xlim(0.5, min(20, max_pose_index) + 0.5)
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
    p.add_argument("--summary", default="output/benchmark/benchmark_summary.csv", help="Path to summary CSV.")
    p.add_argument("--allposes", default="output/benchmark/benchmark_allposes.csv", help="Path to allposes CSV.")
    p.add_argument(
        "--gnina-analysis-dir",
        default=None,
        help="Path to GNINA analysis directory (containing per_target.csv). If set, GNINA figures are generated.",
    )
    p.add_argument("--out-dir", default="plots", help="Output directory for figures.")
    p.add_argument(
        "--figset",
        choices=("standard", "extra", "all"),
        default="standard",
        help="Which figure set to render: standard, extra (manuscript variants), or all.",
    )
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
    gnina_frames = None
    if args.gnina_analysis_dir:
        gnina_dir = _resolve_csv(str(args.gnina_analysis_dir))
        gnina_frames = _load_gnina_frames(gnina_dir, summary_csv=summary_csv)

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

    standard_tasks: list[tuple[str, callable]] = [
        ("RMSD distributions", lambda: plot_rmsd_distributions(frames, allposes_df, out_dir=out_dir, preview_png=preview_png)),
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
            "Contact overlap distributions",
            lambda: plot_contact_overlap_distributions_side_by_side(frames, allposes_df, out_dir=out_dir, preview_png=preview_png),
        ),
    ]

    if gnina_frames is not None:
        standard_tasks.extend([
            ("Top-1 RMSD (Boltz/Vina/GNINA)", lambda: plot_top1_rmsd_methods(gnina_frames, out_dir=out_dir, preview_png=preview_png)),
            ("Sampling vs ranking (GNINA)", lambda: plot_sampling_vs_ranking_gnina(gnina_frames, out_dir=out_dir, preview_png=preview_png)),
            ("Per-target comparisons (GNINA)", lambda: plot_per_target_comparison_gnina(gnina_frames, out_dir=out_dir, preview_png=preview_png)),
            ("Contact overlap (GNINA)", lambda: plot_contact_overlap_methods(gnina_frames, out_dir=out_dir, preview_png=preview_png)),
        ])

    extra_tasks: list[tuple[str, callable]] = [
        (
            "Ligand RMSD distribution (Boltz vs Vina top-1)",
            lambda: plot_ligand_rmsd_distribution_top1(frames, allposes_df, out_dir=out_dir, preview_png=preview_png),
        ),
        (
            "Ligand RMSD distribution (Boltz vs all Vina poses)",
            lambda: plot_ligand_rmsd_distribution_all_vina_poses(
                frames, allposes_df, out_dir=out_dir, preview_png=preview_png
            ),
        ),
        (
            "Paired RMSD (Vina top-1 vs top-20)",
            lambda: plot_paired_rmsd_vina_top1_vs_top20(frames, allposes_df, out_dir=out_dir, preview_png=preview_png),
        ),
    ]

    tasks: list[tuple[str, callable]] = []
    if args.figset in {"standard", "all"}:
        tasks.extend(standard_tasks)
    if args.figset in {"extra", "all"}:
        tasks.extend(extra_tasks)

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
