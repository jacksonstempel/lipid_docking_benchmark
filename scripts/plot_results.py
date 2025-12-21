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
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import textwrap
import shutil
import subprocess
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
except OSError:
    pass

# If we vendored any TeX helper binaries (e.g., dvipng), add them to PATH.
_VENDORED_TEXBIN = _PROJECT_ROOT / "tools" / "texbin" / "usr" / "bin"
if _VENDORED_TEXBIN.is_dir():
    os.environ["PATH"] = f"{_VENDORED_TEXBIN}{os.pathsep}{os.environ.get('PATH','')}"

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
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


_FORCE_USETEX = False
_USETEX_VALIDATED: bool | None = None
_USETEX_FAILURE: str | None = None

@dataclass(frozen=True)
class SummaryFrames:
    """
    Convenience container for the two “matched” summary tables we compare.

    - `boltz`: rows where `method == "boltz"`
    - `vina_best`: rows where `method == "vina_best"`

    These are kept in the same PDBID order so plots can compare the same targets.
    """
    boltz: pd.DataFrame
    vina_best: pd.DataFrame


def _finite(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas column to a clean numeric array.

    - Coerces non-numeric entries (e.g., "NA") to missing values.
    - Drops NaN/Inf values.
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return x


def _kde_xy(x: np.ndarray, *, xmin: float, xmax: float, n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a smooth density curve (KDE) for a 1D distribution.

    Returns `(grid, density)` arrays suitable for plotting, or empty arrays when there
    is not enough data to estimate a density (fewer than 3 points).
    """
    if x.size < 3:
        return np.array([]), np.array([])
    grid = np.linspace(float(xmin), float(xmax), int(n))
    kde = gaussian_kde(x)
    y = kde(grid)
    return grid, y


def _load_frames(summary_csv: Path) -> SummaryFrames:
    """
    Load `benchmark_summary.csv` and split it into Boltz and Vina-best tables.

    We also validate that both tables contain the same set of targets (same PDBIDs),
    so plots compare like-with-like.
    """
    df = pd.read_csv(summary_csv)
    boltz = df[df["method"] == "boltz"].copy()
    vina_best = df[df["method"] == "vina_best"].copy()
    if len(boltz) == 0 or len(vina_best) == 0:
        raise RuntimeError("Missing boltz or vina_best rows in summary CSV.")
    boltz = boltz.sort_values("pdbid").reset_index(drop=True)
    vina_best = vina_best.sort_values("pdbid").reset_index(drop=True)
    if not (boltz["pdbid"].to_numpy() == vina_best["pdbid"].to_numpy()).all():
        raise RuntimeError("Summary CSV boltz/vina_best PDBID sets do not match.")
    return SummaryFrames(boltz=boltz, vina_best=vina_best)


def _apply_pub_style() -> None:
    """
    Set Matplotlib defaults for clean, publication-style figures.

    This adjusts fonts, line widths, tick styles, and PDF settings so the output looks
    consistent across machines. By default we use Matplotlib’s built-in “mathtext”
    Computer Modern look (LaTeX-like, without requiring TeX).
    """
    # We default to Matplotlib's built-in Computer Modern fonts (mathtext) for a LaTeX-like look
    # without requiring a full TeX installation. True usetex can be enabled via --usetex.
    use_tex = _validated_usetex()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            # When usetex is on, prefer the classic LaTeX look (Computer Modern).
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
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
            "text.usetex": use_tex,
            "mathtext.fontset": "cm",
        }
    )
    if use_tex:
        # Keep the preamble minimal for portability.
        plt.rcParams.update(
            {
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            }
        )


def _validated_usetex() -> bool:
    """
    Return whether LaTeX rendering is enabled *and* actually works on this machine.

    Why validate?
    - Enabling `usetex` makes Matplotlib call external TeX binaries (`latex`, `dvipng`).
    - On many systems those binaries are not installed, or are missing dependencies.

    This function performs a small “smoke test” once per run so plotting fails gracefully
    (falling back to mathtext) instead of crashing mid-figure.
    """
    global _USETEX_VALIDATED, _USETEX_FAILURE
    if not _FORCE_USETEX:
        _USETEX_VALIDATED = False
        _USETEX_FAILURE = None
        return False
    if _USETEX_VALIDATED is not None:
        return bool(_USETEX_VALIDATED)

    latex_path = shutil.which("latex")
    dvipng_path = shutil.which("dvipng")
    if not bool(latex_path):
        _USETEX_FAILURE = "missing `latex` executable on PATH"
        _USETEX_VALIDATED = False
        return False
    if not bool(dvipng_path):
        _USETEX_FAILURE = "missing `dvipng` executable on PATH"
        _USETEX_VALIDATED = False
        return False

    test = r"\documentclass{article}\begin{document}lp\end{document}"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "mptest.tex"
            tex_path.write_text(test, encoding="utf-8")
            subprocess.run(
                ["latex", "-interaction=nonstopmode", "--halt-on-error", str(tex_path.name)],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
    except Exception as exc:
        details = ""
        if isinstance(exc, subprocess.CalledProcessError):
            out = str(exc.stdout or "").strip()
            if out:
                tail = "\n".join(out.splitlines()[-8:])
                details = f":\n{tail}"
        _USETEX_FAILURE = (
            f"could not compile a minimal LaTeX document using latex={latex_path}, dvipng={dvipng_path}{details}"
        )
        _USETEX_VALIDATED = False
        return False

    _USETEX_FAILURE = None
    _USETEX_VALIDATED = True
    return True


def _apply_theme() -> None:
    """
    Apply a consistent plotting theme (fonts, styles, and optional scienceplots presets).

    This is called once near the start of `main()` so all figures share a coherent style.
    """
    _apply_pub_style()
    use_tex = bool(plt.rcParams.get("text.usetex", False))
    if scienceplots is not None:
        plt.style.use(["science", "nature"] + ([] if use_tex else ["no-latex"]))
    # Apply our refined settings after any style overrides.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["cmr10", "Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
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


def _palette3() -> tuple[str, str, str]:
    """Return a refined, publication-quality 3-color palette."""
    # Elegant, distinguishable palette with good contrast
    return ("#2E86AB", "#E94F37", "#41B3A3")  # Teal blue, Vermilion, Sea green


def _save(fig: plt.Figure, out_dir: Path, stem: str, *, with_caption: bool = False) -> None:
    """Save a figure to PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _save_preview_png(fig: plt.Figure, out_dir: Path, stem: str, *, with_caption: bool = False) -> None:
    """
    Save a “preview” PNG version of a figure.

    PDFs are preferred for publication, but PNGs are convenient for quick viewing in file
    browsers or chat. These can optionally be pruned after the run.
    """
    save_dir = out_dir / "_preview"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{stem}.png", bbox_inches="tight", dpi=300)


def _prune_non_pdf(out_dir: Path) -> None:
    """Remove preview outputs; PDFs are always preserved."""
    path = out_dir / "_preview"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)

def _add_caption(
    fig: plt.Figure,
    text: str,
    *,
    figure_num: int | None = None,
    caption_top: float = 0.18,
) -> None:
    """
    Add a figure caption beneath the plot area.

    Captions can optionally be numbered (“Figure 1.”). When LaTeX is not available, this
    uses Matplotlib layout primitives to bold only the “Figure N.” prefix.
    """
    text = " ".join(str(text).split())
    fig_width_inches = fig.get_figwidth()
    wrap_width = int(0.96 * fig_width_inches * 13)
    use_tex = bool(plt.rcParams.get("text.usetex", False))

    if figure_num is None:
        wrapped = textwrap.fill(text, width=wrap_width)
        fig.text(
            0.02,
            caption_top,
            wrapped,
            ha="left",
            va="top",
            fontsize=12,
            color="#000000",
            linespacing=1.3,
        )
        return

    prefix_plain = f"Figure {figure_num}. "
    full_plain = f"{prefix_plain}{text}"

    if use_tex:
        prefix_bold = rf"\textbf{{Figure~{figure_num}.}}"
        wrapped = textwrap.fill(f"{prefix_bold} {text}", width=wrap_width)
        fig.text(
            0.02,
            caption_top,
            wrapped,
            ha="left",
            va="top",
            fontsize=12,
            color="#000000",
            linespacing=1.3,
        )
        return

    # Without usetex, build a multi-style caption using offsetboxes so only the
    # "Figure N." prefix is bold while the remaining text stays normal.
    wrapped = textwrap.fill(full_plain, width=wrap_width)
    lines = wrapped.splitlines() if wrapped else [full_plain]
    text_props = {"fontsize": 12, "color": "#000000", "fontfamily": "serif"}
    bold_props = dict(text_props)
    bold_props["fontweight"] = "bold"

    first_line = lines[0] if lines else ""
    if first_line.startswith(prefix_plain):
        first_rest = first_line[len(prefix_plain) :]
        first = HPacker(
            children=[
                TextArea(prefix_plain, textprops=bold_props),
                TextArea(first_rest, textprops=text_props),
            ],
            align="baseline",
            pad=0.0,
            sep=0.0,
        )
    else:
        first = TextArea(first_line, textprops=text_props)

    children = [first] + [TextArea(line, textprops=text_props) for line in lines[1:]]
    vbox = VPacker(children=children, align="left", pad=0.0, sep=2.0)
    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=vbox,
        pad=0.0,
        frameon=False,
        bbox_to_anchor=(0.02, caption_top),
        bbox_transform=fig.transFigure,
        borderpad=0.0,
    )
    fig.add_artist(anchored)


def _layout_with_caption(
    fig: plt.Figure,
    caption: str,
    *,
    bottom: float,
    top: float = 1.0,
    figure_num: int | None = None,
) -> None:
    """
    Reserve space for a caption, then add it below the plot.

    This is separate from `_add_caption()` so we can control layout spacing consistently
    across figures.
    """
    fig.tight_layout(rect=(0.0, bottom, 1.0, top))
    # Position caption just below the plot area
    _add_caption(fig, caption, figure_num=figure_num, caption_top=bottom - 0.02)


def _save_figure(
    fig: plt.Figure,
    out_dir: Path,
    *,
    stem: str,
    preview_png: bool,
    figure_num: int | None,
    caption: str,
    no_caption_rect: tuple[float, float, float, float] | None = None,
    caption_bottom: float | None = None,
    caption_top: float = 1.0,
) -> None:
    """
    Save a figure in the standard output structure.

    What it writes:
    - Always writes a PDF without caption (for clean figure panels).
    - Optionally writes a “with caption” PDF named `figureN.pdf`.
    - Optionally writes PNG previews mirroring the same structure.
    """
    if caption and figure_num is not None and caption_bottom is not None:
        _layout_with_caption(fig, caption, bottom=caption_bottom, top=caption_top, figure_num=figure_num)
        out_stem = f"figure{figure_num}"
    else:
        if no_caption_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=no_caption_rect)
        out_stem = stem

    _save(fig, out_dir, out_stem)
    if preview_png:
        _save_preview_png(fig, out_dir, out_stem)


def plot_rmsd_distributions(
    frames: SummaryFrames,
    allposes_csv: Path,
    *,
    out_dir: Path,
    rmsd_cap_a: float = 10.0,
    preview_png: bool = False,
    figure_num: int | None = None,
    caption: str = "",
) -> None:
    """
    Plot distributions of RMSD values for Boltz vs Vina.

    This figure is meant to answer: “How accurate are the methods overall?” by showing
    smoothed distributions for:
    - ligand RMSD
    - headgroup RMSD

    It also overlays the distribution of all Vina poses (not just the best pose).
    """
    _apply_theme()

    df_all = pd.read_csv(allposes_csv)
    vina_pose = df_all[df_all["method"] == "vina_pose"].copy()

    c_b, c_v, c_all = _palette3()
    colors = {"Boltz": c_b, "Vina best": c_v, "Vina poses": c_all}
    metrics = [
        ("ligand_rmsd", "Ligand RMSD"),
        ("headgroup_rmsd", "Headgroup RMSD"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, title) in zip(axes, metrics):
        xb = _finite(frames.boltz[col])
        xv_best = _finite(frames.vina_best[col])
        xv_all = _finite(vina_pose[col])

        xmin = 0.0
        xmax = float(rmsd_cap_a)

        # Cap to emphasize the bulk; outliers are still present in other plots.
        xb = xb[(xb >= xmin) & (xb <= xmax)]
        xv_best = xv_best[(xv_best >= xmin) & (xv_best <= xmax)]
        xv_all = xv_all[(xv_all >= xmin) & (xv_all <= xmax)]

        # Use SciPy KDE for full control over the support so the filled curves
        # always start at xmin (avoids hard vertical edges at the first sample).
        # Plot Vina poses first (background), then overlay Boltz and Vina best.
        plot_order = [
            ("Vina poses", xv_all, colors["Vina poses"]),
            ("Vina best", xv_best, colors["Vina best"]),
            ("Boltz", xb, colors["Boltz"]),
        ]
        for label, x, color in plot_order:
            gx, gy = _kde_xy(x, xmin=xmin, xmax=xmax)
            if not gx.size:
                continue
            alpha = 0.15 if label == "Vina poses" else 0.25
            lw = 1.8 if label == "Vina poses" else 2.2
            ax.fill_between(gx, 0.0, gy, color=color, alpha=alpha, lw=0.0)
            ax.plot(gx, gy, color=color, lw=lw, label=label)
            if label != "Vina poses" and x.size:
                ax.axvline(float(np.median(x)), color=color, lw=1.2, alpha=0.8, ls="--")

        ax.set_title(title, fontsize=13, fontweight="medium", pad=10)
        ax.set_xlabel(r"RMSD ($\mathrm{\AA}$)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    # Create legend with refined styling
    handles = [
        Line2D([0], [0], color=colors["Boltz"], lw=2.2, label="Boltz"),
        Line2D([0], [0], color=colors["Vina best"], lw=2.2, label="Vina best"),
        Line2D([0], [0], color=colors["Vina poses"], lw=1.8, label="Vina poses"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        columnspacing=2.0,
        handlelength=2.2,
        fontsize=10,
    )

    stem = f"figure{figure_num}" if figure_num else "fig_rmsd_distributions"
    _save_figure(
        fig,
        out_dir,
        stem=stem,
        preview_png=preview_png,
        figure_num=figure_num,
        caption=caption,
        no_caption_rect=(0.0, 0.02, 1.0, 0.88),
        caption_bottom=0.28,
        caption_top=0.88,
    )

    plt.close(fig)


def plot_paired_rmsd(
    frames: SummaryFrames,
    *,
    out_dir: Path,
    preview_png: bool = False,
    figure_num: int | None = None,
    caption: str = "",
) -> None:
    """
    Plot a paired per-target comparison: Boltz RMSD vs Vina-best RMSD.

    Each point is one target. Points are colored by the difference (Vina - Boltz),
    so you can see where one method is better or worse.
    """
    _apply_theme()

    x = _finite(frames.boltz["ligand_rmsd"])
    y = _finite(frames.vina_best["ligand_rmsd"])
    if x.size != y.size:
        raise RuntimeError("Paired plot requires same number of boltz and vina_best RMSDs.")

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    delta = y - x
    if cmocean is not None:
        cmap = cmocean.cm.balance
    else:
        cmap = "RdBu_r"
    vmax = float(np.max(np.abs(delta))) if delta.size else 1.0

    sc = ax.scatter(
        x,
        y,
        c=delta,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        s=50,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    # Refined colorbar
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, aspect=25)
    cb.set_label(r"$\Delta$RMSD (Vina $-$ Boltz)", fontsize=10, labelpad=8)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.5)

    lim = float(max(np.max(x), np.max(y), 1.0)) * 1.05
    ax.plot([0, lim], [0, lim], color="#555555", lw=1.2, ls="--", zorder=2, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(r"Boltz ligand RMSD ($\mathrm{\AA}$)", fontsize=11)
    ax.set_ylabel(r"Vina best ligand RMSD ($\mathrm{\AA}$)", fontsize=11)
    ax.set_title("Per-target RMSD Comparison", fontsize=13, fontweight="medium", pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)

    stem = f"figure{figure_num}" if figure_num else "fig_paired_ligand_rmsd"
    _save_figure(
        fig,
        out_dir,
        stem=stem,
        preview_png=preview_png,
        figure_num=figure_num,
        caption=caption,
        caption_bottom=0.24,
    )

    plt.close(fig)


def plot_contacts_vs_rmsd(
    allposes_csv: Path,
    *,
    out_dir: Path,
    log_counts: bool = False,
    preview_png: bool = False,
    figure_num: int | None = None,
    caption: str = "",
) -> None:
    """
    Plot how contact overlap changes with RMSD across Vina poses.

    Each point is one Vina pose. This helps visualize the relationship between:
    - geometric accuracy (RMSD)
    - interaction accuracy (headgroup contact overlap)
    """
    _apply_theme()

    df = pd.read_csv(allposes_csv)
    df = df[df["method"] == "vina_pose"].copy()

    x = pd.to_numeric(df["ligand_rmsd"], errors="coerce")
    y_env = pd.to_numeric(df["head_env_jaccard"], errors="coerce")
    y_typed = pd.to_numeric(df["headgroup_typed_jaccard"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, y, title in [
        (axes[0], y_env, "Headgroup Environment Overlap"),
        (axes[1], y_typed, "Headgroup Typed Interaction Overlap"),
    ]:
        mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
        xx = x.to_numpy(dtype=float)[mask]
        yy = y.to_numpy(dtype=float)[mask]
        if xx.size:
            xcap = float(np.percentile(xx, 99.0))
            ax.set_xlim(0.0, max(xcap, 1.0))
            # Use a more vibrant, professional colormap with better contrast
            cmap = "YlGnBu"  # Yellow-Green-Blue, good for density plots
            if cmocean is not None:
                cmap = cmocean.cm.dense
            hb = ax.hexbin(
                xx,
                yy,
                gridsize=40,
                mincnt=1,
                cmap=cmap,
                linewidths=0.2,
                edgecolors="face",
                alpha=0.92,
                bins="log" if log_counts else None,
            )
            cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, aspect=25)
            cb.set_label("Pose count" + (" (log)" if log_counts else ""), fontsize=9, labelpad=6)
            cb.ax.tick_params(labelsize=8)
            cb.outline.set_linewidth(0.5)

        # Bin RMSD and plot median + IQR as a robust trend line.
        if xx.size:
            xmax = float(np.percentile(xx, 99))
            bins = np.linspace(0.0, max(xmax, 1.0), 10)
            centers = 0.5 * (bins[:-1] + bins[1:])
            meds = []
            q25s = []
            q75s = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (xx >= lo) & (xx < hi)
                vals = yy[m]
                if vals.size < 10:
                    meds.append(np.nan)
                    q25s.append(np.nan)
                    q75s.append(np.nan)
                    continue
                q25, q50, q75 = np.percentile(vals, [25, 50, 75])
                meds.append(q50)
                q25s.append(q25)
                q75s.append(q75)
            meds = np.array(meds, float)
            q25s = np.array(q25s, float)
            q75s = np.array(q75s, float)
            ok = np.isfinite(meds)
            # Use a bold color for the trend line that stands out
            trend_color = "#C41E3A"  # Cardinal red
            ax.fill_between(
                centers[ok], q25s[ok], q75s[ok], color=trend_color, alpha=0.15, label="IQR", zorder=4
            )
            ax.plot(centers[ok], meds[ok], color=trend_color, lw=2.5, label="Median", zorder=5)

        ax.set_title(title, fontsize=12, fontweight="medium", pad=10)
        ax.set_xlabel(r"Ligand RMSD ($\mathrm{\AA}$)", fontsize=11)
        ax.set_ylabel("Jaccard Overlap", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis="both", which="major", labelsize=9)

    axes[0].legend(frameon=False, loc="upper right", fontsize=9)

    stem = f"figure{figure_num}" if figure_num else "fig_contacts_vs_rmsd_vina_poses"
    _save_figure(
        fig,
        out_dir,
        stem=stem,
        preview_png=preview_png,
        figure_num=figure_num,
        caption=caption,
        caption_bottom=0.26,
    )

    plt.close(fig)


def plot_contact_overlap_distributions(
    frames: SummaryFrames,
    allposes_csv: Path,
    *,
    out_dir: Path,
    metric_col: str,
    title: str,
    stem: str,
    preview_png: bool = False,
    figure_num: int | None = None,
    caption: str = "",
) -> None:
    """
    Plot distributions of contact-overlap scores (Jaccard) for Boltz vs Vina.

    This is the “interaction accuracy” companion to RMSD plots. The `metric_col` selects
    which overlap metric to visualize (e.g., headgroup environment overlap vs typed
    interaction overlap).
    """
    _apply_theme()

    df_all = pd.read_csv(allposes_csv)
    vina_pose = df_all[df_all["method"] == "vina_pose"].copy()

    xb = _finite(frames.boltz[metric_col])
    xv_best = _finite(frames.vina_best[metric_col])
    xv_all = _finite(vina_pose[metric_col])

    xmin, xmax = 0.0, 1.0
    xb = xb[(xb >= xmin) & (xb <= xmax)]
    xv_best = xv_best[(xv_best >= xmin) & (xv_best <= xmax)]
    xv_all = xv_all[(xv_all >= xmin) & (xv_all <= xmax)]

    c_b, c_v, c_all = _palette3()
    colors = {"Boltz": c_b, "Vina best": c_v, "Vina poses": c_all}

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # Plot Vina poses first (background), then overlay Boltz and Vina best.
    plot_order = [
        ("Vina poses", xv_all, colors["Vina poses"]),
        ("Vina best", xv_best, colors["Vina best"]),
        ("Boltz", xb, colors["Boltz"]),
    ]
    for label, x, color in plot_order:
        gx, gy = _kde_xy(x, xmin=xmin, xmax=xmax)
        if not gx.size:
            continue
        alpha = 0.15 if label == "Vina poses" else 0.25
        lw = 1.8 if label == "Vina poses" else 2.2
        ax.fill_between(gx, 0.0, gy, color=color, alpha=alpha, lw=0.0)
        ax.plot(gx, gy, color=color, lw=lw, label=label)
        if label != "Vina poses" and x.size:
            ax.axvline(float(np.median(x)), color=color, lw=1.2, alpha=0.8, ls="--")

    ax.set_title(title, fontsize=13, fontweight="medium", pad=10)
    ax.set_xlabel("Jaccard Overlap", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis="both", which="major", labelsize=9)

    handles = [
        Line2D([0], [0], color=colors["Boltz"], lw=2.2, label="Boltz"),
        Line2D([0], [0], color=colors["Vina best"], lw=2.2, label="Vina best"),
        Line2D([0], [0], color=colors["Vina poses"], lw=1.8, label="Vina poses"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        columnspacing=2.0,
        handlelength=2.2,
        fontsize=10,
    )

    out_stem = f"figure{figure_num}" if figure_num else stem
    _save_figure(
        fig,
        out_dir,
        stem=out_stem,
        preview_png=preview_png,
        figure_num=figure_num,
        caption=caption,
        no_caption_rect=(0.0, 0.02, 1.0, 0.88),
        caption_bottom=0.28,
        caption_top=0.88,
    )

    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    """
    Command-line entry point for plot generation.

    - Reads CSV inputs (summary + all-poses).
    - Generates a standard set of figures with captions.
    - Writes PDFs (and optionally PNG previews) into `--out-dir`.
    """
    global _FORCE_USETEX
    p = argparse.ArgumentParser(description="Generate publication-quality plots from benchmark CSVs.")
    p.add_argument("--summary", default="analysis/benchmark/benchmark_summary.csv", help="Path to summary CSV.")
    p.add_argument("--allposes", default="analysis/benchmark/benchmark_allposes.csv", help="Path to allposes CSV.")
    p.add_argument("--out-dir", default="paper/figures", help="Output directory for figures.")
    p.add_argument("--log-density", action="store_true", help="Use log10 scaling for hexbin pose densities.")
    p.add_argument(
        "--usetex",
        action="store_true",
        help="Render all text via LaTeX (falls back to mathtext if TeX is unavailable).",
    )
    p.add_argument(
        "--texbin",
        default="",
        help="Optional directory to prepend to PATH for TeX binaries (latex, dvipng).",
    )
    p.add_argument("--preview-png", action="store_true", help="Also write PNGs for local preview (then prune).")
    p.add_argument("--keep-preview", action="store_true", help="Keep preview PNGs (no pruning).")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.texbin:
        texbin_path = Path(args.texbin).expanduser().resolve()
        os.environ["PATH"] = f"{texbin_path}{os.pathsep}{os.environ.get('PATH','')}"

    _FORCE_USETEX = bool(args.usetex)
    if _FORCE_USETEX and not _validated_usetex():
        print(
            f"Warning: --usetex requested but {_USETEX_FAILURE}; falling back to mathtext.",
            file=sys.stderr,
        )
        _FORCE_USETEX = False

    summary_csv = Path(args.summary).expanduser().resolve()
    allposes_csv = Path(args.allposes).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not summary_csv.is_file():
        raise FileNotFoundError(summary_csv)
    if not allposes_csv.is_file():
        raise FileNotFoundError(allposes_csv)

    frames = _load_frames(summary_csv)
    preview_png = bool(args.preview_png)

    # Figure captions
    captions = {
        1: "RMSD distributions across benchmark targets. Kernel density estimates show that Boltz and Vina best-scoring poses concentrate at lower RMSD values, while the full Vina pose ensemble exhibits a broad distribution. Dashed lines indicate median values.",
        2: "Per-target RMSD comparison between Boltz and Vina best-scoring poses. Each point represents one target structure. The dashed diagonal line indicates equal performance. Color encodes the RMSD difference (Vina minus Boltz), with blue favoring Vina and red favoring Boltz.",
        3: "Relationship between ligand RMSD and headgroup interaction similarity for all Vina docking poses. Left panel shows distance-based headgroup environment overlap; right panel shows typed interaction overlap from PandaMap. Hexbin density reveals the inverse correlation between structural deviation and interaction fidelity. The red line traces median overlap across RMSD bins with interquartile range shown in shaded regions.",
        4: "Headgroup interaction overlap distributions. Distance-based environment overlap (left) measures residue proximity to the headgroup, while typed interaction overlap (right) requires matching both residue identity and chemical interaction type. Boltz and Vina best-scoring poses show higher overlap than the full Vina ensemble.",
        5: "Typed headgroup interaction overlap using PandaMap's interaction fingerprints. This stringent metric requires both correct residue contacts and proper interaction chemistry. Distributions reveal that Boltz predictions and top-ranked Vina poses better recapitulate native interaction patterns compared to the broader docking ensemble.",
    }

    # Generate all figures with numbering
    plot_rmsd_distributions(
        frames,
        allposes_csv,
        out_dir=out_dir,
        preview_png=preview_png,
        figure_num=1,
        caption=captions[1],
    )

    plot_paired_rmsd(
        frames,
        out_dir=out_dir,
        preview_png=preview_png,
        figure_num=2,
        caption=captions[2],
    )

    plot_contacts_vs_rmsd(
        allposes_csv,
        out_dir=out_dir,
        log_counts=bool(args.log_density),
        preview_png=preview_png,
        figure_num=3,
        caption=captions[3],
    )

    plot_contact_overlap_distributions(
        frames,
        allposes_csv,
        out_dir=out_dir,
        metric_col="head_env_jaccard",
        title="Headgroup Environment Overlap",
        stem="fig_head_env_overlap_distributions",
        preview_png=preview_png,
        figure_num=4,
        caption=captions[4],
    )

    # Commenting out Figure 5 to keep it at 4 figures as requested
    # plot_contact_overlap_distributions(
    #     frames,
    #     allposes_csv,
    #     out_dir=out_dir,
    #     metric_col="headgroup_typed_jaccard",
    #     title="Headgroup Typed Interaction Overlap",
    #     stem="fig_headgroup_typed_overlap_distributions",
    #     takeaway="",
    #     preview_png=preview_png,
    #     figure_num=5,
    #     caption=captions[5],
    # )
    if preview_png and not bool(args.keep_preview):
        _prune_non_pdf(out_dir)
    print(f"Wrote PDF figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
