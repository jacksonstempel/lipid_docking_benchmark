"""
Analysis helpers for the benchmark SQLite database.

Plain-language overview

- The database stores per-pose metrics in a single `allposes` table.
- This module provides small, testable utilities to select top-1 poses,
  best-of-K poses, and to summarize results for manuscript tables/figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


METHOD_BOLTZ = "boltz"
METHOD_VINA = "vina_pose"
METHOD_GNINA_CNN = "gnina_cnn_pose"
METHOD_GNINA_NOCNN = "gnina_nocnn_pose"


NUMERIC_COLUMNS = (
    "ligand_rmsd",
    "headgroup_rmsd",
    "head_env_jaccard",
    "headgroup_typed_jaccard",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def select_top1(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return pose_index=1 rows for a method."""
    subset = df[(df["method"] == method) & (df["pose_index"] == 1)].copy()
    subset["pose_index"] = _numeric(subset["pose_index"])
    return subset


def select_bestk(df: pd.DataFrame, method: str, k: int) -> pd.DataFrame:
    """Return the best-of-K pose (min ligand RMSD) per target for a method."""
    subset = df[(df["method"] == method)].copy()
    subset["pose_index"] = _numeric(subset["pose_index"])
    subset["ligand_rmsd"] = _numeric(subset["ligand_rmsd"])
    subset = subset[subset["pose_index"] <= int(k)]
    if subset.empty:
        return subset
    idx = subset.groupby("pdbid")["ligand_rmsd"].idxmin()
    return subset.loc[idx].copy()


def _summary_stats(values: Iterable[float]) -> Tuple[float, float, float, int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    return mean, median, iqr, int(arr.size)


def format_summary(mean: float, median: float, iqr: float) -> str:
    """Format mean/median/IQR for manuscript tables."""
    if not np.isfinite(mean):
        return "NA"
    return f"{mean:.2f} ({median:.2f}) [{iqr:.2f}]"


def build_per_target(df_allposes: pd.DataFrame, *, k: int = 20) -> pd.DataFrame:
    """
    Build a per-target table with the columns needed for GNINA figures and tables.
    """
    pdbids = sorted(df_allposes["pdbid"].dropna().unique().tolist())
    per = pd.DataFrame({"pdbid": pdbids})

    boltz = select_top1(df_allposes, METHOD_BOLTZ)
    vina_top1 = select_top1(df_allposes, METHOD_VINA)
    vina_best = select_bestk(df_allposes, METHOD_VINA, k)
    gnina_cnn_top1 = select_top1(df_allposes, METHOD_GNINA_CNN)
    gnina_cnn_best = select_bestk(df_allposes, METHOD_GNINA_CNN, k)
    gnina_nocnn_top1 = select_top1(df_allposes, METHOD_GNINA_NOCNN)
    gnina_nocnn_best = select_bestk(df_allposes, METHOD_GNINA_NOCNN, k)

    def _merge(metric_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cols = ["pdbid", *NUMERIC_COLUMNS]
        subset = metric_df[cols].copy()
        rename = {col: f"{prefix}_{col}" for col in NUMERIC_COLUMNS}
        return subset.rename(columns=rename)

    per = per.merge(_merge(boltz, "boltz_top1"), on="pdbid", how="left")
    per = per.merge(_merge(vina_top1, "vina_top1"), on="pdbid", how="left")
    per = per.merge(_merge(gnina_cnn_top1, "gnina_cnn_top1"), on="pdbid", how="left")
    per = per.merge(_merge(gnina_nocnn_top1, "gnina_nocnn_top1"), on="pdbid", how="left")

    # Alias columns to match plotting expectations.
    per["boltz_head_env_jaccard"] = per["boltz_top1_head_env_jaccard"]
    per["boltz_headgroup_typed_jaccard"] = per["boltz_top1_headgroup_typed_jaccard"]
    per["vina_head_env_jaccard"] = per["vina_top1_head_env_jaccard"]
    per["vina_headgroup_typed_jaccard"] = per["vina_top1_headgroup_typed_jaccard"]
    per["gnina_cnn_head_env_jaccard"] = per["gnina_cnn_top1_head_env_jaccard"]
    per["gnina_cnn_headgroup_typed_jaccard"] = per["gnina_cnn_top1_headgroup_typed_jaccard"]
    per["gnina_nocnn_head_env_jaccard"] = per["gnina_nocnn_top1_head_env_jaccard"]
    per["gnina_nocnn_headgroup_typed_jaccard"] = per["gnina_nocnn_top1_headgroup_typed_jaccard"]

    per = per.merge(
        vina_best[["pdbid", "ligand_rmsd", "headgroup_rmsd", "head_env_jaccard", "headgroup_typed_jaccard"]].rename(
            columns={
                "ligand_rmsd": "vina_bestK_ligand_rmsd",
                "headgroup_rmsd": "vina_bestK_headgroup_rmsd",
                "head_env_jaccard": "vina_bestK_head_env_jaccard",
                "headgroup_typed_jaccard": "vina_bestK_headgroup_typed_jaccard",
            }
        ),
        on="pdbid",
        how="left",
    )
    per = per.merge(
        gnina_cnn_best[["pdbid", "ligand_rmsd", "headgroup_rmsd"]].rename(
            columns={
                "ligand_rmsd": "gnina_cnn_bestK_ligand_rmsd",
                "headgroup_rmsd": "gnina_cnn_bestK_headgroup_rmsd",
            }
        ),
        on="pdbid",
        how="left",
    )
    per = per.merge(
        gnina_nocnn_best[["pdbid", "ligand_rmsd", "headgroup_rmsd"]].rename(
            columns={
                "ligand_rmsd": "gnina_nocnn_bestK_ligand_rmsd",
                "headgroup_rmsd": "gnina_nocnn_bestK_headgroup_rmsd",
            }
        ),
        on="pdbid",
        how="left",
    )

    per["vina_gap_ligand_rmsd"] = _numeric(per["vina_top1_ligand_rmsd"]) - _numeric(per["vina_bestK_ligand_rmsd"])
    per["gnina_cnn_gap_ligand_rmsd"] = _numeric(per["gnina_cnn_top1_ligand_rmsd"]) - _numeric(
        per["gnina_cnn_bestK_ligand_rmsd"]
    )
    per["gnina_nocnn_gap_ligand_rmsd"] = _numeric(per["gnina_nocnn_top1_ligand_rmsd"]) - _numeric(
        per["gnina_nocnn_bestK_ligand_rmsd"]
    )

    return per


@dataclass(frozen=True)
class SummaryTable:
    numeric: pd.DataFrame
    formatted: pd.DataFrame


def summarize_methods(per_target: pd.DataFrame) -> SummaryTable:
    """
    Summarize the methods used in the manuscript table.
    """
    rows = []
    fmt_rows = []

    def _add(method: str, lig_col: str, head_col: str, env_col: str, typed_col: str) -> None:
        lig = _summary_stats(_numeric(per_target[lig_col]))
        head = _summary_stats(_numeric(per_target[head_col]))
        env = _summary_stats(_numeric(per_target[env_col]))
        typed = _summary_stats(_numeric(per_target[typed_col]))
        rows.append(
            {
                "method": method,
                "ligand_rmsd_mean": lig[0],
                "ligand_rmsd_median": lig[1],
                "ligand_rmsd_iqr": lig[2],
                "headgroup_rmsd_mean": head[0],
                "headgroup_rmsd_median": head[1],
                "headgroup_rmsd_iqr": head[2],
                "head_env_jaccard_mean": env[0],
                "head_env_jaccard_median": env[1],
                "head_env_jaccard_iqr": env[2],
                "headgroup_typed_jaccard_mean": typed[0],
                "headgroup_typed_jaccard_median": typed[1],
                "headgroup_typed_jaccard_iqr": typed[2],
            }
        )
        fmt_rows.append(
            {
                "method": method,
                "ligand_rmsd": format_summary(*lig[:3]),
                "headgroup_rmsd": format_summary(*head[:3]),
                "head_env_jaccard": format_summary(*env[:3]),
                "headgroup_typed_jaccard": format_summary(*typed[:3]),
            }
        )

    _add(
        "Boltz-2",
        "boltz_top1_ligand_rmsd",
        "boltz_top1_headgroup_rmsd",
        "boltz_top1_head_env_jaccard",
        "boltz_top1_headgroup_typed_jaccard",
    )
    _add(
        "Vina top-1",
        "vina_top1_ligand_rmsd",
        "vina_top1_headgroup_rmsd",
        "vina_top1_head_env_jaccard",
        "vina_top1_headgroup_typed_jaccard",
    )
    _add(
        "GNINA CNN top-1",
        "gnina_cnn_top1_ligand_rmsd",
        "gnina_cnn_top1_headgroup_rmsd",
        "gnina_cnn_top1_head_env_jaccard",
        "gnina_cnn_top1_headgroup_typed_jaccard",
    )
    _add(
        "GNINA no-CNN top-1",
        "gnina_nocnn_top1_ligand_rmsd",
        "gnina_nocnn_top1_headgroup_rmsd",
        "gnina_nocnn_top1_head_env_jaccard",
        "gnina_nocnn_top1_headgroup_typed_jaccard",
    )
    _add(
        "Vina top-20 best",
        "vina_bestK_ligand_rmsd",
        "vina_bestK_headgroup_rmsd",
        "vina_bestK_head_env_jaccard",
        "vina_bestK_headgroup_typed_jaccard",
    )

    return SummaryTable(numeric=pd.DataFrame(rows), formatted=pd.DataFrame(fmt_rows))


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def torsion_bin_table(
    per_target: pd.DataFrame,
    torsions: pd.Series,
    *,
    bins: Tuple[int, int, int] = (20, 40, 10_000),
) -> SummaryTable:
    """
    Compute the torsion-bin table used in the manuscript.
    """
    df = per_target.merge(torsions.rename("torsdof"), left_on="pdbid", right_index=True, how="left")
    df = df.dropna(subset=["torsdof"])
    df["torsdof"] = _numeric(df["torsdof"])

    def _bin_label(n: int) -> str:
        if n <= bins[0]:
            return "0–20"
        if n <= bins[1]:
            return "21–40"
        return "≥41"

    df["torsion_bin"] = df["torsdof"].apply(_bin_label)

    rows = []
    fmt_rows = []
    for label in ["0–20", "21–40", "≥41"]:
        sub = df[df["torsion_bin"] == label]
        n = int(sub.shape[0])
        def _median_success(series: pd.Series) -> Tuple[float, float, float, float]:
            vals = _numeric(series)
            med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
            successes = int(np.sum(vals <= 2.0))
            ci_lo, ci_hi = wilson_interval(successes, n)
            return med, successes / n if n else float("nan"), ci_lo, ci_hi

        boltz = _median_success(sub["boltz_top1_ligand_rmsd"])
        vina1 = _median_success(sub["vina_top1_ligand_rmsd"])
        vina20 = _median_success(sub["vina_bestK_ligand_rmsd"])

        rows.append(
            {
                "bin": label,
                "n": n,
                "boltz_median": boltz[0],
                "boltz_success": boltz[1],
                "boltz_ci_low": boltz[2],
                "boltz_ci_high": boltz[3],
                "vina_top1_median": vina1[0],
                "vina_top1_success": vina1[1],
                "vina_top1_ci_low": vina1[2],
                "vina_top1_ci_high": vina1[3],
                "vina_top20_median": vina20[0],
                "vina_top20_success": vina20[1],
                "vina_top20_ci_low": vina20[2],
                "vina_top20_ci_high": vina20[3],
            }
        )

        fmt_rows.append(
            {
                "bin": label,
                "n": n,
                "boltz": f"{boltz[0]:.2f}; {boltz[1]:.2f} [{boltz[2]:.2f}, {boltz[3]:.2f}]",
                "vina_top1": f"{vina1[0]:.2f}; {vina1[1]:.2f} [{vina1[2]:.2f}, {vina1[3]:.2f}]",
                "vina_top20": f"{vina20[0]:.2f}; {vina20[1]:.2f} [{vina20[2]:.2f}, {vina20[3]:.2f}]",
            }
        )

    return SummaryTable(numeric=pd.DataFrame(rows), formatted=pd.DataFrame(fmt_rows))
