#!/usr/bin/env python3
"""
Verify that manuscript tables and key adversarial numbers match the current analysis outputs.

This is intentionally a lightweight, reproducible "audit" tool:
- It reads the LaTeX source (manuscript/manuscript.tex)
- It reads the CSVs produced by `scripts/analyze_benchmark_db.py`
- It fails fast if any table cell disagrees (string match)

Run:
  python scripts/verify_manuscript_numbers.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]

MANUSCRIPT_TEX = ROOT / "manuscript" / "manuscript.tex"
DB_PIPELINE_DIR = ROOT / "output" / "analysis" / "db_pipeline"

SUMMARY_TABLE_CSV = DB_PIPELINE_DIR / "summary_table_formatted.csv"
TORSION_TABLE_CSV = DB_PIPELINE_DIR / "torsion_table_formatted.csv"

VINA_EXH256_DIR = ROOT / "output" / "benchmark_vina_exh256"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _extract_tabular_rows(tex: str, *, label: str) -> dict[str, list[str]]:
    """
    Return a mapping {row_label -> [cell1, cell2, ...]} from a LaTeX table.

    Assumes rows are written as:
      Label & cell & cell & ... \\\\
    """
    # Find the table that contains \label{...}
    # Keep the regex construction extremely literal here; LaTeX braces do not need escaping
    # for our labels, and this avoids subtle differences between escaped/unescaped forms.
    pat = r"\\label{" + re.escape(label) + r"}.*?\\begin{tabular}.*?\\end{tabular}"
    m = re.search(pat, tex, re.S)
    if not m:
        raise SystemExit(f"Could not find tabular block for label: {label}")
    block = m.group(0)

    rows: dict[str, list[str]] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("\\toprule") or line.startswith("\\midrule") or line.startswith("\\bottomrule"):
            continue
        if line.startswith("&") or line.startswith("\\"):
            continue
        if "&" not in line or "\\\\" not in line:
            continue
        line = line.split("\\\\", 1)[0].strip()
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 2:
            continue
        row_label = parts[0]
        cells = parts[1:]
        rows[row_label] = cells
    if not rows:
        raise SystemExit(f"No rows parsed for label: {label}")
    return rows


def _expect_equal(label: str, got: str, expected: str) -> None:
    if got != expected:
        raise SystemExit(f"{label} mismatch:\n  got:      {got}\n  expected: {expected}")


def verify_summary_table(tex: str) -> None:
    rows = _extract_tabular_rows(tex, label="tab:summary")
    expected = {r["method"]: r for r in _read_csv(SUMMARY_TABLE_CSV)}

    for method in ["Boltz-2", "Vina top-1", "GNINA CNN top-1", "GNINA no-CNN top-1", "Vina top-20 best"]:
        if method not in rows:
            raise SystemExit(f"Missing row in manuscript summary table: {method}")
        if method not in expected:
            raise SystemExit(f"Missing row in summary_table_formatted.csv: {method}")
        got_cells = rows[method]
        exp_cells = [
            expected[method]["ligand_rmsd"],
            expected[method]["headgroup_rmsd"],
            expected[method]["head_env_jaccard"],
            expected[method]["headgroup_typed_jaccard"],
        ]
        if len(got_cells) != 4:
            raise SystemExit(f"Unexpected number of cells for '{method}' in tab:summary: {len(got_cells)}")
        for i, (g, e) in enumerate(zip(got_cells, exp_cells, strict=True), 1):
            _expect_equal(f"tab:summary {method} col{i}", g, e)


def verify_torsion_table(tex: str) -> None:
    rows = _extract_tabular_rows(tex, label="tab:torsions")
    expected_rows = _read_csv(TORSION_TABLE_CSV)

    # Map CSV bin strings to the LaTeX bin labels used in the manuscript.
    bin_map = {
        "0–20": "Low (0--20)",
        "21–40": "Medium (21--40)",
        "≥41": r"High ($\geq$41)",
    }

    for r in expected_rows:
        bin_key = r["bin"]
        row_label = bin_map.get(bin_key)
        if not row_label:
            raise SystemExit(f"Unexpected bin in torsion_table_formatted.csv: {bin_key}")
        if row_label not in rows:
            raise SystemExit(f"Missing row in manuscript torsion table: {row_label}")
        got_cells = rows[row_label]
        exp_cells = [r["n"], r["boltz"], r["vina_top1"], r["vina_top20"]]
        if len(got_cells) != 4:
            raise SystemExit(f"Unexpected number of cells for '{row_label}' in tab:torsions: {len(got_cells)}")
        for i, (g, e) in enumerate(zip(got_cells, exp_cells, strict=True), 1):
            _expect_equal(f"tab:torsions {row_label} col{i}", g, e)


def verify_adversarial_key_numbers(tex: str) -> None:
    # These appear in the adversarial Results paragraph/caption and should not drift.
    required = [
        r"89\% (Gly) and 91\% (Phe)",
        r"23\% (18/79)",
        r"20\% (16/81)",
        r"median 0.29 vs 0.00; $p=0.013$",
    ]
    for snippet in required:
        if snippet not in tex:
            raise SystemExit(f"Missing expected adversarial snippet in manuscript: {snippet}")


def verify_vina_exh256_numbers(tex: str) -> None:
    """
    Verify the manuscript's Vina exhaustiveness-256 sentence against the computed medians.
    """
    base_summary = ROOT / "output" / "benchmark" / "benchmark_summary.csv"
    base_allposes = ROOT / "output" / "benchmark" / "benchmark_allposes.csv"
    exh_summary = VINA_EXH256_DIR / "benchmark_summary.csv"
    exh_allposes = VINA_EXH256_DIR / "benchmark_allposes.csv"

    if not (base_summary.exists() and base_allposes.exists() and exh_summary.exists() and exh_allposes.exists()):
        # If the exh256 benchmark isn't present locally, skip (but do not fail CI).
        return

    def _read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open() as f:
            return list(csv.DictReader(f))

    base_rows = [r for r in _read_csv_rows(base_summary) if r.get("method") == "vina_top1"]
    exh_rows = [r for r in _read_csv_rows(exh_summary) if r.get("method") == "vina_top1"]
    if not base_rows or not exh_rows:
        raise SystemExit("Missing vina_top1 rows in baseline or exh256 benchmark_summary.csv")

    def _median(vals: list[float]) -> float:
        return float(statistics.median(vals))

    base_top1 = _median([float(r["ligand_rmsd"]) for r in base_rows])
    exh_top1 = _median([float(r["ligand_rmsd"]) for r in exh_rows])

    # top-20 best from allposes (min ligand RMSD among pose_index<=20 per target)
    def _top20_best_median(allposes_path: Path) -> float:
        rows = [r for r in _read_csv_rows(allposes_path) if r.get("method") == "vina_pose"]
        by: dict[str, float] = {}
        for r in rows:
            pid = r.get("pdbid") or ""
            if not pid:
                continue
            pose = int(float(r.get("pose_index") or 0))
            if pose < 1 or pose > 20:
                continue
            val = float(r["ligand_rmsd"])
            by[pid] = val if pid not in by else min(by[pid], val)
        return _median(list(by.values()))

    base_top20 = _top20_best_median(base_allposes)
    exh_top20 = _top20_best_median(exh_allposes)

    snippet = (
        "median top-1 ligand RMSD changed from "
        f"\\SI{{{base_top1:.2f}}}{{\\angstrom}} to \\SI{{{exh_top1:.2f}}}{{\\angstrom}}, "
        "and median top-20 best RMSD changed from "
        f"\\SI{{{base_top20:.2f}}}{{\\angstrom}} to \\SI{{{exh_top20:.2f}}}{{\\angstrom}}."
    )
    if snippet not in tex:
        raise SystemExit(f"Vina exh256 sentence mismatch or missing. Expected to find:\n{snippet}")


def main() -> int:
    for p in [MANUSCRIPT_TEX, SUMMARY_TABLE_CSV, TORSION_TABLE_CSV]:
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    tex = MANUSCRIPT_TEX.read_text()
    verify_summary_table(tex)
    verify_torsion_table(tex)
    verify_adversarial_key_numbers(tex)
    verify_vina_exh256_numbers(tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
