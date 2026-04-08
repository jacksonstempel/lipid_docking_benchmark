#!/usr/bin/env python3
"""
Verify that manuscript tables and key adversarial numbers match the current analysis outputs.

This is intentionally a lightweight, reproducible "audit" tool:
- It reads the LaTeX source (manuscript/manuscript.tex)
- It reads the supporting information source (manuscript/supporting_information.tex)
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

from lipid_benchmark.public_archive import canonical_repro_archive, require_paths


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = canonical_repro_archive(ROOT)

MANUSCRIPT_TEX = ROOT / "manuscript" / "manuscript.tex"
SI_TEX = ROOT / "manuscript" / "supporting_information.tex"
DB_PIPELINE_DIR = ROOT / "output" / "analysis" / "db_pipeline"

SUMMARY_TABLE_CSV = DB_PIPELINE_DIR / "summary_table_formatted.csv"
TORSION_TABLE_CSV = DB_PIPELINE_DIR / "torsion_table_formatted.csv"


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


def verify_torsion_table(si_tex: str) -> None:
    rows = _extract_tabular_rows(si_tex, label="tab:flexibility")
    expected_rows = _read_csv(TORSION_TABLE_CSV)

    # Map CSV bin strings to the LaTeX bin labels used in the supporting information.
    bin_map = {
        "0–20": "0--20",
        "21–40": "21--40",
        "≥41": r"$\geq$41",
    }

    for r in expected_rows:
        bin_key = r["bin"]
        row_label = bin_map.get(bin_key)
        if not row_label:
            raise SystemExit(f"Unexpected bin in torsion_table_formatted.csv: {bin_key}")
        if row_label not in rows:
            raise SystemExit(f"Missing row in SI torsion table: {row_label}")
        got_cells = rows[row_label]
        exp_cells = [r["n"], r["boltz"], r["vina_top1"], r["vina_top20"]]
        if len(got_cells) != 4:
            raise SystemExit(f"Unexpected number of cells for '{row_label}' in tab:flexibility: {len(got_cells)}")
        for i, (g, e) in enumerate(zip(got_cells, exp_cells, strict=True), 1):
            _expect_equal(f"tab:flexibility {row_label} col{i}", g, e)


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


def verify_si_robustness_table(si_tex: str) -> None:
    rows = _extract_tabular_rows(si_tex, label="tab:robustness")

    base_summary = ARCHIVE.baseline_summary
    base_allposes = ARCHIVE.baseline_allposes
    exh_summary = ARCHIVE.vina_exh256_summary
    exh_allposes = ARCHIVE.vina_exh256_allposes
    alt_summary = ARCHIVE.boltz_high_sampling_summary
    require_paths([base_summary, base_allposes, exh_summary, exh_allposes, alt_summary])

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

    def _pct_success(vals: list[float], cutoff: float = 2.0) -> str:
        pct = 100.0 * sum(v <= cutoff for v in vals) / len(vals)
        return f"{pct:.0f}\\%"

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

    def _top20_best_values(allposes_path: Path) -> list[float]:
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
        return list(by.values())

    base_top20 = _top20_best_median(base_allposes)
    exh_top20 = _top20_best_median(exh_allposes)

    base_top20_vals = _top20_best_values(base_allposes)
    exh_top20_vals = _top20_best_values(exh_allposes)

    expected_vina = {
        r"Top-1 ligand RMSD median (\AA)": ["100 targets", f"{base_top1:.2f}", f"{exh_top1:.2f}"],
        r"Top-1 success ($\leq$ \SI{2}{\angstrom})": [
            "100 targets",
            _pct_success([float(r["ligand_rmsd"]) for r in base_rows]),
            _pct_success([float(r["ligand_rmsd"]) for r in exh_rows]),
        ],
        r"Top-20 best ligand RMSD median (\AA)": ["100 targets", f"{base_top20:.2f}", f"{exh_top20:.2f}"],
        r"Top-20 success ($\leq$ \SI{2}{\angstrom})": [
            "100 targets",
            _pct_success(base_top20_vals),
            _pct_success(exh_top20_vals),
        ],
    }

    for row_label, exp_cells in expected_vina.items():
        got_cells = rows.get(row_label)
        if got_cells is None:
            raise SystemExit(f"Missing robustness row in SI table: {row_label}")
        if len(got_cells) != 3:
            raise SystemExit(f"Unexpected number of cells for '{row_label}' in tab:robustness: {len(got_cells)}")
        for i, (g, e) in enumerate(zip(got_cells, exp_cells, strict=True), 1):
            _expect_equal(f"tab:robustness {row_label} col{i}", g, e)

    alt_rows = [r for r in _read_csv_rows(alt_summary) if r.get("method") == "boltz"]
    if not alt_rows:
        raise SystemExit("Missing boltz rows in higher-sampling benchmark_summary.csv")

    base_boltz_rows = [r for r in _read_csv_rows(base_summary) if r.get("method") == "boltz"]
    base_by = {r["pdbid"]: r for r in base_boltz_rows}
    alt_by = {r["pdbid"]: r for r in alt_rows}
    shared = sorted(set(base_by) & set(alt_by))
    if not shared:
        raise SystemExit("No shared targets between baseline and higher-sampling Boltz benchmarks")

    def _shared_median(key: str, source: dict[str, dict[str, str]]) -> float:
        return _median([float(source[pid][key]) for pid in shared])

    expected_boltz = {
        r"Ligand RMSD median (\AA)": ["99 shared targets", f"{_shared_median('ligand_rmsd', base_by):.2f}", f"{_shared_median('ligand_rmsd', alt_by):.2f}"],
        r"Headgroup RMSD median (\AA)": ["99 shared targets", f"{_shared_median('headgroup_rmsd', base_by):.2f}", f"{_shared_median('headgroup_rmsd', alt_by):.2f}"],
        r"Protein RMSD median (\AA)": ["99 shared targets", f"{_shared_median('protein_rmsd', base_by):.2f}", f"{_shared_median('protein_rmsd', alt_by):.2f}"],
        "Headgroup environment Jaccard median": [
            "99 shared targets",
            f"{_shared_median('head_env_jaccard', base_by):.2f}",
            f"{_shared_median('head_env_jaccard', alt_by):.2f}",
        ],
    }

    for row_label, exp_cells in expected_boltz.items():
        got_cells = rows.get(row_label)
        if got_cells is None:
            raise SystemExit(f"Missing robustness row in SI table: {row_label}")
        if len(got_cells) != 3:
            raise SystemExit(f"Unexpected number of cells for '{row_label}' in tab:robustness: {len(got_cells)}")
        for i, (g, e) in enumerate(zip(got_cells, exp_cells, strict=True), 1):
            _expect_equal(f"tab:robustness {row_label} col{i}", g, e)


def main() -> int:
    require_paths(
        [
            MANUSCRIPT_TEX,
            SI_TEX,
            SUMMARY_TABLE_CSV,
            TORSION_TABLE_CSV,
        ]
    )

    tex = MANUSCRIPT_TEX.read_text()
    si_tex = SI_TEX.read_text()
    verify_summary_table(tex)
    verify_torsion_table(si_tex)
    verify_adversarial_key_numbers(tex)
    verify_si_robustness_table(si_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
