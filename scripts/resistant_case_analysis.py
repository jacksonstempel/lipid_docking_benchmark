from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from lipid_benchmark.ligands import (
    find_ligand_by_id,
    headgroup_indices_functional,
    _build_rdkit_mol_from_residue,
)
from lipid_benchmark.structures import is_protein_res, load_structure


BACKBONE_ATOMS = {"N", "CA", "C", "O"}


@dataclass(frozen=True)
class ContactStats:
    total_contacts: int
    backbone_fraction: float


def _atom_is_heavy(atom) -> bool:
    elem = getattr(getattr(atom, "element", None), "name", "")
    return str(elem).strip().upper() not in {"", "H"}


def compute_headgroup_contacts(
    cif_path: Path,
    ligand_id: str,
    *,
    cutoff_a: float = 4.0,
) -> ContactStats:
    """Count protein heavy atoms within cutoff of any headgroup heavy atom (experimental structure)."""
    structure = load_structure(cif_path)
    ligand = find_ligand_by_id(structure, ligand_id)
    head_idx = headgroup_indices_functional(ligand)
    head_xyz = np.array(
        [ligand.atoms[i].xyz for i in head_idx if ligand.atoms[i].element.upper() != "H"],
        dtype=float,
    )
    if head_xyz.size == 0:
        raise RuntimeError(f"{cif_path.name}: empty headgroup selection for {ligand_id}")

    prot_xyz: list[list[float]] = []
    prot_is_backbone: list[bool] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not is_protein_res(residue):
                    continue
                for atom in residue:
                    if not _atom_is_heavy(atom):
                        continue
                    prot_xyz.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    prot_is_backbone.append(atom.name.strip().upper() in BACKBONE_ATOMS)
        break  # model 0 only

    xyz = np.asarray(prot_xyz, dtype=float)
    if xyz.size == 0:
        raise RuntimeError(f"{cif_path.name}: no protein heavy atoms found")
    back = np.asarray(prot_is_backbone, dtype=bool)

    cutoff2 = float(cutoff_a) ** 2
    # Vectorized distance check: protein_atom is a contact if min(headgroup distance) <= cutoff.
    d2 = np.sum((xyz[:, None, :] - head_xyz[None, :, :]) ** 2, axis=2)
    contact_mask = d2.min(axis=1) <= cutoff2
    total = int(contact_mask.sum())
    if total == 0:
        return ContactStats(total_contacts=0, backbone_fraction=float("nan"))
    backbone_fraction = float(back[contact_mask].mean())
    return ContactStats(total_contacts=total, backbone_fraction=backbone_fraction)


def _has_phosphate(mol) -> bool:
    return any(a.GetSymbol() == "P" for a in mol.GetAtoms())


def _has_terminal_carboxylic_acid(mol) -> bool:
    # Identify a carbon with >=2 oxygen neighbors where at least one oxygen is terminal (degree==1).
    # This is a practical heuristic for free fatty acids; it avoids counting ester carbonyls.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        o_neigh = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O"]
        if len(o_neigh) < 2:
            continue
        if any(o.GetDegree() == 1 for o in o_neigh):
            return True
    return False


def _count_hetero_atoms(mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() not in {"C", "H"})


def classify_lipid_class(*, mol) -> str:
    """Assign a coarse lipid class from ligand chemistry.

    Categories are intentionally broad (fatty acid, phospholipid, glycolipid, sterol, other).
    """
    rings = int(mol.GetRingInfo().NumRings())
    hetero = _count_hetero_atoms(mol)
    atoms = mol.GetNumAtoms()
    carbon = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    carbon_frac = carbon / atoms if atoms else 0.0

    if _has_phosphate(mol):
        return "phospholipid"
    if rings >= 4 and carbon_frac >= 0.85 and hetero <= 2:
        return "sterol"
    o_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O")
    if rings >= 1 and o_count >= 4:
        return "glycolipid"
    if _has_terminal_carboxylic_acid(mol):
        return "fatty acid"
    return "other"


def classify_headgroup_chemistry(*, mol) -> str:
    """Assign a coarse headgroup chemistry category."""
    if _has_phosphate(mol):
        return "phosphate-containing"
    # Quaternary/amine-rich headgroups typically contain nitrogen.
    if any(a.GetSymbol() == "N" for a in mol.GetAtoms()):
        # Many choline-like headgroups are quaternary N (degree >= 3).
        if any(a.GetSymbol() == "N" and a.GetDegree() >= 3 for a in mol.GetAtoms()):
            return "amine/quaternary nitrogen"
        return "amine/quaternary nitrogen"
    if _has_terminal_carboxylic_acid(mol):
        return "carboxylate"
    return "hydroxyl-dominant"


def _mann_whitney(x: np.ndarray, y: np.ndarray) -> float:
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(res.pvalue)


def _format_mean_median(series: pd.Series) -> tuple[float, float]:
    arr = pd.to_numeric(series, errors="coerce")
    return float(arr.mean()), float(arr.median())


def _fisher_by_category(table: pd.DataFrame) -> pd.DataFrame:
    """Compute 2x2 Fisher exact tests for each category vs all others."""
    out_rows: list[dict[str, object]] = []
    for category in table.index:
        a = int(table.loc[category, "Resistant"])
        b = int(table.loc[category, "Sensitive"])
        c = int(table["Resistant"].sum() - a)
        d = int(table["Sensitive"].sum() - b)
        odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
        out_rows.append({"category": category, "odds_ratio": float(odds), "p_value": float(p)})
    return pd.DataFrame(out_rows).sort_values("p_value")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze memorization-resistant cases after Gly mutagenesis")
    parser.add_argument(
        "--wt-summary",
        type=Path,
        default=Path("output/benchmark/benchmark_summary.csv"),
        help="Wild-type benchmark_summary.csv",
    )
    parser.add_argument(
        "--gly-summary",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A/benchmark_gly/benchmark_summary.csv"),
        help="Gly mutant benchmark_summary.csv",
    )
    parser.add_argument(
        "--mutation-summary",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A/mutation_summary.csv"),
        help="mutation_summary.csv",
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=Path("structures/experimental"),
        help="Directory with experimental CIFs",
    )
    parser.add_argument(
        "--protein-rmsd-cutoff",
        type=float,
        default=3.0,
        help="Fold-success cutoff for protein RMSD (Angstrom)",
    )
    parser.add_argument(
        "--headgroup-rmsd-cutoff",
        type=float,
        default=3.0,
        help="Resistant cutoff for headgroup RMSD (Angstrom)",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=4.0,
        help="Headgroup contact cutoff (Angstrom)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/adversarial/bs_mutagenesis_cutoff5A/resistant_case_analysis"),
        help="Output directory",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wt = pd.read_csv(args.wt_summary)
    wt = wt[wt["method"] == "boltz"].copy()
    wt = wt[["pdbid", "ref_ligand_id", "headgroup_rmsd"]].rename(columns={"headgroup_rmsd": "wt_head_rmsd"})
    wt["resname"] = wt["ref_ligand_id"].str.split(":").str[1]

    gly = pd.read_csv(args.gly_summary)
    gly = gly[gly["method"] == "boltz"].copy()
    gly = gly[["pdbid", "headgroup_rmsd", "protein_rmsd"]].rename(
        columns={"headgroup_rmsd": "gly_head_rmsd", "protein_rmsd": "gly_protein_rmsd"}
    )

    mut = pd.read_csv(args.mutation_summary)
    mut = mut[["pdbid", "actual_changes_gly"]].rename(columns={"actual_changes_gly": "mutations_gly"})

    df = wt.merge(gly, on="pdbid", how="inner").merge(mut, on="pdbid", how="left")
    df = df.dropna(subset=["gly_protein_rmsd", "gly_head_rmsd", "wt_head_rmsd"]).copy()
    df = df[df["gly_protein_rmsd"] <= float(args.protein_rmsd_cutoff)].copy()

    df["group"] = np.where(df["gly_head_rmsd"] < float(args.headgroup_rmsd_cutoff), "Resistant", "Sensitive")

    # Contact + chemistry analysis from experimental structures.
    total_contacts: list[int] = []
    backbone_fracs: list[float] = []
    lipid_classes: list[str] = []
    headgroup_types: list[str] = []
    for _, row in df.iterrows():
        pdbid = str(row["pdbid"])
        cif_path = args.structures_dir / f"{pdbid}.cif"
        ligand_id = str(row["ref_ligand_id"])

        stats_row = compute_headgroup_contacts(cif_path, ligand_id, cutoff_a=float(args.contact_cutoff))
        total_contacts.append(stats_row.total_contacts)
        backbone_fracs.append(stats_row.backbone_fraction)

        structure = load_structure(cif_path)
        ligand = find_ligand_by_id(structure, ligand_id)
        mol, _ = _build_rdkit_mol_from_residue(ligand)
        if mol is None or mol.GetNumAtoms() == 0:
            lipid_classes.append("unknown")
            headgroup_types.append("unknown")
        else:
            lipid_classes.append(classify_lipid_class(mol=mol))
            headgroup_types.append(classify_headgroup_chemistry(mol=mol))

    df["contact_cutoff_a"] = float(args.contact_cutoff)
    df["headgroup_contacts"] = total_contacts
    df["backbone_contact_fraction"] = backbone_fracs
    df["lipid_class"] = lipid_classes
    df["headgroup_chemistry"] = headgroup_types

    # Save the per-target table (useful for follow-up drilling).
    per_target_path = out_dir / "resistant_case_table.csv"
    df.sort_values(["group", "pdbid"]).to_csv(per_target_path, index=False)

    # Summary metrics (Analyses 2-5)
    rows: list[dict[str, object]] = []
    resistant = df[df.group == "Resistant"].copy()
    sensitive = df[df.group == "Sensitive"].copy()

    def _metric_row(label: str, col: str) -> dict[str, object]:
        r_mean, r_med = _format_mean_median(resistant[col])
        s_mean, s_med = _format_mean_median(sensitive[col])
        r_vals = pd.to_numeric(resistant[col], errors="coerce").dropna().to_numpy(float)
        s_vals = pd.to_numeric(sensitive[col], errors="coerce").dropna().to_numpy(float)
        p = _mann_whitney(r_vals, s_vals) if len(r_vals) and len(s_vals) else float("nan")
        return {
            "metric": label,
            "resistant_n": int(len(resistant)),
            "sensitive_n": int(len(sensitive)),
            "resistant_mean": r_mean,
            "resistant_median": r_med,
            "sensitive_mean": s_mean,
            "sensitive_median": s_med,
            "p_value_mann_whitney": p,
        }

    rows.append(_metric_row("Mutation count (Gly)", "mutations_gly"))
    rows.append(_metric_row(f"Headgroup contacts within {args.contact_cutoff:.1f} A", "headgroup_contacts"))
    rows.append(_metric_row("Backbone contact fraction", "backbone_contact_fraction"))
    rows.append(_metric_row("WT headgroup RMSD (Boltz)", "wt_head_rmsd"))

    summary_metrics = pd.DataFrame(rows)
    summary_metrics_path = out_dir / "summary_metrics.csv"
    summary_metrics.to_csv(summary_metrics_path, index=False)

    # WT accuracy stratification (Analysis 5)
    def _frac_lt(series: pd.Series, cutoff: float) -> float:
        vals = pd.to_numeric(series, errors="coerce")
        return float((vals < cutoff).mean())

    wt_strata = pd.DataFrame(
        [
            {
                "group": "Resistant",
                "n": int(len(resistant)),
                "fraction_wt_lt_1_5": _frac_lt(resistant["wt_head_rmsd"], 1.5),
                "fraction_wt_1_5_to_3": float(
                    ((resistant["wt_head_rmsd"] >= 1.5) & (resistant["wt_head_rmsd"] < 3.0)).mean()
                ),
            },
            {
                "group": "Sensitive",
                "n": int(len(sensitive)),
                "fraction_wt_lt_1_5": _frac_lt(sensitive["wt_head_rmsd"], 1.5),
                "fraction_wt_1_5_to_3": float(
                    ((sensitive["wt_head_rmsd"] >= 1.5) & (sensitive["wt_head_rmsd"] < 3.0)).mean()
                ),
            },
        ]
    )
    wt_strata.to_csv(out_dir / "wt_accuracy_strata.csv", index=False)

    # Contingency tables (Analyses 1 & 6)
    lipid_table = (
        df.pivot_table(index="lipid_class", columns="group", values="pdbid", aggfunc="count", fill_value=0)
        .rename_axis(index=None, columns=None)
        .rename(columns={"Resistant": "Resistant", "Sensitive": "Sensitive"})
        .sort_index()
    )
    lipid_table.to_csv(out_dir / "lipid_class_contingency.csv")
    _fisher_by_category(lipid_table).to_csv(out_dir / "lipid_class_fisher.csv", index=False)

    head_table = (
        df.pivot_table(
            index="headgroup_chemistry", columns="group", values="pdbid", aggfunc="count", fill_value=0
        )
        .rename_axis(index=None, columns=None)
        .rename(columns={"Resistant": "Resistant", "Sensitive": "Sensitive"})
        .sort_index()
    )
    head_table.to_csv(out_dir / "headgroup_chemistry_contingency.csv")
    _fisher_by_category(head_table).to_csv(out_dir / "headgroup_chemistry_fisher.csv", index=False)

    # Human-readable markdown report.
    report_path = out_dir / "report.md"
    excluded = pd.read_csv(args.gly_summary)
    excluded = excluded[(excluded.method == "boltz") & (excluded.protein_rmsd > float(args.protein_rmsd_cutoff))]
    excluded_ids = ", ".join(sorted(excluded.pdbid.astype(str).tolist()))

    def _md_table(df_: pd.DataFrame, floatfmt: str = ".3f") -> str:
        """Render a DataFrame as a small Markdown table without extra deps (tabulate)."""
        fmt = floatfmt.lstrip(".")
        cols = list(df_.columns)

        def _format(v) -> str:
            if v is None:
                return ""
            if isinstance(v, float):
                if not np.isfinite(v):
                    return ""
                return format(v, fmt)
            if isinstance(v, (np.floating,)):
                fv = float(v)
                if not np.isfinite(fv):
                    return ""
                return format(fv, fmt)
            if isinstance(v, (np.integer, int)):
                return str(int(v))
            if isinstance(v, (bool, np.bool_)):
                return "true" if bool(v) else "false"
            return str(v)

        rows = [[_format(v) for v in row] for row in df_.itertuples(index=False, name=None)]
        header = "| " + " | ".join(cols) + " |"
        rule = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join([header, rule, *body])

    md = []
    md.append("# Memorization-Resistant Case Analysis (Gly arm)\n")
    md.append(
        f"Included targets: {len(df)} (protein RMSD <= {args.protein_rmsd_cutoff:.1f} A). "
        f"Excluded for fold failure: {len(excluded)} ({excluded_ids or 'none'}).\n"
    )
    md.append(
        f"Resistant definition: Gly headgroup RMSD < {args.headgroup_rmsd_cutoff:.1f} A. "
        f"Sensitive: Gly headgroup RMSD >= {args.headgroup_rmsd_cutoff:.1f} A.\n"
    )

    md.append("## Summary Metrics (Analyses 2-5)\n")
    md.append(_md_table(summary_metrics, floatfmt=".4g") + "\n")

    md.append("## WT Accuracy Strata (Analysis 5)\n")
    md.append(_md_table(wt_strata, floatfmt=".3f") + "\n")

    md.append("## Lipid Class Breakdown (Analysis 1)\n")
    md.append(_md_table(lipid_table.reset_index().rename(columns={"index": "lipid_class"})) + "\n")
    md.append("### Fisher exact tests (class vs all others)\n")
    md.append(
        _md_table(pd.read_csv(out_dir / "lipid_class_fisher.csv"), floatfmt=".4g")
        + "\n"
    )

    md.append("## Headgroup Chemistry Breakdown (Analysis 6)\n")
    md.append(_md_table(head_table.reset_index().rename(columns={"index": "headgroup_chemistry"})) + "\n")
    md.append("### Fisher exact tests (chemistry vs all others)\n")
    md.append(
        _md_table(pd.read_csv(out_dir / "headgroup_chemistry_fisher.csv"), floatfmt=".4g")
        + "\n"
    )

    md.append("## Notes / Caveats\n")
    md.append(
        "- Lipid class and headgroup chemistry categories are heuristic, based on RDKit-inferred bonding from the "
        "experimental coordinates. They are intended as a coarse signal, not an authoritative lipid taxonomy.\n"
    )
    md.append(
        f"- Contact counts are distance-based (protein heavy atoms within {args.contact_cutoff:.1f} A of any headgroup "
        "heavy atom in the experimental structure).\n"
    )
    report_path.write_text("\n".join(md))

    print(f"Wrote: {per_target_path}")
    print(f"Wrote: {summary_metrics_path}")
    print(f"Wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
