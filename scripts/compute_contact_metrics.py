#!/usr/bin/env python3
"""Compute contact-level metrics and RMSD for Boltz and Vina poses."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

if __package__ in {None, ""}:  # pragma: no cover
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.lib.ligand_pose_core import _pair_ligand_atoms, _select_single_ligand, measure_ligand_pose_all
from scripts.lib.structures import load_structure, split_models

# Optional: silence RDKit chatter
try:  # pragma: no cover
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass

Contact = Tuple[str, str, str]  # (ligand_atom, residue, contact_type)
AtomMap = Dict[str, str]
VINA_MAX_POSES = 20  # use all 20 Vina poses


@dataclass
class ContactSet:
    contacts: Set[Contact]
    distances: Dict[Contact, float]


def _load_contacts(path: Path) -> Dict[str, Dict[int, ContactSet]]:
    out: Dict[str, Dict[int, ContactSet]] = defaultdict(lambda: defaultdict(lambda: ContactSet(set(), {})))
    with path.open() as f:
        for row in csv.DictReader(f):
            pdbid = row["pdbid"]
            pose = int(row.get("pose_index") or 1)
            contact = (row["ligand_atom"], row["residue"], row["contact_type"])
            try:
                dist = float(row["distance"]) if row["distance"] not in ("", None) else np.nan
            except Exception:
                dist = np.nan
            cset = out[pdbid][pose]
            cset.contacts.add(contact)
            cset.distances[contact] = dist
    return out


def _metrics_strict(ref: ContactSet, pred: ContactSet) -> Dict[str, float | int]:
    ref_c = ref.contacts
    pred_c = pred.contacts
    shared = ref_c & pred_c
    tp = len(shared)
    fp = len(pred_c - ref_c)
    fn = len(ref_c - pred_c)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    denom = len(ref_c | pred_c)
    jaccard = tp / denom if denom > 0 else 0.0
    mae_vals = []
    for c in shared:
        d_ref = ref.distances.get(c, np.nan)
        d_pred = pred.distances.get(c, np.nan)
        if np.isfinite(d_ref) and np.isfinite(d_pred):
            mae_vals.append(abs(d_ref - d_pred))
    mae = float(np.mean(mae_vals)) if mae_vals else np.nan
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "distance_mae": mae,
        "ref_contacts": len(ref_c),
        "pred_contacts": len(pred_c),
        "shared_contacts": tp,
    }


def _set_metrics(ref_set: Set[str], pred_set: Set[str], prefix: str) -> Dict[str, float | int]:
    shared = ref_set & pred_set
    tp = len(shared)
    fp = len(pred_set - ref_set)
    fn = len(ref_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    denom = len(ref_set | pred_set)
    jaccard = tp / denom if denom > 0 else 0.0
    return {
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
        f"{prefix}_jaccard": jaccard,
        f"{prefix}_shared": tp,
        f"{prefix}_ref_size": len(ref_set),
        f"{prefix}_pred_size": len(pred_set),
    }


def _metrics_residue(ref: ContactSet, pred: ContactSet) -> Dict[str, float | int]:
    ref_res = {res for _, res, _ in ref.contacts}
    pred_res = {res for _, res, _ in pred.contacts}
    return _set_metrics(ref_res, pred_res, "residue")


def _parse_res(res_id: str) -> Tuple[str, str, int] | None:
    try:
        chain, name, seq = res_id.split(":")
        return chain, name, int(seq)
    except Exception:
        return None


def _align_boltz_residues(ref: ContactSet, pred: ContactSet) -> ContactSet:
    """Relabel Boltz residue numbers when a clear per-chain offset exists."""
    ref_res = [_parse_res(r[1]) for r in ref.contacts]
    pred_res = [_parse_res(r[1]) for r in pred.contacts]
    ref_res = [r for r in ref_res if r]
    pred_res = [r for r in pred_res if r]
    if not ref_res or not pred_res:
        return pred

    ref_by_chain = defaultdict(lambda: defaultdict(list))
    pred_by_chain = defaultdict(lambda: defaultdict(list))
    for chain, name, seq in ref_res:
        ref_by_chain[chain][name].append(seq)
    for chain, name, seq in pred_res:
        pred_by_chain[chain][name].append(seq)

    offsets: Dict[str, int] = {}
    for chain in set(ref_by_chain) & set(pred_by_chain):
        diffs = []
        for name in set(ref_by_chain[chain]) & set(pred_by_chain[chain]):
            for seq_ref in ref_by_chain[chain][name]:
                for seq_pred in pred_by_chain[chain][name]:
                    diffs.append(seq_ref - seq_pred)
        if diffs:
            freq = Counter(diffs)
            best_diff, count = freq.most_common(1)[0]
            if count >= 3:
                offsets[chain] = best_diff
    if not offsets:
        return pred

    new_contacts: Set[Contact] = set()
    new_distances: Dict[Contact, float] = {}
    for (lig, res, ctype), dist in pred.distances.items():
        parsed = _parse_res(res)
        if parsed and parsed[0] in offsets:
            chain, name, seq = parsed
            res = f"{chain}:{name}:{seq + offsets[chain]}"
        key = (lig, res, ctype)
        if key in new_distances:
            prev = new_distances[key]
            if np.isfinite(dist) and (not np.isfinite(prev) or dist < prev):
                new_distances[key] = dist
        else:
            new_distances[key] = dist
        new_contacts.add(key)
    for key in list(new_contacts):
        if key not in new_distances:
            new_distances[key] = np.nan
    return ContactSet(new_contacts, new_distances)


def _remap_contacts(cset: ContactSet, atom_map: AtomMap) -> ContactSet:
    if not atom_map:
        return cset
    new_contacts: Set[Contact] = set()
    new_distances: Dict[Contact, float] = {}
    for (lig, res, ctype), dist in cset.distances.items():
        lig_mapped = atom_map.get(lig, lig)
        key = (lig_mapped, res, ctype)
        if key in new_distances:
            prev = new_distances[key]
            if np.isfinite(dist) and (not np.isfinite(prev) or dist < prev):
                new_distances[key] = dist
        else:
            new_distances[key] = dist
        new_contacts.add(key)
    for key in list(new_contacts):
        if key not in new_distances:
            new_distances[key] = np.nan
    return ContactSet(new_contacts, new_distances)


def _read_rmsd_csv(path: Path | None) -> Dict[str, Dict[str, object]]:
    """Return {pdbid: {'best': row, 'poses': {pose_idx: row}}} from RMSD CSV."""
    data: Dict[str, Dict[str, object]] = {}
    if not path or not path.exists():
        return data
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            pdbid = row.get("pdbid", "").strip()
            if not pdbid:
                continue
            try:
                pose_idx = int(row.get("pose_index") or 1)
                rmsd_val = float(row.get("ligand_rmsd", "nan"))
            except Exception:
                continue
            entry = data.setdefault(pdbid, {"poses": {}})
            entry["poses"][pose_idx] = row
            best = entry.get("best")
            if best is None:
                entry["best"] = row
            else:
                try:
                    if rmsd_val < float(best.get("ligand_rmsd", "inf")):
                        entry["best"] = row
                except Exception:
                    pass
    return data


def compute_metrics(
    project_root: Path,
    ref_path: Path,
    boltz_path: Path,
    vina_path: Path,
    *,
    boltz_rmsd_path: Path | None = None,
    vina_rmsd_path: Path | None = None,
    quiet: bool = True,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ref = _load_contacts(ref_path)
    boltz = _load_contacts(boltz_path)
    vina = _load_contacts(vina_path)

    ref_ids = {p.stem for p in (project_root / "benchmark_references").glob("*.cif")}
    boltz_ids = {p.name.split("_")[0] for p in (project_root / "model_outputs" / "boltz").glob("*.cif")}
    vina_ids = {p.stem for p in (project_root / "model_outputs" / "vina").glob("*.pdbqt")}
    pdbids = sorted(set(ref) & set(boltz) & set(vina) & ref_ids & boltz_ids & vina_ids)

    boltz_rmsd_data = _read_rmsd_csv(boltz_rmsd_path)
    vina_rmsd_data = _read_rmsd_csv(vina_rmsd_path)

    full_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    def _base_row(
        pdbid: str,
        method: str,
        pose_index: int | None,
        rmsd_entry: Dict[str, object] | None = None,
        status: str = "ok",
        error: str = "",
    ) -> Dict[str, object]:
        rmsd_entry = rmsd_entry or {}
        return {
            "pdbid": pdbid,
            "method": method,
            "pose_index": pose_index if pose_index is not None else "",
            "pairing_method": rmsd_entry.get("pairing_method", ""),
            "ligand_heavy_atoms": rmsd_entry.get("ligand_heavy_atoms", ""),
            "ligand_rmsd": rmsd_entry.get("ligand_rmsd", ""),
            "protein_pairs": rmsd_entry.get("protein_pairs", ""),
            "protein_rmsd": rmsd_entry.get("protein_rmsd", ""),
            "status": status,
            "error": error,
        }

    total = len(pdbids)
    for idx, pdbid in enumerate(pdbids, start=1):
        if quiet:
            print(f"\rContact metrics: {idx}/{total}", end="", flush=True)
        else:
            print(f"[INFO] {idx}/{total} {pdbid}")
        ref_pose = ref[pdbid].get(1)
        if not ref_pose:
            continue

        ref_file = project_root / "benchmark_references" / f"{pdbid}.cif"
        boltz_file = project_root / "model_outputs" / "boltz" / f"{pdbid}_model_0.cif"
        vina_file = project_root / "model_outputs" / "vina" / f"{pdbid}.pdbqt"

        # Load reference structure/ligand once per PDB
        try:
            ref_struct = load_structure(ref_file)
            ref_lig = _select_single_ligand(ref_struct, include_h=False)
        except Exception as exc:  # noqa: BLE001
            full_rows.append(_base_row(pdbid, "boltz", 1, status="error", error=f"Ref load failed: {exc}"))
            continue

        boltz_map: AtomMap = {}
        try:
            boltz_struct = load_structure(boltz_file)
            boltz_lig = _select_single_ligand(boltz_struct, include_h=False)
            pairs, _ = _pair_ligand_atoms(boltz_lig, ref_lig)
            boltz_map = {boltz_lig.atoms[i].name: ref_lig.atoms[j].name for i, j in pairs}
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Boltz mapping failed for {pdbid}: {exc}")

        vina_maps: Dict[int, AtomMap] = {}
        try:
            vina_struct = load_structure(vina_file)
            for pose_idx, pose in enumerate(split_models(vina_struct, VINA_MAX_POSES), start=1):
                try:
                    pred_lig = _select_single_ligand(pose, include_h=False)
                    pairs, _ = _pair_ligand_atoms(pred_lig, ref_lig)
                    vina_maps[pose_idx] = {pred_lig.atoms[i].name: ref_lig.atoms[j].name for i, j in pairs}
                except Exception:
                    continue
        except Exception as exc:  # noqa: BLE001
            full_rows.append(_base_row(pdbid, "vina_pose", 1, status="error", error=f"Vina prep failed: {exc}"))
            continue

        vina_contact_sets = vina.get(pdbid, {})

        # Boltz row
        boltz_pose = boltz[pdbid].get(1)
        boltz_best = boltz_rmsd_data.get(pdbid, {}).get("best")
        if boltz_pose and boltz_best and boltz_map:
            boltz_norm = _align_boltz_residues(ref_pose, _remap_contacts(boltz_pose, boltz_map))
            row = {
                **_base_row(pdbid, "boltz", 1, boltz_best),
                **_metrics_strict(ref_pose, boltz_norm),
                **_metrics_residue(ref_pose, boltz_norm),
            }
            full_rows.append(row)
            summary_rows.append({**row, "method": "boltz"})
        else:
            full_rows.append(_base_row(pdbid, "boltz", 1, status="error", error="Boltz contacts or RMSD missing"))

        # Vina per pose
        vina_ok_rows: List[Dict[str, object]] = []
        for pose_idx in range(1, VINA_MAX_POSES + 1):
            cset = vina_contact_sets.get(pose_idx)
            rmsd_entry = vina_rmsd_data.get(pdbid, {}).get("poses", {}).get(pose_idx)
            amap = vina_maps.get(pose_idx)
            if not cset or not rmsd_entry or not amap:
                full_rows.append(
                    _base_row(
                        pdbid,
                        "vina_pose",
                        pose_idx,
                        status="error",
                        error="missing contacts/rmsd/mapping",
                    )
                )
                continue
            norm = _remap_contacts(cset, amap)
            row = {
                **_base_row(pdbid, "vina_pose", pose_idx, rmsd_entry),
                **_metrics_strict(ref_pose, norm),
                **_metrics_residue(ref_pose, norm),
            }
            vina_ok_rows.append(row)
            full_rows.append(row)

        if vina_ok_rows:
            best_row = min(vina_ok_rows, key=lambda r: float(r["ligand_rmsd"]))
            summary_rows.append({**best_row, "method": "vina_best"})

            numeric_cols = [
                "ligand_heavy_atoms",
                "ligand_rmsd",
                "protein_pairs",
                "protein_rmsd",
                "precision",
                "recall",
                "f1",
                "jaccard",
                "distance_mae",
                "ref_contacts",
                "pred_contacts",
                "shared_contacts",
                "residue_precision",
                "residue_recall",
                "residue_f1",
                "residue_jaccard",
                "residue_shared",
                "residue_ref_size",
                "residue_pred_size",
            ]
            median_row = {**_base_row(pdbid, "vina_median", None, rmsd_entry=None)}
            for col in numeric_cols:
                vals = [float(r[col]) for r in vina_ok_rows if r.get(col) not in ("", None)]
                vals = [v for v in vals if not np.isnan(v)]
                median_row[col] = float(np.nanmedian(vals)) if vals else ""
            # Carry non-numeric informative fields
            median_row["status"] = "ok"
            median_row["error"] = ""
            summary_rows.append(median_row)

    if quiet:
        print()
    return full_rows, summary_rows


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute contact-level metrics vs reference.")
    p.add_argument("--ref", default="analysis/pandamap_contacts/ref_contacts.csv", help="Reference contacts CSV.")
    p.add_argument("--boltz", default="analysis/pandamap_contacts/boltz_contacts.csv", help="Boltz contacts CSV.")
    p.add_argument("--vina", default="analysis/pandamap_contacts/vina_contacts.csv", help="Vina contacts CSV.")
    p.add_argument("--per-pose-out", help="Per-pose output CSV (default: analysis/contact_metrics_allposes_<timestamp>.csv)")
    p.add_argument("--summary-out", help="Summary output CSV (default: analysis/contact_metrics_summary_<timestamp>.csv)")
    p.add_argument("--boltz-rmsd", help="Boltz RMSD CSV (from measure_ligand_pose_batch).")
    p.add_argument("--vina-rmsd", help="Vina RMSD CSV (from measure_ligand_pose_batch).")
    p.add_argument("--quiet", action="store_true", default=True, help="Compact progress; suppress per-entry logs.")
    p.add_argument("--no-quiet", action="store_false", dest="quiet", help="Disable quiet mode.")
    args = p.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    ref_path = (project_root / args.ref).resolve()
    boltz_path = (project_root / args.boltz).resolve()
    vina_path = (project_root / args.vina).resolve()
    for label, path in [("ref", ref_path), ("boltz", boltz_path), ("vina", vina_path)]:
        if not path.exists():
            print(f"[ERROR] {label} contacts file not found: {path}")
            return 2

    boltz_rmsd_path = Path(args.boltz_rmsd).resolve() if args.boltz_rmsd else None
    vina_rmsd_path = Path(args.vina_rmsd).resolve() if args.vina_rmsd else None

    full_rows, summary_rows = compute_metrics(
        project_root,
        ref_path,
        boltz_path,
        vina_path,
        boltz_rmsd_path=boltz_rmsd_path,
        vina_rmsd_path=vina_rmsd_path,
        quiet=args.quiet,
    )
    if not full_rows:
        print("[ERROR] No metrics computed.")
        return 2

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_pose_path = (
        (project_root / args.per_pose_out).resolve()
        if args.per_pose_out
        else (project_root / "analysis" / f"contact_metrics_allposes_{ts}.csv")
    )
    summary_path = (
        (project_root / args.summary_out).resolve()
        if args.summary_out
        else (project_root / "analysis" / f"contact_metrics_summary_{ts}.csv")
    )

    per_pose_fields = [
        "pdbid",
        "method",
        "pose_index",
        "pairing_method",
        "ligand_heavy_atoms",
        "ligand_rmsd",
        "protein_pairs",
        "protein_rmsd",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "distance_mae",
        "ref_contacts",
        "pred_contacts",
        "shared_contacts",
        "residue_precision",
        "residue_recall",
        "residue_f1",
        "residue_jaccard",
        "residue_shared",
        "residue_ref_size",
        "residue_pred_size",
        "status",
        "error",
    ]

    summary_fields = [f for f in per_pose_fields if f not in ("status", "error")]

    per_pose_path.parent.mkdir(parents=True, exist_ok=True)
    with per_pose_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_pose_fields)
        writer.writeheader()
        writer.writerows(full_rows)

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            row = {k: v for k, v in row.items() if k in summary_fields}
            writer.writerow(row)

    print(f"[INFO] Wrote per-pose metrics to {per_pose_path}")
    print(f"[INFO] Wrote summary metrics to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
