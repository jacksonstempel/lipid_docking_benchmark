#!/usr/bin/env python3
"""
Minimal, deterministic batch runner for pose_benchmark.py
Outputs a single condensed CSV (one row per protein) into:
  ~/lipid_docking_benchmark/analysis/aggregates/aggregate_{M}.{D}_{H}.{MIN:02d}.csv

Columns (left→right):
  pdbid, protein_rmsd, protein_rmsd_ca_allfit, ligand, policy,
  rmsd_locked_global, rmsd_locked_pocket, n_residues, n_ligand_atoms, n_pocket_residues

Design:
- Deterministic prediction discovery: prefer canonical predictions/<ID>/<ID>_model_0.cif
  else rank by predictions segment → shallower path → newer mtime → alphabetical.
- No external deps besides stdlib. Counts are read from pose_benchmark per‑target CSVs.
- Picks best ligand row by minimum rmsd_locked_global.
"""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from datetime import datetime

# ------------------------ Helpers ------------------------

def _norm_id(s: str) -> str:
    s = s.strip()
    return s.upper() if s else s


def find_ref_cif(pdbid: str, refs_root: Path) -> Optional[Path]:
    """Find a reference CIF under refs_root. Prefer exact stem match; shallower path wins."""
    wanted = pdbid.lower()
    cands: List[Path] = []
    # direct subdir <refs>/<ID>/*.cif
    sub = refs_root / pdbid
    if sub.is_dir():
        cands.extend(sorted(sub.glob('*.cif')))
    # any *.cif with stem == ID
    for p in refs_root.rglob('*.cif'):
        try:
            if p.stem.lower() == wanted:
                cands.append(p)
        except Exception:
            continue
    if not cands:
        return None
    def score(p: Path) -> Tuple[int, str]:
        return (len(p.parts), str(p))
    return sorted(set(cands), key=score)[0]


def find_pred_cif(pdbid: str, preds_root: Path) -> Optional[Path]:
    pid = pdbid.upper()
    canonical = preds_root / f"{pid}_output/boltz_results_{pid}/predictions/{pid}/{pid}_model_0.cif"
    if canonical.is_file():
        return canonical
    base = preds_root / f"{pid}_output"
    if not base.is_dir():
        return None
    matches = [p for p in base.rglob(f"{pid}_model_0.cif") if p.is_file()]
    if not matches:
        return None
    def has_pred_segment(p: Path) -> bool:
        parts = [q.lower() for q in p.parts]
        try:
            i = parts.index('predictions')
            return i + 1 < len(parts) and parts[i+1] == pid.lower()
        except ValueError:
            return False
    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0
    def score(p: Path) -> Tuple[int, int, float, str]:
        return (0 if has_pred_segment(p) else 1, len(p.parts), -mtime(p), str(p))
    return sorted(matches, key=score)[0]


@dataclass
class Target:
    pdbid: str
    ref: Path
    pred: Path


def discover_targets(refs_root: Path, preds_root: Path, ids: Optional[Iterable[str]]) -> List[Target]:
    todo: List[str] = []
    if ids:
        todo = [_norm_id(x) for x in ids if _norm_id(x)]
    else:
        seen: set[str] = set()
        for p in refs_root.rglob('*.cif'):
            if p.stem and len(p.stem) >= 4:
                pid = _norm_id(p.stem)
                if p.stem.lower() == pid.lower():
                    seen.add(pid)
        todo = sorted(seen)
    out: List[Target] = []
    for pid in todo:
        r = find_ref_cif(pid, refs_root)
        p = find_pred_cif(pid, preds_root)
        if r is None:
            print(f"[WARN] Missing ref for {pid}")
            continue
        if p is None:
            print(f"[WARN] Missing pred for {pid}")
            continue
        out.append(Target(pid, r, p))
    return out


# ------------------------ Runner ------------------------

def run_pose(pose_script: Path, t: Target, full: bool, verbose: bool) -> int:
    cmd = [sys.executable, str(pose_script), t.pdbid, '--ref', str(t.ref), '--pred', str(t.pred)]
    if full:
        cmd.append('--full')
    if verbose:
        cmd.append('-v')
    print('[RUN]', ' '.join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(f"[ERROR] pose_benchmark.py not found: {pose_script}")
        return 127


def collect_per_target_csvs(home: Path, ids: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pid in ids:
        p = home / f"lipid_docking_benchmark/analysis/{pid}/{pid}_analysis/{pid}_analysis.csv"
        if p.is_file():
            paths.append(p)
        else:
            print(f"[WARN] Missing analysis CSV for {pid}: {p}")
    return paths


# ------------------------ Condense ------------------------

# preferred header names for each output field
FIELD_MAP = {
    'protein_rmsd': [
        'protein_rmsd', 'protein_rmsd_ca_pruned', 'protein_rmsd_ca_all_under_pruned'
    ],
    'protein_rmsd_ca_allfit': [
        'protein_rmsd_ca_allfit', 'protein_ca_rmsd_allfit'
    ],
    'ligand': [
        'ref_resname','pred_resname','ligand','ligand_name','resname'
    ],
    'policy': [
        'policy','pairing_policy'
    ],
    'rmsd_locked_global': [
        'rmsd_locked_global','ligand_rmsd_locked_global','ligand_rmsd_global'
    ],
    'rmsd_locked_pocket': [
        'rmsd_locked_pocket','ligand_rmsd_locked_pocket','ligand_rmsd_pocket'
    ],
    'n_residues': [
        'protein_pairs_pruned','protein_pairs_all','n_residues','residues_total'
    ],
    'n_ligand_atoms': [
        'n','n_ligand_atoms','ligand_atom_count','n_atoms_ligand'
    ],
    'n_pocket_residues': [
        'pocket_pairs','n_pocket_residues','pocket_residue_count','pocket_residues'
    ],
}


def _get_first(d: dict, keys: List[str]) -> str:
    for k in keys:
        if k in d and d[k] != '':
            return d[k]
    return ''


def _best_ligand(rows: List[dict]) -> Optional[dict]:
    ligs = [r for r in rows if r.get('type','').lower() == 'ligand']
    best = None
    best_val = float('inf')
    for r in ligs:
        val = _get_first(r, FIELD_MAP['rmsd_locked_global'])
        try:
            v = float(val)
        except Exception:
            v = float('inf')
        if v < best_val:
            best_val = v
            best = r
    return best


def write_condensed(per_target_csvs: List[Path], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = [
        'pdbid', 'protein_rmsd', 'protein_rmsd_ca_allfit', 'ligand', 'policy',
        'rmsd_locked_global', 'rmsd_locked_pocket', 'n_residues', 'n_ligand_atoms', 'n_pocket_residues'
    ]
    rows_written = 0
    with dest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for path in per_target_csvs:
            try:
                with path.open('r', newline='') as fin:
                    reader = csv.DictReader(fin)
                    by_id: dict[str, List[dict]] = {}
                    for row in reader:
                        pid = (row.get('pdbid') or row.get('PDBID') or row.get('id') or '').strip()
                        if not pid:
                            continue
                        by_id.setdefault(pid, []).append(row)
                for pid, rows in sorted(by_id.items()):
                    prot = next((r for r in rows if r.get('type','').lower() == 'protein'), None)
                    lig = _best_ligand(rows)
                    out = {'pdbid': pid}
                    # protein metrics
                    if prot:
                        out['protein_rmsd'] = _get_first(prot, FIELD_MAP['protein_rmsd'])
                        out['protein_rmsd_ca_allfit'] = _get_first(prot, FIELD_MAP['protein_rmsd_ca_allfit'])
                        out['n_residues'] = _get_first(prot, FIELD_MAP['n_residues'])
                    else:
                        out['protein_rmsd'] = ''
                        out['protein_rmsd_ca_allfit'] = ''
                        out['n_residues'] = ''
                    # ligand metrics
                    if lig:
                        out['ligand'] = _get_first(lig, FIELD_MAP['ligand'])
                        out['policy'] = _get_first(lig, FIELD_MAP['policy'])
                        out['rmsd_locked_global'] = _get_first(lig, FIELD_MAP['rmsd_locked_global'])
                        out['rmsd_locked_pocket'] = _get_first(lig, FIELD_MAP['rmsd_locked_pocket'])
                        out['n_ligand_atoms'] = _get_first(lig, FIELD_MAP['n_ligand_atoms'])
                        out['n_pocket_residues'] = _get_first(lig, FIELD_MAP['n_pocket_residues'])
                    else:
                        out['ligand'] = ''
                        out['policy'] = ''
                        out['rmsd_locked_global'] = ''
                        out['rmsd_locked_pocket'] = ''
                        out['n_ligand_atoms'] = ''
                        out['n_pocket_residues'] = ''
                    writer.writerow(out)
                    rows_written += 1
            except Exception as e:
                print(f"[WARN] Failed reading {path}: {e}")
    print(f"[OK] Wrote {rows_written} rows → {dest}")


# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description='Minimal deterministic batch wrapper for pose_benchmark.py')
    ap.add_argument('--refs', type=Path, required=True)
    ap.add_argument('--preds', type=Path, required=True)
    ap.add_argument('--pose', type=Path, required=True)
    ap.add_argument('--ids', type=Path, help='Optional text file of PDB IDs (one per line)')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    ids_list: Optional[List[str]] = None
    if args.ids and args.ids.is_file():
        ids_list = [ln.strip() for ln in args.ids.read_text().splitlines() if ln.strip()]

    targets = discover_targets(args.refs, args.preds, ids_list)
    if not targets:
        print('[ERROR] No targets found. Check --refs/--preds and --ids.')
        return 2

    failed: List[str] = []
    for t in targets:
        rc = run_pose(args.pose, t, args.full, args.verbose)
        if rc != 0:
            print(f"[WARN] pose_benchmark failed for {t.pdbid} (rc={rc})")
            failed.append(t.pdbid)

    home = Path.home()
    ts = datetime.now()
    fname = f"aggregate_{ts.month}.{ts.day}_{ts.hour}.{ts.minute:02d}.csv"
    out_path = home / 'lipid_docking_benchmark/analysis/aggregates' / fname

    per_target = collect_per_target_csvs(home, [t.pdbid for t in targets if t.pdbid not in failed])
    write_condensed(per_target, out_path)

    if failed:
        print('[INFO] Failed targets:', ','.join(failed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
