#!/usr/bin/env python3
"""
Pose Benchmark – protein+ligand (Boltz vs PDB)

V17 — Name-based ligand pairing + altLoc dedupe + case-agnostic CIF
  • Prefer **by-name** heavy-atom pairing (ChimeraX-like) when ≥3 common names.
  • Fall back to **chimera-order** (file order) or **fixed-rank** as needed.
  • Deduplicate ligand altLocs by keeping the highest-occupancy atom per name.
  • Reference CIF discovery remains case-agnostic (filename & extension).
  • Keeps V16 outputs: fixed-rank rows added under --full, CSV keys mirror summary.json.

Core behavior
  • Protein fit: Chimera-style pruning (cutoff 2.0 Å; iteratively drop min(10% of kept,
    50% of kept beyond cutoff), refit until none > cutoff). Returns pruned transform.
  • Ligands: **LOCKED** RMSD only (no ligand refit). Pairing policy = by-name →
    chimera-order → fixed-rank-fallback. We report locked_global & locked_pocket.
"""
from __future__ import annotations

import argparse
import json
import logging as log
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Iterable, Set

import numpy as np
from scipy.optimize import linear_sum_assignment

import gemmi
from Bio.Align import PairwiseAligner

INF = 1e9
AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'SEC':'U','PYL':'O'
}
WAT_NAMES = {"HOH","H2O","WAT"}
ELEM_ORDER = {e:i for i,e in enumerate(['C','N','O','P','S','F','CL','BR','I'])}
MIN_LIG_HEAVY = 10

# -------------------- small utilities --------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    return R, t


def apply_rt_to_structure(st: gemmi.Structure, R: np.ndarray, t: np.ndarray) -> None:
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    v = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                    v2 = v @ R + t
                    atom.pos = gemmi.Position(float(v2[0]), float(v2[1]), float(v2[2]))


def is_protein_res(res: gemmi.Residue) -> bool:
    nm = res.name.upper()
    return (nm not in WAT_NAMES) and (nm in AA3_TO_1)


def is_water_res(res: gemmi.Residue) -> bool:
    return res.name.upper() in WAT_NAMES


# -------------------- IO helpers --------------------

def load_structure(path: Path) -> gemmi.Structure:
    st = gemmi.read_structure(str(path))
    st.remove_empty_chains()
    return st


def write_pdb_structure(st: gemmi.Structure, path: Path) -> None:
    lines: List[str] = []
    serial = 1
    for model_index, model in enumerate(st, start=1):
        lines.append(f"MODEL     {model_index}")
        for ch in model:
            for r in ch:
                resname = r.name
                chain = ch.name[:1] if ch.name else 'A'
                seqid = r.seqid.num if hasattr(r.seqid, 'num') else int(str(r.seqid).strip() or 0)
                icode = r.seqid.icode if hasattr(r.seqid, 'icode') else ' '
                record = 'ATOM  ' if is_protein_res(r) else 'HETATM'
                for a in r:
                    name = a.name.strip()
                    aname = name.rjust(4) if len(a.element.name.strip()) == 1 and len(name) < 4 else name.ljust(4)
                    x, y, z = a.pos.x, a.pos.y, a.pos.z
                    occ, b = 1.00, 0.00
                    elem = a.element.name.strip().rjust(2)
                    lines.append(
                        f"{record}{serial:5d} {aname}{resname:>3s} {chain:1s}{seqid:4d}{icode:1s}   "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          {elem:>2s}"
                    )
                    serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def write_transformed_structures(st: gemmi.Structure, out_prefix: str, outdir: Path) -> Tuple[Path, Path]:
    pdb_path = outdir / f'pred_transformed_{out_prefix}.pdb'
    write_pdb_structure(st, pdb_path)
    cif_path = outdir / f'pred_transformed_{out_prefix}.cif'
    try:
        if hasattr(st, 'make_mmcif_document'):
            doc = st.make_mmcif_document()
            doc.write_file(str(cif_path))
            log.info('Wrote transformed structure (CIF): %s', cif_path)
        else:
            raise AttributeError('Structure.make_mmcif_document not available')
    except Exception as e:
        log.warning('CIF write failed (%s). PDB written at %s. Continuing.', e, pdb_path)
    return cif_path, pdb_path


# -------------------- sequence & chain pairing --------------------
@dataclass
class ChainSeq:
    chain: gemmi.Chain
    seq: str
    ca_xyz: List[np.ndarray]
    res_objs: List[gemmi.Residue]


def _get_ca(res: gemmi.Residue):
    for a in res:
        if a.name.strip().upper() == 'CA':
            return a
    return None


def extract_chain_sequences(st: gemmi.Structure) -> List[ChainSeq]:
    out: List[ChainSeq] = []
    model = st[0]
    for ch in model:
        res_list = [r for r in ch if is_protein_res(r)]
        if not res_list:
            continue
        seq = ''.join(AA3_TO_1.get(r.name.upper(), 'X') for r in res_list)
        ca = []
        for r in res_list:
            a = _get_ca(r)
            ca.append(None if a is None else np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float))
        out.append(ChainSeq(chain=ch, seq=seq, ca_xyz=ca, res_objs=res_list))
    return out


@dataclass
class ChainPair:
    pred: ChainSeq
    ref: ChainSeq
    res_pairs: List[Tuple[int, int]]


def identity_and_pairs(a: ChainSeq, b: ChainSeq) -> Tuple[float, List[Tuple[int,int]]]:
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -1
    aligner.extend_gap_score = -0.5
    aln = aligner.align(a.seq, b.seq)[0]
    A, B = aln.aligned[0], aln.aligned[1]
    pairs: List[Tuple[int,int]] = []
    matches = 0
    aligned_len = 0
    for (a0,a1), (b0,b1) in zip(A,B):
        for ia, ib in zip(range(a0,a1), range(b0,b1)):
            ra, rb = a.res_objs[ia], b.res_objs[ib]
            if AA3_TO_1.get(ra.name.upper(),'X') == AA3_TO_1.get(rb.name.upper(),'X'):
                matches += 1
            aligned_len += 1
            pairs.append((ia, ib))
    ident = (matches / aligned_len) if aligned_len else 0.0
    return ident, pairs


def pair_chains(pred: List[ChainSeq], ref: List[ChainSeq]) -> List[ChainPair]:
    if not pred or not ref:
        return []
    M = np.full((len(pred), len(ref)), 1.0, dtype=float)
    all_pairs: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
    for i, a in enumerate(pred):
        for j, b in enumerate(ref):
            ident, pairs = identity_and_pairs(a,b)
            M[i,j] = 1.0 - ident
            all_pairs[(i,j)] = pairs
    ri, cj = linear_sum_assignment(M)
    out = []
    for i,j in zip(ri,cj):
        pairs = all_pairs[(i,j)]
        if pairs:
            out.append(ChainPair(pred[i], ref[j], pairs))
    return out


# -------------------- Chimera-style pruning --------------------
@dataclass
class FitResult:
    R: np.ndarray
    t: np.ndarray
    rmsd_pruned: float
    n_pruned: int
    rmsd_all_under_pruned: float
    n_all: int
    rmsd_allfit: float
    kept_mask: np.ndarray | None = None


def chimera_pruned_fit(P: np.ndarray, Q: np.ndarray, cutoff: float = 2.0, full: bool=False) -> FitResult:
    n = P.shape[0]
    if n < 3:
        raise RuntimeError('Not enough pairs for fitting')
    # all-fit baseline (always)
    R0, t0 = kabsch(P, Q)
    rmsd_allfit = float(np.sqrt(((P@R0 + t0 - Q)**2).sum()/n))

    keep = np.ones(n, dtype=bool)
    R = np.eye(3); t = np.zeros(3)
    for _ in range(200):
        idx_keep = np.where(keep)[0]
        if idx_keep.size < 3:
            break
        Pk, Qk = P[idx_keep], Q[idx_keep]
        R, t = kabsch(Pk, Qk)
        d_keep = np.sqrt(((P[idx_keep]@R + t - Q[idx_keep])**2).sum(axis=1))
        above_mask = d_keep > cutoff
        a = int(np.count_nonzero(above_mask))
        if a == 0:
            break
        k10 = max(1, int(math.ceil(0.10 * idx_keep.size)))
        k50 = max(1, int(math.ceil(0.50 * a)))
        if k50 <= k10:
            cand_local = np.where(above_mask)[0]
            worst_local = np.argsort(d_keep[cand_local])[-k50:]
            to_drop_local = cand_local[worst_local]
        else:
            worst_local = np.argsort(d_keep)[-k10:]
            to_drop_local = worst_local
        to_drop_global = idx_keep[to_drop_local]
        keep[to_drop_global] = False

    idx_keep = np.where(keep)[0]
    Pk, Qk = P[idx_keep], Q[idx_keep]
    if Pk.shape[0] >= 3:
        R, t = kabsch(Pk, Qk)
    rmsd_pruned = float(np.sqrt(((Pk@R + t - Qk)**2).sum()/max(Pk.shape[0],1)))
    rmsd_all_under_pruned = float(np.sqrt(((P@R + t - Q)**2).sum()/n))

    return FitResult(R=R, t=t,
                     rmsd_pruned=rmsd_pruned, n_pruned=idx_keep.size,
                     rmsd_all_under_pruned=rmsd_all_under_pruned, n_all=n,
                     rmsd_allfit=rmsd_allfit,
                     kept_mask=keep if full else None)


# -------------------- ligands --------------------
@dataclass
class SimpleAtom:
    name: str
    element: str
    xyz: np.ndarray

@dataclass
class SimpleResidue:
    chain_id: str
    res_name: str
    res_id: str
    atoms: List[SimpleAtom]


def collect_ligands(st: gemmi.Structure, include_h: bool=False) -> List[SimpleResidue]:
    """Collect ligands (non-protein, non-water). Deduplicate altLocs by highest occupancy
    per atom name; ignore H unless include_h=True."""
    ligs: List[SimpleResidue] = []
    for model in st:
        for ch in model:
            for r in ch:
                if is_protein_res(r) or is_water_res(r):
                    continue
                best: Dict[str, Tuple[SimpleAtom, float]] = {}
                for a in r:
                    el = a.element.name.upper()
                    if (el == 'H') and (not include_h):
                        continue
                    name_raw = a.name.strip()
                    key = name_raw.upper()
                    xyz = np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float)
                    occ = getattr(a, 'occ', 1.0)
                    sa = SimpleAtom(name=name_raw, element=el, xyz=xyz)
                    if key not in best or occ > best[key][1]:
                        best[key] = (sa, occ)
                atoms = [v[0] for v in best.values()]
                if atoms:
                    ligs.append(SimpleResidue(ch.name, r.name, str(r.seqid), atoms))
    return ligs


def heavy_count(res: SimpleResidue) -> int:
    return sum(1 for a in res.atoms if a.element != 'H')

_name_num_tail = re.compile(r"^(?P<base>[A-Z]+?)(?P<num>\d+)(?P<trail>[A-Z])?$")


def _rank_key(atom: SimpleAtom) -> Tuple[int, int, str]:
    el_ord = ELEM_ORDER.get(atom.element, 50)
    m = _name_num_tail.match(atom.name.upper().strip())
    num = int(m.group('num')) if m else 10**6
    return (el_ord, num, atom.name.upper())


def pairs_chimera_order(pred: SimpleResidue, ref: SimpleResidue) -> List[Tuple[int,int]]:
    P = [i for i,a in enumerate(pred.atoms) if a.element != 'H']
    R = [j for j,a in enumerate(ref.atoms) if a.element != 'H']
    n = min(len(P), len(R))
    return [(P[k], R[k]) for k in range(n)]


def pairs_fixed_rank(pred: SimpleResidue, ref: SimpleResidue) -> List[Tuple[int,int]]:
    P = sorted(range(len(pred.atoms)), key=lambda i: _rank_key(pred.atoms[i]))
    R = sorted(range(len(ref.atoms)), key=lambda j: _rank_key(ref.atoms[j]))
    n = min(len(P), len(R))
    return [(P[k], R[k]) for k in range(n)]


def pairs_by_name(pred: SimpleResidue, ref: SimpleResidue) -> List[Tuple[int,int]]:
    """Pair heavy atoms with the same atom NAME (case-insensitive)."""
    pmap = {a.name.upper(): i for i, a in enumerate(pred.atoms) if a.element != 'H'}
    rmap = {a.name.upper(): j for j, a in enumerate(ref.atoms)  if a.element != 'H'}
    common = sorted(set(pmap.keys()) & set(rmap.keys()))
    return [(pmap[nm], rmap[nm]) for nm in common]


def locked_rmsd(P_atoms: List[SimpleAtom], R_atoms: List[SimpleAtom], pairs: List[Tuple[int,int]], R: np.ndarray, t: np.ndarray) -> Tuple[float,int]:
    if len(pairs) < 3:
        return math.inf, 0
    P = np.array([P_atoms[i].xyz for i,_ in pairs], dtype=float)
    Q = np.array([R_atoms[j].xyz for _,j in pairs], dtype=float)
    P = P @ R + t
    rmsd = float(np.sqrt(((P-Q)**2).sum() / P.shape[0]))
    return rmsd, P.shape[0]


# pocket selection & local protein-only fit

def residue_key(chain: gemmi.Chain, res: gemmi.Residue) -> Tuple[str,str]:
    return (chain.name, str(res.seqid))


def ca_of(res: gemmi.Residue) -> np.ndarray | None:
    for a in res:
        if a.name.strip().upper() == 'CA':
            return np.array([a.pos.x,a.pos.y,a.pos.z], dtype=float)
    return None


def find_pocket_keys(ref_st: gemmi.Structure, ref_lig: SimpleResidue, radius: float) -> Set[Tuple[str,str]]:
    lig_xyz = np.array([a.xyz for a in ref_lig.atoms], dtype=float)
    keys: Set[Tuple[str,str]] = set()
    for model in ref_st:
        for ch in model:
            for r in ch:
                if not is_protein_res(r):
                    continue
                ca = _get_ca(r)
                if ca is None:
                    continue
                # precise: any heavy atom within radius
                close = False
                for a in r:
                    if a.element.name.upper() == 'H':
                        continue
                    apos = np.array([a.pos.x, a.pos.y, a.pos.z])
                    if np.min(np.linalg.norm(lig_xyz - apos, axis=1)) <= radius:
                        close = True
                        break
                if close:
                    keys.add(residue_key(ch, r))
    return keys


def build_ref_to_pred_resmap(chain_pairs: List[ChainPair]) -> Dict[Tuple[str,str], Tuple[str,str,gemmi.Residue,gemmi.Residue]]:
    m: Dict[Tuple[str,str], Tuple[str,str,gemmi.Residue,gemmi.Residue]] = {}
    for cp in chain_pairs:
        for ia, ib in cp.res_pairs:
            rref = cp.ref.res_objs[ib]
            rpred = cp.pred.res_objs[ia]
            m[(cp.ref.chain.name, str(rref.seqid))] = (cp.pred.chain.name, str(rpred.seqid), rpred, rref)
    return m


def local_pocket_fit(pred_st: gemmi.Structure, ref_st: gemmi.Structure, chain_pairs: List[ChainPair], ref_lig: SimpleResidue, radius: float) -> Tuple[np.ndarray,np.ndarray,int]:
    keys = find_pocket_keys(ref_st, ref_lig, radius)
    ref_to_pred = build_ref_to_pred_resmap(chain_pairs)
    P: List[np.ndarray] = []
    Q: List[np.ndarray] = []
    for key in keys:
        if key not in ref_to_pred:
            continue
        _, _, rpred, rref = ref_to_pred[key]
        c_pred = ca_of(rpred)
        c_ref  = ca_of(rref)
        if c_pred is None or c_ref is None:
            continue
        P.append(c_pred)
        Q.append(c_ref)
    if len(P) < 3:
        return np.eye(3), np.zeros(3), 0
    Pm = np.stack(P, axis=0)
    Qm = np.stack(Q, axis=0)
    Rl, tl = kabsch(Pm, Qm)
    return Rl, tl, len(P)


# -------------------- paths --------------------
@dataclass
class Paths:
    root: Path
    analysis_dir: Path
    data_dir: Path
    pred: Path
    ref: Path
    pdbid_up: str


def _resolve_ref_path_case_agnostic(base: Path, pdbid: str, ref: str|None) -> Path:
    """
    If --ref is provided, honor it.
    Otherwise, look under raw_structures/monomers/<PDBID>/ for a .cif file:
      1) Prefer a file whose stem matches the PDB ID case-insensitively.
      2) Otherwise use the first .cif found (case-insensitive extension).
      3) If nothing exists yet, fall back to <pdbid.lower()>.cif in that dir.
    """
    if ref:
        return Path(ref)
    dir_path = base / f"raw_structures/monomers/{pdbid.upper()}"
    if dir_path.exists():
        cands = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() == ".cif"]
        if cands:
            target = pdbid.lower()
            exact = [p for p in cands if p.stem.lower() == target]
            if exact:
                return exact[0]
            # deterministic choice if multiple
            return sorted(cands)[0]
    return dir_path / f"{pdbid.lower()}.cif"


def infer_paths(base: Path, pdbid: str, pred: str|None, ref: str|None) -> Paths:
    pdbid_up = pdbid.upper()
    pred_p = Path(pred) if pred else base / f"model_outputs/{pdbid_up}_output/boltz_results_{pdbid_up}/predictions/{pdbid_up}/{pdbid_up}_model_0.cif"
    ref_p  = _resolve_ref_path_case_agnostic(base, pdbid, ref)
    root   = base / f"analysis/{pdbid_up}"
    analysis_dir = root / f"{pdbid_up}_analysis"
    data_dir     = root / f"{pdbid_up}_data"
    return Paths(root=root, analysis_dir=analysis_dir, data_dir=data_dir, pred=pred_p, ref=ref_p, pdbid_up=pdbid_up)


# -------------------- main --------------------

def main(argv: Sequence[str]|None=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark Boltz poses vs PDB with LOCKED ligand RMSD and Chimera-style protein fit")
    p.add_argument('pdbid', help='PDB ID (e.g., 1HMS)')
    p.add_argument('--pred', help='Path to predicted CIF/PDB', default=None)
    p.add_argument('--ref', help='Path to reference mmCIF/PDB', default=None)
    p.add_argument('--include-h', action='store_true', help='Include hydrogens (default: heavy atoms only)')
    p.add_argument('--include-small', action='store_true', help=f'Include small ligands (<{MIN_LIG_HEAVY} heavy atoms). Default skips them.')
    p.add_argument('--no-pocket', action='store_true', help='Disable pocket-local alignment (report global frame only)')
    p.add_argument('--pocket-radius', type=float, default=5.0, help='Å radius for pocket Cα alignment')
    p.add_argument('--full', action='store_true', help='Add fixed-rank comparison and per-pair protein distances to data.csv')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = p.parse_args(argv)

    # Simple log format: no timestamps
    log.basicConfig(level=log.DEBUG if args.verbose else log.INFO, format='[%(levelname)s] %(message)s')

    base = Path.home() / 'lipid_docking_benchmark'
    paths = infer_paths(base, args.pdbid, args.pred, args.ref)
    ensure_dir(paths.root)
    ensure_dir(paths.analysis_dir)
    ensure_dir(paths.data_dir)

    log.info('Output dir: %s', paths.root)
    log.info('Reference file: %s', paths.ref)
    log.info('Reading structures…')
    ref_st = load_structure(paths.ref)
    pred_st = load_structure(paths.pred)

    # ---- protein global fit with Chimera-style pruning ----
    pred_ch = extract_chain_sequences(pred_st)
    ref_ch  = extract_chain_sequences(ref_st)
    chain_pairs = pair_chains(pred_ch, ref_ch)

    Pxyz: List[np.ndarray] = []
    Qxyz: List[np.ndarray] = []
    for cp in chain_pairs:
        for ia, ib in cp.res_pairs:
            a = cp.pred.ca_xyz[ia]; b = cp.ref.ca_xyz[ib]
            if a is None or b is None:
                continue
            Pxyz.append(a); Qxyz.append(b)
    if len(Pxyz) < 3:
        raise RuntimeError('Not enough Cα pairs to align proteins')
    P = np.stack(Pxyz, axis=0); Q = np.stack(Qxyz, axis=0)

    fit = chimera_pruned_fit(P, Q, cutoff=2.0, full=args.full)
    log.info('Protein Cα RMSD (pruned=%d): %.3f Å', fit.n_pruned, fit.rmsd_pruned)
    log.info('Protein Cα RMSD (all under pruned): %.3f Å across %d pairs', fit.rmsd_all_under_pruned, fit.n_all)
    log.info('Protein Cα RMSD (all-fit baseline): %.3f Å', fit.rmsd_allfit)

    # move predicted into reference frame using the pruned transform
    apply_rt_to_structure(pred_st, fit.R, fit.t)
    write_transformed_structures(pred_st, 'global', paths.data_dir)

    # ---- ligands ----
    pred_ligs_all = collect_ligands(pred_st, include_h=args.include_h)
    ref_ligs_all  = collect_ligands(ref_st, include_h=args.include_h)

    if args.include_small:
        pred_ligs = pred_ligs_all
        ref_ligs  = ref_ligs_all
    else:
        pred_ligs = [r for r in pred_ligs_all if heavy_count(r) >= MIN_LIG_HEAVY]
        ref_ligs  = [r for r in ref_ligs_all  if heavy_count(r) >= MIN_LIG_HEAVY]
        log.info('Skipping small ligands (<%d heavy atoms): kept %d/%d pred, %d/%d ref',
                 MIN_LIG_HEAVY, len(pred_ligs), len(pred_ligs_all), len(ref_ligs), len(ref_ligs_all))

    log.info('Ref ligands to evaluate: %d; Pred candidates: %d', len(ref_ligs), len(pred_ligs))

    def choose_pairs(rp: SimpleResidue, rr: SimpleResidue) -> Tuple[str, List[Tuple[int,int]]]:
        # 1) Try exact NAME matching (ChimeraX semantics)
        pn = pairs_by_name(rp, rr)
        if len(pn) >= 3:
            return 'by-name', pn
        # 2) Fall back to file-order (legacy)
        po = pairs_chimera_order(rp, rr)
        if len(po) >= 3:
            return 'chimera-order', po
        # 3) Last resort fixed-rank
        pf = pairs_fixed_rank(rp, rr)
        return 'fixed-rank-fallback', pf

    # Hungarian assignment cost on by-name locked_global (fallbacks where needed)
    cache_pairs: List[List[Tuple[str, List[Tuple[int,int]]]]] = [[('none',[]) for _ in ref_ligs] for _ in pred_ligs]
    M = [[INF]*len(ref_ligs) for _ in range(len(pred_ligs))]
    for i, rp in enumerate(pred_ligs):
        for j, rr in enumerate(ref_ligs):
            pol, pr = choose_pairs(rp, rr)
            cache_pairs[i][j] = (pol, pr)
            if len(pr) >= 3:
                lg_cost, _ = locked_rmsd(rp.atoms, rr.atoms, pr, np.eye(3), np.zeros(3))
                if math.isfinite(lg_cost):
                    M[i][j] = lg_cost
    if len(pred_ligs) and len(ref_ligs):
        ri, cj = linear_sum_assignment(np.array(M, dtype=float))
        matches = [(i,j) for i,j in zip(ri,cj) if M[i][j] < INF/10]
    else:
        matches = []

    # CSVs (analysis + data)
    # Header uses *exact* summary.json key names where applicable
    analysis_rows = [
        'type,pdbid,protein_pairs_pruned,protein_rmsd_ca_pruned,protein_pairs_all,protein_rmsd_ca_all_under_pruned,protein_rmsd_ca_allfit,'
        'pred_chain,pred_resname,pred_resid,ref_chain,ref_resname,ref_resid,policy,n,rmsd_locked_global,rmsd_locked_pocket,pocket_pairs'
    ]
    analysis_rows.append(
        f"protein,{paths.pdbid_up},{fit.n_pruned},{fit.rmsd_pruned:.3f},{fit.n_all},{fit.rmsd_all_under_pruned:.3f},{fit.rmsd_allfit:.3f},,,,,,,,,,,"
    )

    data_rows = ['type,pair_index,pdbid,pred_chain,pred_resname,pred_resid,ref_chain,ref_resname,ref_resid,policy,n,locked_global,locked_pocket,pocket_pairs,protein_metric,protein_value']
    if args.full and fit.kept_mask is not None:
        d_all = np.sqrt(((P@fit.R + fit.t - Q)**2).sum(axis=1))
        for i_idx, d in enumerate(d_all):
            kept = 1 if fit.kept_mask[i_idx] else 0
            data_rows.append(f"protein_pair,{i_idx},{paths.pdbid_up},,,,,,,,,,,kept,{kept}")
            data_rows.append(f"protein_pair,{i_idx},{paths.pdbid_up},,,,,,,,,,,dist_A,{d:.3f}")

    best_metrics: List[Dict] = []

    for i,j in matches:
        rp, rr = pred_ligs[i], ref_ligs[j]
        policy, pairs = cache_pairs[i][j]
        lg, n = locked_rmsd(rp.atoms, rr.atoms, pairs, np.eye(3), np.zeros(3))
        pocket_pairs = 0
        lp = lg
        Rl = np.eye(3); tl = np.zeros(3)
        if not args.no_pocket:
            Rl, tl, pocket_pairs = local_pocket_fit(pred_st, ref_st, chain_pairs, rr, args.pocket_radius)
            if pocket_pairs >= 3:
                lp, _ = locked_rmsd(rp.atoms, rr.atoms, pairs, Rl, tl)
        # analysis row for the primary policy
        analysis_rows.append(
            f"ligand,{paths.pdbid_up},{fit.n_pruned},{fit.rmsd_pruned:.3f},{fit.n_all},{fit.rmsd_all_under_pruned:.3f},{fit.rmsd_allfit:.3f},"
            f"{rp.chain_id},{rp.res_name},{rp.res_id},{rr.chain_id},{rr.res_name},{rr.res_id},{policy},{n},{lg:.3f},{lp:.3f},{pocket_pairs}"
        )
        # data row for the primary policy
        data_rows.append(
            f"ligand,,{paths.pdbid_up},{rp.chain_id},{rp.res_name},{rp.res_id},{rr.chain_id},{rr.res_name},{rr.res_id},{policy},{n},{lg:.3f},{lp:.3f},{pocket_pairs},,"
        )
        best_metrics.append({
            'pred': {'chain': rp.chain_id, 'name': rp.res_name, 'id': rp.res_id},
            'ref' : {'chain': rr.chain_id, 'name': rr.res_name, 'id': rr.res_id},
            'policy': policy,
            'n': n,
            'rmsd_locked_global': lg,
            'rmsd_locked_pocket': lp,
            'pocket_pairs': pocket_pairs,
        })

        if policy == 'by-name':
            log.info('Ligand %s:%s (%s) ↔ %s:%s (%s) — by-name locked_global=%.3f Å, locked_pocket=%.3f Å (n=%d, pocket_pairs=%d)',
                     rp.chain_id, rp.res_name, rp.res_id, rr.chain_id, rr.res_name, rr.res_id, lg, lp, n, pocket_pairs)
        elif policy == 'chimera-order':
            log.info('Ligand %s:%s (%s) ↔ %s:%s (%s) — chimera-order locked_global=%.3f Å, locked_pocket=%.3f Å (n=%d, pocket_pairs=%d)',
                     rp.chain_id, rp.res_name, rp.res_id, rr.chain_id, rr.res_name, rr.res_id, lg, lp, n, pocket_pairs)
        elif policy == 'fixed-rank-fallback':
            log.error('By-name and chimera-order pairing failed (n<3) for %s:%s (%s) vs %s:%s (%s); using fixed-rank fallback (n=%d).',
                      rp.chain_id, rp.res_name, rp.res_id, rr.chain_id, rr.res_name, rr.res_id, n)

        # When --full, also compute and record the fixed-rank comparison (if primary was not fallback)
        if args.full and policy != 'fixed-rank-fallback':
            pr2 = pairs_fixed_rank(rp, rr)
            if len(pr2) >= 3:
                lg2, n2 = locked_rmsd(rp.atoms, rr.atoms, pr2, np.eye(3), np.zeros(3))
                lp2 = lg2
                if not args.no_pocket and pocket_pairs >= 3:
                    lp2, _ = locked_rmsd(rp.atoms, rr.atoms, pr2, Rl, tl)
                # add a side-by-side fixed-rank row to *both* CSVs
                analysis_rows.append(
                    f"ligand,{paths.pdbid_up},{fit.n_pruned},{fit.rmsd_pruned:.3f},{fit.n_all},{fit.rmsd_all_under_pruned:.3f},{fit.rmsd_allfit:.3f},"
                    f"{rp.chain_id},{rp.res_name},{rp.res_id},{rr.chain_id},{rr.res_name},{rr.res_id},fixed-rank,{n2},{lg2:.3f},{lp2:.3f},{pocket_pairs}"
                )
                data_rows.append(
                    f"ligand,,{paths.pdbid_up},{rp.chain_id},{rp.res_name},{rp.res_id},{rr.chain_id},{rr.res_name},{rr.res_id},fixed-rank,{n2},{lg2:.3f},{lp2:.3f},{pocket_pairs},,"
                )

    # write CSVs
    analysis_csv = paths.analysis_dir / f"{paths.pdbid_up}_analysis.csv"
    (analysis_csv).write_text("\n".join(analysis_rows) + "\n")

    data_csv = paths.data_dir / f"{paths.pdbid_up}_data.csv"
    (data_csv).write_text("\n".join(data_rows) + "\n")

    # summary json
    summary = {
        'protein_rmsd_ca_pruned': fit.rmsd_pruned,
        'protein_pairs_pruned': fit.n_pruned,
        'protein_rmsd_ca_all_under_pruned': fit.rmsd_all_under_pruned,
        'protein_pairs_all': fit.n_all,
        'protein_rmsd_ca_allfit': fit.rmsd_allfit,
        'ligand_best_metrics': best_metrics,
    }
    (paths.analysis_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    log.info('Wrote %s', paths.analysis_dir/'summary.json')

    # pocket preview into DATA dir
    if not args.no_pocket and ref_ligs:
        Rl, tl, npairs = local_pocket_fit(pred_st, ref_st, chain_pairs, ref_ligs[0], args.pocket_radius)
        if npairs >= 3:
            apply_rt_to_structure(pred_st, Rl, tl)
            write_transformed_structures(pred_st, 'pocket', paths.data_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())

