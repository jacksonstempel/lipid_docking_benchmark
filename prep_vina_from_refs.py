#!/usr/bin/env python3
"""
Prep Vina inputs from benchmark references.

For each <PDBID>.cif in <refs_dir>:
  - Extract the receptor without non-protein ligands (waters/ions optional).
  - Pick a single "primary" ligand (largest non-polymer, excluding HOH/ions).
  - Write:
      docking/<PDBID>/prep/receptor_no_ligand.pdb
      docking/<PDBID>/prep/receptor.pdbqt
      docking/<PDBID>/prep/ligand.pdb
      docking/<PDBID>/prep/ligand.pdbqt
  - Compute a Vina box (default: ligand bbox + padding) and write:
      docking/vina/box/<PDBID>.txt
    Format: center_x center_y center_z size_x size_y size_z

Requirements:
  - Python packages: gemmi, numpy
  - CLI tool: Open Babel `obabel` on PATH (for PDB→PDBQT)

Usage:
  python scripts/prep_vina_from_refs.py \
    --refs ~/lipid_docking_benchmark/raw_structures/benchmark_references \
    --repo ~/lipid_docking_benchmark \
    --pad 6.0 \
    --keep-ions \
    --no-keep-waters

Box modes:
  --box-mode ligand      (default) box from primary ligand atoms + padding
  --box-mode residues    box from residues listed in a YAML/CSV (see --residue-file)
  --box-mode fpocket     box from fpocket top pocket (requires 'fpocket' in PATH)
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import gemmi

ION_NAMES = {"NA","K","CL","MG","MN","ZN","CA","FE","CU","CO","NI","CD","SR","CS","AL"}
WATER_NAMES = {"HOH","WAT","H2O"}

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.stdout

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def is_polymer_res(res: gemmi.Residue) -> bool:
    # Treat standard AA/NA as polymer; everything else as hetero
    return res.het_flag == gemmi.AtomFlag.NOT_HETATM

def pick_primary_ligand(model: gemmi.Model):
    """Heuristic: choose the non-polymer residue with the most heavy atoms, excluding waters/ions."""
    best = None
    best_atoms = -1
    for chain in model:
        for res in chain:
            if res.het_flag != gemmi.AtomFlag.HETATM:
                continue
            rname = res.name.strip().upper()
            if rname in WATER_NAMES or rname in ION_NAMES:
                continue
            heavy = sum(1 for a in res if a.element.name != "H")
            if heavy > best_atoms:
                best_atoms = heavy
                best = (chain.name, res)
    return best  # (chain_id, residue)

def residue_coords(res: gemmi.Residue):
    return np.array([ [a.pos.x, a.pos.y, a.pos.z] for a in res if a.element.name != "H" ], dtype=float)

def write_receptor_no_ligand(struct: gemmi.Structure, out_pdb: Path, keep_waters: bool, keep_ions: bool):
    st = struct.clone()
    for model in st:
        for chain in list(model):
            for res in list(chain):
                if res.het_flag == gemmi.AtomFlag.HETATM:
                    rname = res.name.strip().upper()
                    if keep_waters and rname in WATER_NAMES:
                        continue
                    if keep_ions and rname in ION_NAMES:
                        continue
                    # drop all other hetero residues (ligands, etc.)
                    chain.remove_residue(res)
    st.remove_empty_chains()
    st.remove_empty_models()
    st.write_minimal_pdb(str(out_pdb))

def write_single_residue_pdb(model: gemmi.Model, chain_id: str, res: gemmi.Residue, out_pdb: Path):
    st = gemmi.Structure()
    st.cell = model.get_cell()
    new_model = gemmi.Model("A")
    new_chain = gemmi.Chain(chain_id)
    new_res = gemmi.Residue()
    new_res.name = res.name
    new_res.seqid = res.seqid
    for a in res:
        new_res.add_atom(a)
    new_chain.add_residue(new_res)
    new_model.add_chain(new_chain)
    st.add_model(new_model)
    st.write_minimal_pdb(str(out_pdb))

def pdb_to_pdbqt(in_pdb: Path, out_pdbqt: Path, ph: float, is_ligand: bool):
    # Open Babel flags: add hydrogens at pH, compute Gasteiger charges
    # For ligands, --gen3d helps if coordinates are odd; we skip it if input has 3D
    cmd = ["obabel", str(in_pdb), "-O", str(out_pdbqt), "-p", f"{ph}", "--partialcharge", "gasteiger"]
    if is_ligand:
        # only generate 3D if no Z-coordinates variance (flat)
        try:
            with open(in_pdb) as f:
                zs = [float(line[46:54]) for line in f if line.startswith(("ATOM","HETATM"))]
            if len(zs) > 3 and np.std(zs) < 1e-3:
                cmd.insert(3, "--gen3d")
        except Exception:
            cmd.insert(3, "--gen3d")
    run(cmd)

def ligand_box(coords: np.ndarray, pad: float):
    mins, maxs = coords.min(0), coords.max(0)
    center = coords.mean(0)
    size = (maxs - mins) + pad
    return center, size

def fpocket_box(receptor_pdb: Path):
    # Run fpocket, take top pocket center/size (approx via alpha sphere cluster)
    out = run(["fpocket", "-f", str(receptor_pdb)])
    # fpocket writes a dir receptor_pdb_out/ with pockets; read pocket0 info
    out_dir = receptor_pdb.with_suffix("").as_posix() + "_out"
    pocket0 = Path(out_dir) / "pockets" / "pocket0_vert.pqr"
    if not pocket0.exists():
        raise RuntimeError("fpocket did not produce pocket0_vert.pqr")
    # Compute center/size from pocket vertices as a rough box
    coords=[]
    with open(pocket0) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
                coords.append((x,y,z))
    c=np.array(coords); mins, maxs = c.min(0), c.max(0)
    center=c.mean(0); size=(maxs-mins)
    return center, size

def residues_box(receptor_pdb: Path, residue_file: Path, pad: float):
    # residue_file supports CSV with columns: chain,resnum  OR YAML list of {chain,resnum}
    coords=[]
    if residue_file.suffix.lower() in {".yaml",".yml"}:
        import yaml
        data = yaml.safe_load(open(residue_file))
        res_list = [(d["chain"], str(d["resnum"])) for d in data]
    else:
        res_list=[]
        with open(residue_file) as f:
            reader = csv.DictReader(f)
            if "chain" in reader.fieldnames and "resnum" in reader.fieldnames:
                for row in reader:
                    res_list.append((row["chain"], str(row["resnum"])))
            else:
                raise ValueError("CSV must have headers: chain,resnum")
    st = gemmi.read_structure(str(receptor_pdb))
    for ch, rn in res_list:
        try:
            res = st[0][ch].get_residue(rn)
        except Exception:
            continue
        for a in res:
            if a.element.name != "H":
                coords.append([a.pos.x, a.pos.y, a.pos.z])
    c=np.array(coords)
    mins, maxs = c.min(0), c.max(0)
    center=c.mean(0); size=(maxs-mins)+pad
    return center, size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="Directory with reference .cif files")
    ap.add_argument("--repo", required=True, help="Repo root (e.g., ~/lipid_docking_benchmark)")
    ap.add_argument("--pad", type=float, default=6.0, help="Padding (Å) added to bbox sizes")
    ap.add_argument("--ph", type=float, default=7.4, help="Protonation pH for PDBQT conversion")
    ap.add_argument("--box-mode", choices=["ligand","residues","fpocket"], default="ligand")
    ap.add_argument("--residue-file", help="CSV or YAML with residues for --box-mode residues")
    ap.add_argument("--keep-waters", action="store_true", default=False)
    ap.add_argument("--keep-ions", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None, help="Only process first N CIFs (for quick tests)")
    args = ap.parse_args()

    refs_dir = Path(os.path.expanduser(args.refs))
    repo = Path(os.path.expanduser(args.repo))

    # sanity: obabel available?
    try:
        obv = run(["obabel", "-V"]).strip()
        print(f"[ok] Open Babel: {obv}")
    except Exception as e:
        print("[error] Open Babel (obabel) not found on PATH. Install via `mamba install -c conda-forge openbabel`.")
        sys.exit(1)

    cif_paths = sorted(refs_dir.glob("*.cif"))
    if args.limit:
        cif_paths = cif_paths[:args.limit]
    if not cif_paths:
        print(f"[warn] No .cif files found under {refs_dir}")
        return

    box_dir = ensure_dir(repo / "docking" / "vina" / "box")

    for cif in cif_paths:
        pdbid = cif.stem.upper()
        print(f"\n=== {pdbid} ===")
        # per-target dirs
        tdir = ensure_dir(repo / "docking" / pdbid / "prep")
        # read structure
        st = gemmi.read_structure(str(cif))
        st.remove_alternative_conformations()
        model = st[0]

        # pick primary ligand
        pick = pick_primary_ligand(model)
        if pick is None:
            print(f"[skip] {pdbid}: no suitable ligand found (only waters/ions?).")
            continue
        chain_id, lig_res = pick
        lig_coords = residue_coords(lig_res)
        print(f"[info] ligand: {lig_res.name} in chain {chain_id} with {lig_coords.shape[0]} heavy atoms")

        # write receptor without ligands
        rec_pdb = tdir / "receptor_no_ligand.pdb"
        write_receptor_no_ligand(st, rec_pdb, keep_waters=args.keep_waters, keep_ions=args.keep_ions)

        # write ligand-only PDB
        lig_pdb = tdir / "ligand.pdb"
        write_single_residue_pdb(model, chain_id, lig_res, lig_pdb)

        # convert to PDBQT
        rec_pdbqt = tdir / "receptor.pdbqt"
        lig_pdbqt = tdir / "ligand.pdbqt"
        pdb_to_pdbqt(rec_pdb, rec_pdbqt, ph=args.ph, is_ligand=False)
        pdb_to_pdbqt(lig_pdb, lig_pdbqt, ph=args.ph, is_ligand=True)

        # compute box
        if args.box_mode == "ligand":
            center, size = ligand_box(lig_coords, args.pad)
        elif args.box_mode == "fpocket":
            center, size = fpocket_box(rec_pdb)
        else:
            if not args.residue_file:
                raise ValueError("--residue-file required for --box-mode residues")
            center, size = residues_box(rec_pdb, Path(args.residue_file), args.pad)

        cx, cy, cz = map(float, center)
        sx, sy, sz = map(float, size)
        # guard against zero/negative sizes
        eps = 8.0  # minimum practical box edge for Vina
        sx = max(sx, eps); sy = max(sy, eps); sz = max(sz, eps)

        box_file = box_dir / f"{pdbid}.txt"
        with open(box_file, "w") as f:
            f.write(f"{cx:.3f} {cy:.3f} {cz:.3f} {sx:.3f} {sy:.3f} {sz:.3f}\n")

        # write a small manifest for reproducibility
        manifest = {
            "pdbid": pdbid,
            "source_cif": str(cif),
            "receptor_pdb": str(rec_pdb),
            "ligand_pdb": str(lig_pdb),
            "receptor_pdbqt": str(rec_pdbqt),
            "ligand_pdbqt": str(lig_pdbqt),
            "box_mode": args.box_mode,
            "box_file": str(box_file),
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "size": [round(sx,3), round(sy,3), round(sz,3)],
            "ph": args.ph,
            "keep_waters": args.keep_waters,
            "keep_ions": args.keep_ions,
        }
        with open(tdir / "run_manifest.json", "w") as mf:
            json.dump(manifest, mf, indent=2)

        print(f"[ok] Wrote box → {box_file}")
        print(f"[ok] Wrote PDBQT → {rec_pdbqt.name}, {lig_pdbqt.name}")

if __name__ == "__main__":
    main()
