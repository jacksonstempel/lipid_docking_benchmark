#!/usr/bin/env python3
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import gemmi

ION_NAMES   = {"NA","K","CL","MG","MN","ZN","CA","FE","CU","CO","NI","CD","SR","CS","AL"}
WATER_NAMES = {"HOH","WAT","H2O"}

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}")
    return p.stdout

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True); return p

def pick_primary_ligand(model: gemmi.Model):
    best = None; best_atoms = -1
    for chain in model:
        for res in chain:
            if res.het_flag != "H":  # polymer? skip
                continue
            rname = res.name.strip().upper()
            if rname in WATER_NAMES or rname in ION_NAMES:
                continue
            heavy = sum(1 for a in res if a.element.name != "H")
            if heavy > best_atoms:
                best_atoms = heavy; best = (chain.name, res)
    return best  # (chain_id, residue)

def residue_coords(res: gemmi.Residue):
    return np.array([[a.pos.x,a.pos.y,a.pos.z] for a in res if a.element.name!="H"], float)

def write_receptor_no_ligand(struct: gemmi.Structure, out_pdb: Path, keep_waters: bool, keep_ions: bool):
    st = gemmi.Structure()
    st.cell = struct.cell
    st.spacegroup_hm = struct.spacegroup_hm

    for model_idx, src_model in enumerate(struct):
        if model_idx > 0:
            break  # Vina expects a single rigid receptor model
        model_copy = gemmi.Model(src_model.num)
        for src_chain in src_model:
            chain_copy = gemmi.Chain(src_chain.name)
            for res in src_chain:
                rname = res.name.strip().upper()
                if res.het_flag == "H":
                    if keep_waters and rname in WATER_NAMES:
                        pass
                    elif keep_ions and rname in ION_NAMES:
                        pass
                    else:
                        continue

                res_copy = gemmi.Residue()
                res_copy.name = res.name
                res_copy.seqid = res.seqid
                res_copy.het_flag = res.het_flag
                for atom in res:
                    res_copy.add_atom(atom.clone())
                chain_copy.add_residue(res_copy)
            if len(chain_copy):
                model_copy.add_chain(chain_copy)
        if len(model_copy):
            st.add_model(model_copy)

    st.write_minimal_pdb(str(out_pdb))

def write_single_residue_pdb(struct: gemmi.Structure, chain_id: str, res: gemmi.Residue, out_pdb: Path):
    st = gemmi.Structure()
    st.cell = struct.cell
    st.spacegroup_hm = struct.spacegroup_hm
    model = gemmi.Model(1)
    chain = gemmi.Chain(chain_id)
    res_copy = gemmi.Residue()
    res_copy.name = res.name
    res_copy.seqid = res.seqid
    res_copy.het_flag = res.het_flag
    for atom in res:
        res_copy.add_atom(atom.clone())
    chain.add_residue(res_copy)
    model.add_chain(chain)
    st.add_model(model)
    st.write_minimal_pdb(str(out_pdb))

def pdb_to_pdbqt(in_pdb: Path, out_pdbqt: Path, ph: float, is_ligand: bool):
    cmd = ["obabel", str(in_pdb), "-O", str(out_pdbqt)]
    if is_ligand:
        # generate 3D only if coords are effectively 2D
        need_gen3d = False
        try:
            zs = [float(line[46:54]) for line in open(in_pdb) if line.startswith(("ATOM","HETATM"))]
            if len(zs) > 3 and np.std(zs) < 1e-3:
                need_gen3d = True
        except Exception:
            need_gen3d = True
        if need_gen3d:
            cmd.append("--gen3d")
    else:
        cmd.append("-xr")  # receptor mode: keep rigid, no torsion tree
    cmd.extend(["-p", f"{ph}", "--partialcharge", "gasteiger"])
    run(cmd)

def ligand_box(coords: np.ndarray, pad: float):
    mins, maxs = coords.min(0), coords.max(0)
    center = coords.mean(0)
    size   = (maxs - mins) + pad
    return center, size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="Directory with reference .cif files")
    ap.add_argument("--repo", required=True, help="Repo root (e.g., ~/lipid_docking_benchmark)")
    ap.add_argument("--pad", type=float, default=6.0, help="Padding (Å) added to bbox sizes")
    ap.add_argument("--ph",  type=float, default=7.4, help="Protonation pH for PDBQT conversion")
    ap.add_argument("--keep-waters", action="store_true", default=False)
    ap.add_argument("--keep-ions",   action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    refs_dir = Path(os.path.expanduser(args.refs))
    repo     = Path(os.path.expanduser(args.repo))

    # sanity: Open Babel
    try:
        print(f"[ok] Open Babel:", run(["obabel","-V"]).strip())
    except Exception:
        print("[error] obabel not found. Install: mamba install -c conda-forge openbabel"); sys.exit(1)

    cif_paths = sorted(refs_dir.glob("*.cif"))
    if args.limit: cif_paths = cif_paths[:args.limit]
    if not cif_paths:
        print(f"[warn] No .cif files in {refs_dir}"); return

    # Output roots (UPDATED LAYOUT)
    box_dir = ensure_dir(repo / "docking" / "vina" / "box")
    prep_root = ensure_dir(repo / "docking" / "prep")

    for cif in cif_paths:
        pdbid = cif.stem.upper()
        print(f"\n=== {pdbid} ===")
        tprep = ensure_dir(prep_root / pdbid)

        st = gemmi.read_structure(str(cif))
        st.remove_alternative_conformations()
        model = st[0]

        # choose primary ligand
        pick = pick_primary_ligand(model)
        if pick is None:
            print(f"[skip] {pdbid}: no non-ion/non-water ligand found."); continue
        chain_id, lig_res = pick
        lig_coords = residue_coords(lig_res)
        print(f"[info] ligand {lig_res.name} (chain {chain_id}) heavy atoms: {lig_coords.shape[0]}")

        # write receptor without ligands
        rec_pdb = tprep / "receptor_no_ligand.pdb"
        write_receptor_no_ligand(st, rec_pdb, keep_waters=args.keep_waters, keep_ions=args.keep_ions)

        # write ligand-only PDB
        lig_pdb = tprep / "ligand.pdb"
        write_single_residue_pdb(st, chain_id, lig_res, lig_pdb)

        # convert to PDBQT
        rec_pdbqt = tprep / "receptor.pdbqt"
        lig_pdbqt = tprep / "ligand.pdbqt"
        pdb_to_pdbqt(rec_pdb, rec_pdbqt, ph=args.ph, is_ligand=False)
        pdb_to_pdbqt(lig_pdb, lig_pdbqt, ph=args.ph, is_ligand=True)

        # compute ligand-based box (simple, first-pass)
        center, size = ligand_box(lig_coords, args.pad)
        cx,cy,cz = map(float, center)
        sx,sy,sz = map(float, size)
        eps = 8.0
        sx,sy,sz = max(sx,eps), max(sy,eps), max(sz,eps)

        box_file = box_dir / f"{pdbid}.txt"
        with open(box_file, "w") as f:
            f.write(f"{cx:.3f} {cy:.3f} {cz:.3f} {sx:.3f} {sy:.3f} {sz:.3f}\n")

        manifest = {
            "pdbid": pdbid,
            "source_cif": str(cif),
            "prep_dir": str(tprep),
            "receptor_pdb": str(rec_pdb),
            "ligand_pdb": str(lig_pdb),
            "receptor_pdbqt": str(rec_pdbqt),
            "ligand_pdbqt": str(lig_pdbqt),
            "box_file": str(box_file),
            "center": [round(cx,3), round(cy,3), round(cz,3)],
            "size":   [round(sx,3), round(sy,3), round(sz,3)],
            "ph": args.ph,
            "keep_waters": args.keep_waters,
            "keep_ions": args.keep_ions,
        }
        with open(tprep / "run_manifest.json","w") as mf:
            json.dump(manifest, mf, indent=2)

        print(f"[ok] prep → {tprep}")
        print(f"[ok] box  → {box_file}")

if __name__ == "__main__":
    main()
