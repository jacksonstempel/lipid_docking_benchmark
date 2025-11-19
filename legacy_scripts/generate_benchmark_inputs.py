from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import gemmi


def read_accepted_table(xlsx_path: Path) -> pd.DataFrame:
    if xlsx_path.suffix.lower() == ".xlsx" and xlsx_path.exists():
        try:
            return pd.read_excel(xlsx_path, sheet_name="accepted")
        except Exception:
            pass
    # CSV fallbacks
    acc_csv = xlsx_path.with_suffix(".accepted.csv")
    if acc_csv.exists():
        return pd.read_csv(acc_csv)
    # If a plain CSV exists at the path, read it
    if xlsx_path.exists() and xlsx_path.suffix.lower() == ".csv":
        return pd.read_csv(xlsx_path)
    raise FileNotFoundError(f"Could not load accepted table from {xlsx_path} or CSV fallback")


def extract_sequence_from_cif(cif_path: Path) -> str:
    st = gemmi.read_structure(str(cif_path))
    # Take first model, first polymer chain
    for model in st:
        for chain in model:
            pol = chain.get_polymer()
            if pol:
                letters = []
                for res in pol:
                    name = res.name.upper().strip()
                    try:
                        one = gemmi.find_tabulated_residue(name).one_letter_code
                    except Exception:
                        one = "X"
                    if not one:
                        one = "X"
                    letters.append(one.upper())
                return "".join(letters)
        break
    raise ValueError(f"No polymer chain found in {cif_path}")


def load_existing_benchmark_ids(bm_dir: Path) -> set[str]:
    return {p.stem.upper() for p in bm_dir.glob("*.yaml")}


def write_yaml(out_path: Path, sequence: str, ligand_ccd: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Emit minimal YAML matching repo style
    content = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {sequence}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      ccd: {ligand_ccd}\n"
    )
    out_path.write_text(content)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Boltz benchmark YAMLs for accepted lipid targets")
    ap.add_argument("--accepted", type=Path, default=Path("docs/lipid_candidates.xlsx"), help="Path to accepted table (.xlsx or CSV)")
    ap.add_argument("--cifs", type=Path, default=Path("raw_structures/pdb_lipid_candidates"), help="Directory containing accepted CIFs")
    ap.add_argument("--benchmark-existing", type=Path, default=Path("model_inputs/benchmark_inputs"), help="Dir of existing benchmark YAMLs to exclude duplicates")
    ap.add_argument("--out", type=Path, default=Path("model_inputs/benchmark_expansion"), help="Output directory for new YAMLs")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    df = read_accepted_table(args.accepted)
    if "pdb_id" not in df.columns or "lipid_comp_id" not in df.columns:
        raise ValueError("Accepted table must contain columns: pdb_id, lipid_comp_id")

    existing = load_existing_benchmark_ids(args.benchmark_existing)
    created = 0
    skipped = 0
    missing = 0

    for _, row in df.iterrows():
        pid = str(row["pdb_id"]).upper().strip()
        if not pid or len(pid) < 4:
            continue
        if pid in existing:
            skipped += 1
            if args.verbose:
                print(f"Skip {pid}: already in benchmark_inputs")
            continue
        ccd = str(row["lipid_comp_id"]).upper().strip()
        cif_path = args.cifs / f"{pid}.cif"
        if not cif_path.exists():
            missing += 1
            if args.verbose:
                print(f"Missing CIF for {pid}: {cif_path}")
            continue
        try:
            seq = extract_sequence_from_cif(cif_path)
        except Exception as e:
            if args.verbose:
                print(f"Error extracting sequence for {pid}: {e}")
            continue
        out_path = args.out / f"{pid}.yaml"
        write_yaml(out_path, seq, ccd)
        created += 1
        if args.verbose:
            print(f"Wrote {out_path}")

    print(f"Done. created={created} skipped_existing={skipped} missing_cif={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

