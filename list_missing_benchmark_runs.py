#!/usr/bin/env python3
"""
List benchmark YAMLs that have NOT been run yet, with sequence lengths.

What this script does
---------------------
* Searches for YAML specs under the configured benchmark inputs directory.
* Derives PDB ID from the YAML filename (e.g., 1S8G.yaml → 1S8G)
* Checks if a prediction CIF exists under: <preds>/<ID>_output/**
  looking specifically for:                <ID>_model_0.cif  (case-insensitive)
* Reports YAMLs whose predictions are **missing**, along with a best-guess
  sequence length parsed from the YAML (handles common keys and FASTA refs).
* Optionally writes the report to a CSV.

Usage
-----
Basic (uses paths from config.yaml by default):
    python list_missing_benchmark_runs.py -v

Custom locations and CSV output:
    python list_missing_benchmark_runs.py \
        --inputs-dir /path/to/model_inputs/benchmark_inputs \
        --preds /path/to/model_outputs \
        --csv /path/to/output/missing_benchmark_runs.csv -v

Flags
-----
- --config        Path to config.yaml (default: project root/config.yaml)
- --inputs-dir    Directory containing benchmark YAMLs (default from config)
- --preds         Directory that contains <ID>_output trees (default from config)
- --csv           Optional path to write a CSV report
- --all           List ALL YAMLs with status and sequence length (not just missing)
- -v/--verbose    Chatty progress logs

CSV columns
-----------
  pdbid, yaml_path, status, seq_len, pred_cif_found, pred_cif_path
Where status is one of: MISSING (no pred), PRESENT (pred exists), ERROR (YAML unreadable)

Notes
-----
* Requires PyYAML. If you see an error about missing yaml, install with:
      pip install pyyaml
* Sequence length is pulled from common YAML keys like 'sequence' or from a
  referenced FASTA ('fasta', 'fasta_path', etc.). If multiple sequences are
  found, the **longest** is reported (typical for multi-chain inputs).
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import re
from typing import Any, Iterable, List, Optional, Tuple

from scripts.lib.config import load_config
from scripts.lib.paths import PathResolver

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None


# ------------------------------- File discovery -------------------------------

def list_yaml_files(inputs_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not inputs_dir.exists():
        return out
    for p in sorted(inputs_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".yml", ".yaml"}:
            out.append(p)
    return out


def pdbid_from_filename(path: Path) -> str:
    return path.stem.upper()


# ----------------------------- Prediction lookup -----------------------------

def find_prediction_cif(outputs_root: Path, pdbid: str) -> Optional[Path]:
    """Search <outputs_root>/<ID>_output/** for <ID>_model_0.cif (case-insensitive)."""
    id_lower = pdbid.lower()
    root = outputs_root / f"{pdbid}_output"
    if not root.exists():
        return None
    preferred = f"{id_lower}_model_0.cif"
    found_any: Optional[Path] = None
    try:
        for p in root.rglob("*.cif"):
            name = p.name.lower()
            if not name.endswith("_model_0.cif"):
                continue
            if name == preferred:
                return p
            if found_any is None:
                found_any = p
    except Exception:
        pass
    return found_any


# ----------------------------- Sequence extraction ---------------------------
_AA_RE = re.compile(r"[^A-Z]", re.I)
AA_VALID = set("ACDEFGHIKLMNPQRSTVWYBXZOU")  # include uncommon codes and seleno/pyl


def _clean_seq(s: str) -> str:
    # Remove whitespace and non-letters, uppercase
    s2 = _AA_RE.sub("", s.upper())
    # Drop anything not in amino alphabet to avoid long text blobs
    return "".join(ch for ch in s2 if ch in AA_VALID)


def _read_fasta_len(path: Path) -> int:
    if not path.exists():
        return -1
    try:
        seq = []
        for line in path.read_text().splitlines():
            if line.startswith(">"):
                continue
            seq.append(_clean_seq(line))
        s = "".join(seq)
        return len(s) if s else -1
    except Exception:
        return -1


def _extract_seq_lens_from_obj(obj: Any, yaml_dir: Path) -> List[int]:
    lens: List[int] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in {"fasta", "fasta_path", "fa", "sequence_fasta"} and isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    p = (yaml_dir / p).resolve()
                n = _read_fasta_len(p)
                if n > 0:
                    lens.append(n)
            if "seq" in kl and isinstance(v, str):
                s = _clean_seq(v)
                if len(s) >= 1:
                    lens.append(len(s))
            # Recurse
            lens.extend(_extract_seq_lens_from_obj(v, yaml_dir))
    elif isinstance(obj, list):
        for it in obj:
            lens.extend(_extract_seq_lens_from_obj(it, yaml_dir))
    # ignore scalars
    return lens


def sequence_length_from_yaml(yaml_path: Path) -> int:
    if yaml is None:
        return -1
    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception:
        return -1
    lens = _extract_seq_lens_from_obj(data, yaml_path.parent)
    return max(lens) if lens else -1


# ----------------------------------- Main ------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="List benchmark YAMLs that have not produced a prediction CIF yet, with sequence lengths.")
    ap.add_argument("--config", default=None, help="Path to project config (default: config.yaml)")
    ap.add_argument("--inputs-dir", default=None, help="Directory with benchmark YAMLs (default from config)")
    ap.add_argument("--preds", default=None, help="Override predictions root directory")
    ap.add_argument("--outputs-root", default=None, help="Alias for --preds")
    ap.add_argument("--csv", default=None, help="Optional path to write CSV report")
    ap.add_argument("--all", action="store_true", help="List ALL YAMLs with status (not only missing preds)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = ap.parse_args(argv)

    config = load_config(args.config)
    preds_override = args.preds or args.outputs_root
    resolver = PathResolver(config, preds=preds_override)

    inputs_dir = (
        Path(args.inputs_dir).expanduser().resolve()
        if args.inputs_dir
        else config.paths.benchmark_inputs
    )
    outputs_root = resolver.preds_root

    if args.verbose:
        print(f"[INFO] Inputs dir:   {inputs_dir}")
        print(f"[INFO] Outputs root: {outputs_root}")

    yamls = list_yaml_files(inputs_dir)
    if not yamls:
        print(f"[DONE] No YAML files found in {inputs_dir}")
        return 0

    rows: List[Tuple[str, Path, str, int, bool, str]] = []

    for y in yamls:
        pid = pdbid_from_filename(y)
        pred = find_prediction_cif(outputs_root, pid)
        seq_len = sequence_length_from_yaml(y)
        if pred is None:
            status = "MISSING"
            pred_found = False
            pred_path = ""
        else:
            status = "PRESENT"
            pred_found = True
            pred_path = str(pred)
        if args.all or not pred_found:
            rows.append((pid, y, status, seq_len, pred_found, pred_path))
        if args.verbose:
            sl = seq_len if seq_len >= 0 else -1
            print(f"[CHECK] {pid}: status={status}  seq_len={sl}  pred={'yes' if pred_found else 'no'}")

    # Pretty print
    if rows:
        print("\nPDBID    status     seq_len   pred_cif_path")
        print("-"*72)
        for pid, y, status, seq_len, pred_found, pred_path in rows:
            sl = f"{seq_len}" if seq_len >= 0 else "NA"
            path_str = pred_path if pred_found else "(missing)"
            print(f"{pid:<7} {status:<9} {sl:<8} {path_str}")
    else:
        print("[DONE] All YAMLs appear to have predictions present.")

    # CSV
    if args.csv:
        outp = Path(args.csv).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pdbid","yaml_path","status","seq_len","pred_cif_found","pred_cif_path"])
            for pid, y, status, seq_len, pred_found, pred_path in rows:
                w.writerow([pid, str(y), status, seq_len, int(pred_found), pred_path])
        print(f"[OK] Wrote CSV: {outp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
