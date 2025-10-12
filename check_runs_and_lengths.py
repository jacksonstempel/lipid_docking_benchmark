#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import re
from typing import Iterable, List, Set, Tuple

from scripts.lib.config import load_config
from scripts.lib.paths import PathResolver


def load_outputs_index(outputs_dir: Path) -> Set[str]:
    """Return lowercase PDB IDs that have been run (strip a trailing '_output')."""
    ran: Set[str] = set()
    if not outputs_dir.is_dir():
        return ran
    for entry in outputs_dir.iterdir():
        if entry.is_dir():
            key = re.sub(r'_output$', '', entry.name, flags=re.IGNORECASE).lower()
            ran.add(key)
    return ran


def seq_len_from_yaml(path: Path) -> int | None:
    """
    Extract sequence length from a YAML file without external deps.
    Handles:
      sequence: GSHEVL...
      sequence: "GSHEVL..."
      sequence: | / >
        GSHEV...
    Counts letters only (A–Z, case-insensitive).
    """
    lines = path.read_text(encoding='utf-8').splitlines()

    # find the 'sequence:' line
    for i, line in enumerate(lines):
        m = re.match(r'^(\s*)sequence\s*:\s*(.*)$', line, flags=re.IGNORECASE)
        if not m:
            continue
        indent = len(m.group(1).expandtabs(2))  # measure indentation
        rest = m.group(2).strip()

        # inline scalar
        if rest and rest not in {'|', '>', '|-', '>-'}:
            # strip surrounding quotes if present
            if (rest.startswith('"') and rest.endswith('"')) or (rest.startswith("'") and rest.endswith("'")):
                rest = rest[1:-1]
            seq = re.findall(r'[A-Za-z]', rest)
            return len(seq)

        # block scalar: collect lines more indented than 'sequence:'
        seq_chunks: List[str] = []
        for nxt in lines[i + 1 :]:
            if nxt.strip() == '':
                continue
            lead = len(re.match(r'^\s*', nxt).group(0).expandtabs(2))
            if lead <= indent:
                break
            # append content, not indentation
            seq_chunks.append(nxt.strip())
        joined = ''.join(seq_chunks)
        letters = re.findall(r'[A-Za-z]', joined)
        return len(letters)

    return None  # no 'sequence:' key found


def iter_yaml_paths(inputs_dir: Path) -> List[Path]:
    return sorted(inputs_dir.glob("*.y*ml"))


def write_report(path: Path, rows: Iterable[Tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as out:
        out.write('# Proteins not yet run and their sequence lengths\n')
        out.write('# source: model_inputs/benchmark_inputs; checked against model_outputs/\n')
        out.write('PDB_ID\tSEQ_LEN\n')
        for pdb, slen in rows:
            out.write(f'{pdb}\t{slen}\n')


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report benchmark YAMLs that have not produced model outputs yet."
    )
    parser.add_argument("--config", default=None, help="Path to config file (default: config.yaml)")
    parser.add_argument("--inputs-dir", default=None, help="Directory with benchmark YAMLs (default from config)")
    parser.add_argument("--preds", default=None, help="Override predictions root directory")
    parser.add_argument("--report", default=None, help="Output report path (default from config)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args(argv)

    config = load_config(args.config)
    resolver = PathResolver(config, preds=args.preds)

    inputs_dir = (
        Path(args.inputs_dir).expanduser().resolve()
        if args.inputs_dir
        else config.paths.benchmark_inputs
    )
    outputs_dir = resolver.preds_root
    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else config.reports.unrun_proteins
    )

    if args.verbose:
        print(f"[INFO] Inputs dir:   {inputs_dir}")
        print(f"[INFO] Outputs dir:  {outputs_dir}")
        print(f"[INFO] Report path:  {report_path}")

    ran_ids = load_outputs_index(outputs_dir)
    yaml_paths = iter_yaml_paths(inputs_dir)
    if not yaml_paths:
        print(f'No YAML files found in {inputs_dir}')
        return 0

    missing: List[Tuple[str, int]] = []
    for yp in yaml_paths:
        pdb = yp.stem.upper()
        if pdb.lower() in ran_ids:
            continue
        slen = seq_len_from_yaml(yp)
        if slen is None:
            slen = 0
        missing.append((pdb, slen))

    write_report(report_path, missing)

    print(f'Wrote {len(missing)} entries to {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
