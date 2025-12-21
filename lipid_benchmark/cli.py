"""
Command-line entry point for the lipid docking benchmark.

Plain-language overview

- Reads a “pairs CSV”: a simple spreadsheet-like table listing, for each target (PDB ID),
  the experimental structure file and the prediction files to evaluate.
- Runs the benchmark pipeline (RMSD + contact metrics) using library code in
  `lipid_benchmark/pipeline.py`.
- Writes two CSV outputs (per-pose and per-target summary) under `analysis/` by default.

If you are a user, you typically run this via `python scripts/benchmark.py`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .io import default_pairs_path, find_project_root, read_pairs_csv, write_csv
from .pipeline import BENCHMARK_FIELDNAMES, run_benchmark


def _default_cache_root(project_root: Path) -> Path:
    """
    Return the repo-local cache directory.

    This is where we keep generated artifacts that are useful for speed/debugging but are
    not “final outputs” a user would typically publish (e.g., normalized PDBs, cached
    contact extraction).
    """
    return project_root / ".cache" / "lipid_benchmark"


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    This defines the “knobs” a user can change without editing code, such as:
    - which pairs CSV to use
    - where to write outputs
    - how many Vina poses to consider
    - whether to use caching and/or multiple CPU processes
    """
    p = argparse.ArgumentParser(description="Run the lipid docking benchmark.")
    p.add_argument("--pairs", help="Pairs CSV (default: config.yaml paths.pairs, else scripts/pairs_curated.csv).")
    p.add_argument("--out-dir", default="analysis/benchmark", help="Output directory (default: analysis/benchmark).")
    p.add_argument(
        "--cache-dir",
        default="",
        help="Cache directory for normalized PDBs and cached contacts (default: .cache/lipid_benchmark).",
    )
    p.add_argument("--vina-max-poses", type=int, default=20, help="Upper bound Vina poses to evaluate (default: 20).")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1).")
    p.add_argument("--no-cache-normalized", action="store_true", help="Disable cached normalized complexes.")
    p.add_argument("--no-cache-contacts", action="store_true", help="Disable cached PandaMap contacts.")
    p.add_argument("--quiet", action="store_true", default=True, help="Compact progress output.")
    p.add_argument("--no-quiet", action="store_false", dest="quiet", help="Verbose progress output.")
    return p


def _resolve_out_dir(project_root: Path, out_dir: str) -> Path:
    """
    Turn a user-provided output directory into an absolute path.

    - If the user passes a relative path like `analysis/benchmark`, it is interpreted relative
      to the repository root.
    - If the user passes an absolute path, it is used as-is.
    """
    path = Path(out_dir).expanduser()
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def main(argv: list[str] | None = None) -> int:
    """
    Run the benchmark from the command line and write output CSVs.

    What happens, at a high level:

    1) Find the repository root (so relative paths are stable).
    2) Load the pairs CSV and resolve the files listed in it.
    3) Run the benchmark for each target.
    4) Write `benchmark_allposes.csv` and `benchmark_summary.csv` to the output directory.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="[%(levelname)s] %(message)s")

    project_root = find_project_root()
    if args.pairs:
        p = Path(args.pairs).expanduser()
        pairs_path = (project_root / p).resolve() if not p.is_absolute() else p.resolve()
    else:
        pairs_path = default_pairs_path(project_root)
    entries = read_pairs_csv(project_root, pairs_path)

    out_dir = _resolve_out_dir(project_root, str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(args.cache_dir).strip():
        cache_dir_arg = Path(str(args.cache_dir)).expanduser()
        cache_root = (project_root / cache_dir_arg).resolve() if not cache_dir_arg.is_absolute() else cache_dir_arg.resolve()
    else:
        cache_root = _default_cache_root(project_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    normalized_dir = cache_root / "normalized"
    allposes, summary = run_benchmark(
        entries,
        vina_max_poses=int(args.vina_max_poses),
        normalized_dir=normalized_dir,
        quiet=bool(args.quiet),
        workers=int(args.workers),
        cache_normalized=not bool(args.no_cache_normalized),
        cache_contacts=not bool(args.no_cache_contacts),
    )

    write_csv(out_dir / "benchmark_allposes.csv", allposes, BENCHMARK_FIELDNAMES)
    write_csv(out_dir / "benchmark_summary.csv", summary, BENCHMARK_FIELDNAMES)
    return 0
