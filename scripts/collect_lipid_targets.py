from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Allow running without package install
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.lib.pdb_search import SearchConfig, search_candidates
from scripts.lib.pdb_filters import ChemCompCache, FilterConfig, evaluate_entry, load_id_list


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect lipid-binding monomeric X-ray structures from RCSB")
    sub = p.add_subparsers(dest="cmd", required=False)

    p.add_argument("--res-max", type=float, default=3.0, help="Maximum resolution (Å)")
    p.add_argument("--tier", choices=["laptop", "cluster"], default="cluster", help="Size tier (residue cap)")
    p.add_argument("--whitelist", type=Path, default=Path("metadata/ligand_whitelist.yaml"), help="Whitelist YAML for non-lipid molecules")
    p.add_argument("--blocklist", type=Path, default=Path("metadata/lipid_blocklist.yaml"), help="Blocklist YAML for detergents/amphiphiles")
    p.add_argument("--out-dir", type=Path, default=Path("raw_structures/pdb_lipid_candidates"), help="Output directory for CIFs")
    p.add_argument("--xlsx", type=Path, default=Path("docs/lipid_candidates.xlsx"), help="Review spreadsheet path")
    p.add_argument("--limit", type=int, default=0, help="Optional hard limit on number of accepted entries")
    p.add_argument("--progress-every", type=int, default=100, help="Emit progress summary every N processed entries")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose: log every rejection as well as accepts")
    p.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2), help="Parallel workers for fetching/filtering")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.xlsx.parent.mkdir(parents=True, exist_ok=True)

    wl = load_id_list(args.whitelist)
    bl = load_id_list(args.blocklist)
    cfg = FilterConfig(res_max=args.res_max, tier=args.tier, whitelist=wl, blocklist=bl)
    search_cfg = SearchConfig(res_max=args.res_max)

    # Always print top-level milestones
    print(f"Searching RCSB (method={search_cfg.method}, res_max={search_cfg.res_max}) ...", flush=True)

    ids = search_candidates(search_cfg)
    print(f"Found {len(ids)} candidate entries with nonpolymers.", flush=True)

    chem_cache = ChemCompCache()
    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    reason_counts: Counter[str] = Counter()

    def worker(pid: str):
        out_cif = args.out_dir / f"{pid}.cif"
        try:
            decision = evaluate_entry(pid, out_cif, cfg, chem_cache)
            return pid, decision, None
        except Exception as e:
            return pid, None, e

    processed = 0
    accepted = 0
    submitted = 0

    # Submit all tasks (or as many as we need when a limit is set)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for pid in ids:
            if args.limit and accepted >= args.limit:
                break
            fut = ex.submit(worker, pid)
            futures[fut] = pid
            submitted += 1

        for fut in as_completed(futures):
            pid = futures[fut]
            i = processed + 1
            decision = None
            err = None
            try:
                pid, decision, err = fut.result()
            except Exception as e:
                err = e
            if err is not None or decision is None:
                rejected_rows.append({"pdb_id": pid, "status": "error", "reason": f"exception: {err}"})
                reason_counts["error"] += 1
                if args.verbose:
                    print(f"[{i}/{len(ids)}] {pid}: error {err}", flush=True)
            else:
                if decision.accept:
                    row = {"pdb_id": pid, "status": decision.reason}
                    row.update(decision.details)
                    accepted_rows.append(row)
                    accepted += 1
                    reason_counts["accepted"] += 1
                    print(
                        f"[{i}/{len(ids)}] {pid}: ACCEPT lipid={decision.details.get('lipid_comp_id')} "
                        f"len={decision.details.get('polymer_length')} res={decision.details.get('resolution')}",
                        flush=True,
                    )
                    if args.limit and accepted >= args.limit:
                        # We reached limit; no early cancellation here, but we stop reporting accepts beyond limit
                        pass
                else:
                    row = {"pdb_id": pid, "status": "rejected", "reason": decision.reason}
                    row.update(decision.details)
                    rejected_rows.append(row)
                    reason_counts[decision.reason] += 1
                    if args.verbose:
                        print(f"[{i}/{len(ids)}] {pid}: reject {decision.reason}", flush=True)

            processed += 1
            if args.progress_every and (processed % args.progress_every == 0):
                rej_total = len(rejected_rows)
                print(
                    f"Progress: processed={processed} accepts={accepted} rejects={rej_total} "
                    f"(top rejections: {', '.join(f'{k}:{v}' for k,v in reason_counts.most_common(4) if k!='accepted')})",
                    flush=True,
                )
            if args.limit and accepted >= args.limit:
                # Drain remaining futures quickly to avoid noisy output; stop reading further
                break

    # Write spreadsheet with accepted entries (primary) and a second sheet with rejections
    try:
        with pd.ExcelWriter(args.xlsx) as xw:
            if accepted_rows:
                df_ok = pd.DataFrame(accepted_rows)
                df_ok.sort_values(["category", "pdb_id"], inplace=True)
                df_ok.to_excel(xw, index=False, sheet_name="accepted")
            df_bad = pd.DataFrame(rejected_rows)
            if not df_bad.empty:
                df_bad.to_excel(xw, index=False, sheet_name="rejected")
        wrote = str(args.xlsx)
    except Exception as e:
        # Fallback to CSVs if Excel backends are unavailable
        base = args.xlsx.with_suffix("")
        ok_csv = base.with_suffix(".accepted.csv")
        bad_csv = base.with_suffix(".rejected.csv")
        df_ok = pd.DataFrame(accepted_rows)
        if not df_ok.empty:
            df_ok.sort_values(["category", "pdb_id"], inplace=True)
            df_ok.to_csv(ok_csv, index=False)
        df_bad = pd.DataFrame(rejected_rows)
        if not df_bad.empty:
            df_bad.to_csv(bad_csv, index=False)
        wrote = f"{ok_csv} / {bad_csv} (CSV fallback)"

    # Final summary
    total = len(accepted_rows) + len(rejected_rows)
    print(
        f"Done. candidates={len(ids)} processed={total} accepted={len(accepted_rows)} "
        f"rejected={len(rejected_rows)}",
        flush=True,
    )
    if reason_counts:
        reasons_str = ", ".join(f"{k}:{v}" for k, v in reason_counts.items() if k != "accepted")
        print(f"Rejection breakdown: {reasons_str}", flush=True)
    print(f"Wrote CIFs to {args.out_dir} and spreadsheet to {wrote}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
