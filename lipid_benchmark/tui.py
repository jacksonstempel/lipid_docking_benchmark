#!/usr/bin/env python3
"""
Interactive text UI (TUI) for running the benchmark.

Plain-language overview

- This provides a simple menu in the terminal so you can run the benchmark without
  remembering command-line flags.
- It calls the same underlying pipeline as `scripts/benchmark.py`.
- It writes the same output CSVs under `output/`.

How to use

Run `python scripts/benchmark.py --tui` and follow the on-screen menu.
"""

from __future__ import annotations

import curses
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from .io import find_project_root


class BenchmarkTUI:
    """
    A minimal terminal UI for running the benchmark.

    This class handles:
    - drawing menus and screens
    - asking for a few high-level run settings (fast test vs full run)
    - displaying progress while the benchmark is running
    - showing simple summary statistics after completion
    """

    def __init__(self, stdscr) -> None:
        """
        Initialize UI state.

        `stdscr` is the curses “main screen” object provided by `curses.wrapper()`.
        """
        self.stdscr = stdscr
        self.project_root = find_project_root()
        self.out_dir = self.project_root / "output"
        self.cache_root = self.project_root / ".cache" / "lipid_benchmark"
        self.state = {
            "vina_max_poses": 20,
            "workers": 1,
            "cache_normalized": True,
            "cache_contacts": True,
            "limit_entries": 0,
        }
        self.progress = {"completed": 0, "total": 1, "label": ""}
        self.current_stage = ""
        self.stats_rows: List[Dict[str, object]] = []

    def run(self) -> None:
        """
        Main menu loop.

        Users can:
        - run the benchmark
        - view summary stats
        - view output paths
        - quit
        """
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        self.stdscr.nodelay(False)
        while True:
            self._render_menu()
            key = self.stdscr.getch()
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("1"),):
                self._run_benchmark_flow()
            elif key in (ord("2"),):
                self._show_summary()
            elif key in (ord("3"),):
                self._show_paths()

    def _render_menu(self) -> None:
        """Draw the main menu screen."""
        self.stdscr.clear()
        self.stdscr.addstr(1, 2, "Lipid Docking Benchmark TUI")
        self.stdscr.addstr(3, 4, "1) Run benchmark")
        self.stdscr.addstr(4, 4, "2) Show summary stats")
        self.stdscr.addstr(5, 4, "3) Show output paths")
        self.stdscr.addstr(7, 4, "q) Quit")
        self.stdscr.refresh()

    def _prompt(self, prompt: str, default: str) -> str:
        """
        Ask the user for a value in the terminal.

        Returns the typed string, or `default` if the user presses Enter without typing.
        """
        curses.echo()
        self.stdscr.clear()
        self.stdscr.addstr(1, 2, f"{prompt} [{default}]: ")
        self.stdscr.refresh()
        value = self.stdscr.getstr(1, 2 + len(prompt) + len(default) + 4).decode().strip()
        curses.noecho()
        return value if value else default

    def _prompt_bool(self, prompt: str, default: bool) -> bool:
        """
        Ask the user a yes/no question.

        The user enters something starting with “y” to mean yes; anything else means no.
        """
        default_str = "y" if default else "n"
        value = self._prompt(prompt, default_str).lower()
        return value.startswith("y")

    def _run_benchmark_flow(self) -> None:
        """
        Guided “run benchmark” flow.

        This collects a few high-level settings and then runs `run_benchmark()` while:
        - showing a progress bar
        - showing which stage is currently running (RMSD vs contacts)
        - showing median metrics “so far” as targets complete
        """
        preset = self._prompt(
            "Preset: 1=fast sanity, 2=full, 3=no-cache, 4=no-cache (8 cores), 5=custom",
            "2",
        )
        if preset == "1":
            self.state.update({"vina_max_poses": 3, "workers": 1, "cache_normalized": True, "cache_contacts": True})
            self.state["limit_entries"] = 3
        elif preset == "3":
            self.state.update({"vina_max_poses": 20, "workers": 1, "cache_normalized": False, "cache_contacts": False})
            self.state["limit_entries"] = 0
        elif preset == "4":
            self.state.update({"vina_max_poses": 20, "workers": 8, "cache_normalized": False, "cache_contacts": False})
            self.state["limit_entries"] = 0
        elif preset == "5":
            self.state["vina_max_poses"] = int(self._prompt("Max Vina poses", str(self.state["vina_max_poses"])))
            self.state["workers"] = int(self._prompt("Workers", str(self.state["workers"])))
            self.state["cache_normalized"] = self._prompt_bool(
                "Use cached normalized complexes", self.state["cache_normalized"]
            )
            self.state["cache_contacts"] = self._prompt_bool("Use cached contacts", self.state["cache_contacts"])
            self.state["limit_entries"] = int(self._prompt("Limit entries (0=all)", str(self.state["limit_entries"])))
        else:
            self.state.update({"vina_max_poses": 20, "workers": self.state["workers"], "cache_normalized": True, "cache_contacts": True})
            self.state["limit_entries"] = 0

        self._render_progress(0, 1, "Starting...")
        try:
            from lipid_benchmark.io import write_csv
            from lipid_benchmark.pipeline import BENCHMARK_FIELDNAMES, run_benchmark

            self.cache_root.mkdir(parents=True, exist_ok=True)
            normalized_dir = self.cache_root / "normalized"
            self.current_stage = ""
            self.stats_rows = []
            allposes, summary = run_benchmark(
                self._load_pairs(),
                vina_max_poses=self.state["vina_max_poses"],
                normalized_dir=normalized_dir,
                quiet=True,
                workers=self.state["workers"],
                cache_normalized=self.state["cache_normalized"],
                cache_contacts=self.state["cache_contacts"],
                progress_cb=self._render_progress,
                stage_cb=self._render_stage,
                entry_cb=self._update_stats,
            )
            write_csv(self.out_dir / "benchmark_allposes.csv", allposes, BENCHMARK_FIELDNAMES)
            write_csv(self.out_dir / "benchmark_summary.csv", summary, BENCHMARK_FIELDNAMES)
            self._show_summary()
        except Exception as exc:
            self._display_lines("Run Failed", [str(exc)])

    def _show_paths(self) -> None:
        """Show the main output file/folder locations for the default output directory."""
        contacts_cache = self.cache_root / "contacts"
        normalized_cache = self.cache_root / "normalized"
        paths = [
            self.out_dir / "benchmark_allposes.csv",
            self.out_dir / "benchmark_summary.csv",
            normalized_cache,
            contacts_cache,
        ]
        lines = [f"{p} {'(missing)' if not p.exists() else ''}" for p in paths]
        self._display_lines("Output Paths", lines)

    def _load_pairs(self):
        """
        Load the pairs CSV used by the benchmark.

        This uses `config.yaml` (`paths.pairs`) or falls back to `scripts/pairs_curated.csv`.
        Optionally, a “limit entries” setting restricts to only the first N targets for
        quick sanity checks.
        """
        from lipid_benchmark.io import default_pairs_path, read_pairs_csv

        pairs_path = default_pairs_path(self.project_root)
        entries = read_pairs_csv(self.project_root, pairs_path)
        limit = int(self.state.get("limit_entries") or 0)
        return entries[:limit] if limit > 0 else entries

    def _show_summary(self) -> None:
        """
        Show summary statistics computed from `benchmark_summary.csv`.

        This is meant to answer quick questions like:
        - “What is the median RMSD for Boltz vs Vina-best?”
        - “How good is headgroup contact overlap on average?”
        """
        summary_path = self.out_dir / "benchmark_summary.csv"
        if not summary_path.exists():
            self._display_lines("Summary Stats", ["Summary file not found. Run the benchmark first."])
            return

        stats = self._summary_stats(summary_path)
        lines = [f"{k}: {v}" for k, v in stats]
        self._display_lines("Summary Stats", lines)

    def _summary_stats(self, path: Path) -> List[Tuple[str, str]]:
        """
        Compute a few “headline” statistics from the summary CSV.

        This intentionally stays simple (medians, counts) so it is fast and robust.
        """
        import csv

        by_method: Dict[str, Dict[str, List[float]]] = {}
        ref_contacts: Dict[str, int] = {}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                method = row.get("method", "")
                if not method:
                    continue
                by_method.setdefault(method, {"ligand_rmsd": [], "headgroup_typed_jaccard": []})
                rmsd = str(row.get("ligand_rmsd") or "").strip()
                if rmsd and rmsd.upper() != "NA":
                    by_method[method]["ligand_rmsd"].append(float(rmsd))
                jacc = str(row.get("headgroup_typed_jaccard") or "").strip()
                if jacc and jacc.upper() != "NA":
                    by_method[method]["headgroup_typed_jaccard"].append(float(jacc))
                pdbid = row.get("pdbid", "")
                if pdbid and row.get("headgroup_contacts_ref"):
                    try:
                        ref_contacts.setdefault(pdbid, int(row["headgroup_contacts_ref"]))
                    except ValueError:
                        pass

        lines: List[Tuple[str, str]] = []
        for method in sorted(by_method.keys()):
            rmsd_vals = by_method[method]["ligand_rmsd"]
            jacc_vals = by_method[method]["headgroup_typed_jaccard"]
            rmsd_med = f"{statistics.median(rmsd_vals):.3f}" if rmsd_vals else "NA"
            jacc_med = f"{statistics.median(jacc_vals):.3f}" if jacc_vals else "NA"
            lines.append((f"{method} median ligand_rmsd", rmsd_med))
            lines.append((f"{method} median headgroup_typed_jaccard", jacc_med))

        zeros = sum(1 for v in ref_contacts.values() if v == 0)
        lines.append(("PDBs with zero ref headgroup contacts", str(zeros)))
        return lines

    def _display_lines(self, title: str, lines: List[str]) -> None:
        """
        Display a “page” of text and wait for a keypress.

        If there are too many lines to fit on screen, the output is truncated.
        """
        self.stdscr.clear()
        self.stdscr.addstr(1, 2, title)
        row = 3
        max_y, _ = self.stdscr.getmaxyx()
        for line in lines:
            if row >= max_y - 2:
                self.stdscr.addstr(row, 2, "... (truncated)")
                break
            self.stdscr.addstr(row, 2, line)
            row += 1
        self.stdscr.addstr(max_y - 2, 2, "Press any key to return...")
        self.stdscr.refresh()
        self.stdscr.getch()

    def _render_progress(self, completed: int, total: int, label: str) -> None:
        """Callback used by the benchmark pipeline to update the progress bar."""
        self.progress = {"completed": completed, "total": total, "label": label}
        self._draw_running()

    def _render_stage(self, stage: str, detail: str) -> None:
        """Callback used by the benchmark pipeline to show which stage is running."""
        self.current_stage = f"{stage}: {detail}"
        self._draw_running()

    def _update_stats(self, entry_all, entry_summary, completed: int, total: int) -> None:
        """
        Callback used by the benchmark pipeline after finishing one target.

        We store per-target summary rows so we can display “median so far” stats.
        """
        self.stats_rows.extend(entry_summary)
        self._draw_running()

    def _draw_running(self) -> None:
        """
        Draw the “benchmark running” screen (progress bar + live stats).

        This is called repeatedly as progress updates arrive.
        """
        self.stdscr.clear()
        self.stdscr.addstr(1, 2, "Benchmark running...")
        max_y, max_x = self.stdscr.getmaxyx()
        bar_width = max(10, max_x - 10)
        completed = self.progress["completed"]
        total = self.progress["total"]
        label = self.progress["label"]
        ratio = 0.0 if total <= 0 else min(1.0, completed / total)
        filled = int(ratio * (bar_width - 2))
        bar = "[" + ("#" * filled) + ("-" * ((bar_width - 2) - filled)) + "]"
        self.stdscr.addstr(3, 2, bar[: max_x - 4])
        self.stdscr.addstr(5, 2, f"{completed}/{total} {label}")
        if self.current_stage:
            self.stdscr.addstr(6, 2, f"Stage: {self.current_stage}"[: max_x - 4])

        stats_lines = self._live_stats_lines()
        row = 8
        for line in stats_lines:
            if row >= max_y - 2:
                break
            self.stdscr.addstr(row, 2, line[: max_x - 4])
            row += 1
        self.stdscr.refresh()

    def _live_stats_lines(self) -> List[str]:
        """
        Compute a small set of live “median so far” lines.

        This is purely a convenience feature for the UI; it does not affect the benchmark
        output files.
        """
        if not self.stats_rows:
            return ["Stats: waiting for first target..."]

        by_method: Dict[str, List[float]] = {"boltz": [], "vina_best": [], "vina_best_headgroup": []}
        by_method_hg: Dict[str, List[float]] = {"boltz": [], "vina_best": [], "vina_best_headgroup": []}
        for row in self.stats_rows:
            method = str(row.get("method") or "")
            if method not in by_method:
                continue
            rmsd = str(row.get("ligand_rmsd") or "").strip()
            jacc = str(row.get("headgroup_typed_jaccard") or "").strip()
            if rmsd and rmsd.upper() != "NA":
                by_method[method].append(float(rmsd))
            if jacc and jacc.upper() != "NA":
                by_method_hg[method].append(float(jacc))

        lines = ["Stats (median so far):"]
        for method in ("boltz", "vina_best", "vina_best_headgroup"):
            rmsd_vals = by_method[method]
            jacc_vals = by_method_hg[method]
            rmsd_med = f"{statistics.median(rmsd_vals):.3f}" if rmsd_vals else "NA"
            jacc_med = f"{statistics.median(jacc_vals):.3f}" if jacc_vals else "NA"
            lines.append(f"{method}: RMSD {rmsd_med} | headgroup_jacc {jacc_med}")
        return lines


def main() -> int:
    """Launch the curses UI and return an exit code suitable for the shell."""
    curses.wrapper(lambda stdscr: BenchmarkTUI(stdscr).run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
