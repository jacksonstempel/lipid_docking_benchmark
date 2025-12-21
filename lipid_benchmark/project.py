"""
Locate the repository root.

Plain-language overview

Many inputs in this project are written as paths relative to the repository root
(`benchmark_references/...`, `model_outputs/...`, etc.). To keep that consistent regardless of
where a command is launched from, we find the root directory first.
"""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Return the repository root directory.

    How it works:
    - Start from `start` (or the current working directory if omitted).
    - Walk up parent directories until we find `config.yaml`.

    Why `config.yaml`?
    - It is a simple, reliable “marker file” that only exists at the repo root in this project.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config.yaml").is_file():
            return candidate
    raise RuntimeError(f"Could not find config.yaml in {here} or any parent directory.")
