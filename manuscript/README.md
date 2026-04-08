# Manuscript

## Layout

- `manuscript.tex` — main LaTeX source
- `supporting_information.tex` — supporting information LaTeX source
- `references.bib` — bibliography database
- `figures/` — figure files included by the LaTeX sources
- `build/` — LaTeX build artifacts (generated)
- `dist/` — final PDF output (generated)
- `build_si/` — SI build artifacts (generated)
- `dist_si/` — final SI PDF output (generated)

## Build

From the repo root:

```bash
make -C manuscript \
  all
```

This builds both the main manuscript and the supporting information.
It uses `tectonic` if available (recommended), otherwise falls back to `pdflatex`/`bibtex`.

To build only one PDF:

```bash
make -C manuscript \
  dist/manuscript.pdf
```

```bash
make -C manuscript \
  dist_si/supporting_information.pdf
```

To force an engine:

```bash
make -C manuscript \
  TEX_ENGINE=tectonic \
  all
```

```bash
make -C manuscript \
  TEX_ENGINE=pdflatex \
  all
```

To clean generated files:

```bash
make -C manuscript \
  clean
```
