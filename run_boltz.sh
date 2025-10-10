#!/usr/bin/env bash
# ============================================================
# run_boltz.sh — Minimal, laptop-safe Boltz runner
# Now takes a PDB ID and finds the YAML at:
#   ~/lipid_docking_benchmark/model_inputs/test_inputs/{PDB_ID}.yaml
# ============================================================

set -euo pipefail

# ---------- 1) validate CLI input --------------------------------------------
if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <PDB_ID>" >&2
  exit 1
fi

ID_RAW="$1"
ID_UPPER="${ID_RAW^^}"
ID_LOWER="${ID_RAW,,}"

INPUT_ROOT="$HOME/lipid_docking_benchmark/model_inputs/benchmark_inputs"
CAND1="${INPUT_ROOT}/${ID_RAW}.yaml"
CAND2="${INPUT_ROOT}/${ID_UPPER}.yaml"
CAND3="${INPUT_ROOT}/${ID_LOWER}.yaml"

# Resolve YAML (allow raw/upper/lower case)
if   [[ -f "$CAND1" ]]; then YAML_PATH="$CAND1"
elif [[ -f "$CAND2" ]]; then YAML_PATH="$CAND2"
elif [[ -f "$CAND3" ]]; then YAML_PATH="$CAND3"
else
  echo "YAML not found for ${ID_UPPER} in ${INPUT_ROOT} (tried ${ID_RAW}, ${ID_UPPER}, ${ID_LOWER})." >&2
  exit 2
fi

# ---------- 2) derive PDB ID & output dir ------------------------------------
PDB_ID="$ID_UPPER"
BASE_OUT="$HOME/lipid_docking_benchmark/model_outputs"
OUT_DIR="${BASE_OUT}/${PDB_ID}_output"
mkdir -p "$OUT_DIR"
echo "$OUT_DIR"

# ---------- 3) run Boltz ------------------------------------------------------
boltz predict "$YAML_PATH" \
      --use_msa_server \
      --no_kernels \
      --sampling_steps 25 \
      --recycling_steps 1 \
      --diffusion_samples 1 \
      --output_format mmcif \
      --out_dir "$OUT_DIR"

echo "✅  Boltz finished for ${PDB_ID}"
echo "📂  Results in $OUT_DIR"
