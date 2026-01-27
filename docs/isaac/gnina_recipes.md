# GNINA ISAAC run recipes (for reproducibility)

This file records exact ISAAC submission commands we used during development so they can be re-run later.

## Practical ISAAC tips we learned (2026-01-25/26)

These are cluster-specific “gotchas” that came up while running GNINA jobs on ISAAC:

- **Know your QoS limits**
  - `campus-gpu` QoS is job-limited (e.g., max running jobs was 3 and max submit 6 for this user), so using a large `--shards` wastes time in `PENDING`.
  - Prefer **fewer shards** (e.g., 2–3) so you saturate the job limit but don’t queue many idle shards.

- **Partition/QoS pairing matters**
  - You cannot run on the `short` partition using `--qos campus` (partition allowlist blocks it).
  - `short` QoS also enforced a smaller walltime (in this session: **max 3 hours**). If you request longer, submission fails with `QOSMaxWallDurationPerJobLimit`.

- **Short QoS CPU cap can throttle array concurrency**
  - We saw array elements go `PENDING (QOSMaxCpuPerUserLimit)` under `short` QoS. If that happens:
    - reduce `--array=...%N` (lower concurrency), or
    - reduce `--cpus-per-task`, or
    - use the `campus` QoS/partition instead.

- **Memory is effectively “per CPU” on ISAAC partitions**
  - Example (from this session):
    - `short`: `DefMemPerCPU=3800`, `MaxMemPerCPU=4300` (≈ 3.8–4.3 GB per CPU)
    - `campus`: `DefMemPerCPU=3800`, `MaxMemPerCPU=3800`
  - So `--cpus-per-task=12` implies an upper bound of roughly 45–52 GB; requesting `--mem=200G` will be rejected.
  - Usually, omit `--mem` and just request the right number of CPUs.

- **OpenMP policy warnings**
  - ISAAC prints a warning when requesting multiple CPUs: “ensure `export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}` appears in your batch script”.
  - If using `sbatch --wrap`, the checker may still warn even if you export it inside the wrapped command. It’s typically **not fatal**.

- **Avoid long `sbatch --wrap` one-liners**
  - We hit a quoting failure where the job immediately exited with code `127` and the error `-lc: command not found` (i.e., `bash -lc ...` got broken into a command named `-lc`).
  - Prefer writing an actual `.sbatch` script file (or generating one from a helper script) to avoid nested-quote footguns.

- **Shebang and `#SBATCH` must start at column 1**
  - Slurm requires the first line to begin with `#!` (no leading spaces).
  - Likewise, `#SBATCH ...` directives are ignored if they have leading whitespace.

- **“Idle” nodes may still be unavailable**
  - `sinfo` can show nodes as `idle`, but `salloc -w <node>` may still queue. This is usually due to reservations/backfill/priority policies.

- **Keep GNINA outputs on scratch**
  - Store the image under `$SCRATCHDIR/containers/gnina.sif` and outputs under `$SCRATCHDIR/gnina/runs/<run_name>/{flat,logs}` to avoid filling `$HOME`.

- **Expect slow outliers**
  - Some targets can be dramatically slower (e.g., very flexible ligands / many torsions). If you see a single target taking far longer than others:
    - increase walltime, and/or
    - rerun that target alone, and/or
    - keep `exhaustiveness` consistent with your baseline method if comparing.

## 2026-01-26 — Rerun 4 missing/no-output targets (no-CNN)

Purpose: fix missing or 0-byte outputs in `gnina_full_no_cnn_exh8_cpu24` for:

- `8GOT`, `8QJZ` (0-byte `flat/*.pdbqt`)
- `8IVL`, `8T5T` (missing `flat/*.pdbqt`)

Command (run on ISAAC login node):

```bash
OUT="$SCRATCHDIR/gnina/runs/gnina_full_no_cnn_exh8_cpu24" \
  && IN="$SCRATCHDIR/gnina/tmp/vina_inputs_rerun_nocnn" \
  && SIF="$SCRATCHDIR/containers/gnina.sif" \
  && sbatch \
    -A acf-utk0011 \
    -p campus \
    --qos campus \
    --array=0-3%4 \
    --cpus-per-task=12 \
    --mem=200G \
    -t 12:00:00 \
    -J gnina_nocnn_fix4 \
    --output="$OUT/slurm_fix4_%A_%a.out" \
    --error="$OUT/slurm_fix4_%A_%a.err" \
    --wrap "set -euo pipefail; \
      export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}; \
      ids=(8GOT 8QJZ 8IVL 8T5T); \
      id=\${ids[\$SLURM_ARRAY_TASK_ID]}; \
      box=\$(cat \"$IN/box/\$id.txt\"); read -r cx cy cz sx sy sz <<<\"\$box\"; \
      rec=\"$IN/prep/\$id/receptor_no_ligand.pdb\"; \
      lig=\"$IN/prep/\$id/ligand.pdbqt\"; \
      outp=\"$OUT/flat/\$id.pdbqt\"; \
      logp=\"$OUT/logs/\$id.log\"; \
      mkdir -p \"$OUT/flat\" \"$OUT/logs\"; \
      rm -f \"\$outp\"; \
      apptainer exec \"$SIF\" bash -lc \" \
        set -euo pipefail; \
        tmp=\\\$(mktemp -d); \
        obabel -ipdbqt '$lig' -osdf -O \\\\\\\"\\\$tmp/ligand.sdf\\\\\\\" >/dev/null; \
        /usr/local/bin/gnina \
          -r '$rec' -l \\\\\\\"\\\$tmp/ligand.sdf\\\\\\\" \
          --center_x \$cx --center_y \$cy --center_z \$cz \
          --size_x \$sx --size_y \$sy --size_z \$sz \
          --exhaustiveness 8 --num_modes 20 --cpu 12 \
          --cnn_scoring none --scoring vina --addH 0 \
          --out '$OUT/flat/\$id.pdbqt' --log '$OUT/logs/\$id.log' --seed 0 \
          --no_gpu; \
        rm -rf \\\\\\\"\\\$tmp\\\\\\\"; \
      \""
```

### Note: container `/tmp` can be unwritable on some nodes/partitions

We observed `obabel` failures like “Cannot write to /tmp/.../ligand.sdf” when using an in-container temp dir created under `/tmp`.

If that happens, use a temp directory under `$SCRATCHDIR` (or under the run `OUT` directory) and **bind it into the container**, e.g.:

```bash
OUT="$SCRATCHDIR/gnina/runs/gnina_full_no_cnn_exh8_cpu24" \
  && IN="$SCRATCHDIR/gnina/tmp/vina_inputs_rerun_nocnn" \
  && SIF="$SCRATCHDIR/containers/gnina.sif" \
  && sbatch \
    -A acf-utk0011 \
    -p short \
    --qos short \
    --array=0-3%4 \
    --cpus-per-task=12 \
    --mem=200G \
    -t 03:00:00 \
    -J gnina_nocnn_fix4 \
    --output="$OUT/slurm_fix4_%A_%a.out" \
    --error="$OUT/slurm_fix4_%A_%a.err" \
    --wrap "set -euo pipefail; \
      export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}; \
      ids=(8GOT 8QJZ 8IVL 8T5T); \
      id=\${ids[\$SLURM_ARRAY_TASK_ID]}; \
      box=\$(cat \"$IN/box/\$id.txt\"); read -r cx cy cz sx sy sz <<<\"\$box\"; \
      rec=\"$IN/prep/\$id/receptor_no_ligand.pdb\"; \
      lig=\"$IN/prep/\$id/ligand.pdbqt\"; \
      outp=\"$OUT/flat/\$id.pdbqt\"; \
      logp=\"$OUT/logs/\$id.log\"; \
      mkdir -p \"$OUT/flat\" \"$OUT/logs\"; \
      rm -f \"\$outp\"; \
      tmp=\$(mktemp -d -p \"$OUT\" \"_tmp_\$id.XXXXXX\"); \
      apptainer exec --bind \"$IN:$IN\" --bind \"$OUT:$OUT\" --bind \"\$tmp:\$tmp\" \"$SIF\" bash -lc \" \
        set -euo pipefail; \
        obabel -ipdbqt '$lig' -osdf -O '\$tmp/ligand.sdf' >/dev/null; \
        /usr/local/bin/gnina \
          -r '$rec' -l '\$tmp/ligand.sdf' \
          --center_x \$cx --center_y \$cy --center_z \$cz \
          --size_x \$sx --size_y \$sy --size_z \$sz \
          --exhaustiveness 8 --num_modes 20 --cpu 12 \
          --cnn_scoring none --scoring vina --addH 0 \
          --out '$OUT/flat/\$id.pdbqt' --log '$OUT/logs/\$id.log' --seed 0 \
          --no_gpu; \
      \"; \
      rm -rf \"\$tmp\""
```

### Quick “did my run produce real outputs?” checks

0-byte PDBQT outputs (usually indicates a hard failure or early exit):

```bash
RUN="$SCRATCHDIR/gnina/runs/<run_name>" \
  && find "$RUN/flat" -maxdepth 1 -type f -name '*.pdbqt' -size 0 -printf '%f\n' | sort
```

Missing outputs vs the box list (assumes Vina inputs live at `/nfs/home/$USER/vina_inputs`):

```bash
RUN="$SCRATCHDIR/gnina/runs/<run_name>" \
  && comm -23 \
    <(cd /nfs/home/$USER/vina_inputs/box && for f in *.txt; do echo "${f%.txt}"; done | sort) \
    <(cd "$RUN/flat" && for f in *.pdbqt; do echo "${f%.pdbqt}"; done | sort)
```

### Rerun just a few failed targets (recommended workflow)

Rather than resubmitting a whole 101-target run, make a tiny “rerun inputs” directory containing only failing targets:

```bash
VINA_IN="/nfs/home/$USER/vina_inputs" \
  && RERUN_IN="$SCRATCHDIR/gnina/tmp/vina_inputs_rerun" \
  && mkdir -p "$RERUN_IN/box" "$RERUN_IN/prep" \
  && for id in <PDBID1> <PDBID2>; do \
       ln -sf "$VINA_IN/box/$id.txt" "$RERUN_IN/box/$id.txt"; \
       mkdir -p "$RERUN_IN/prep/$id"; \
       ln -sf "$VINA_IN/prep/$id/receptor_no_ligand.pdb" "$RERUN_IN/prep/$id/"; \
       ln -sf "$VINA_IN/prep/$id/ligand.pdbqt" "$RERUN_IN/prep/$id/"; \
     done
```
