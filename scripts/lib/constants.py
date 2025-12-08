from __future__ import annotations

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

WATER_RES_NAMES = {"HOH", "H2O", "WAT"}

# Small ligands inflate reporting noise, so the default pipeline drops any with fewer
# heavy atoms than this constant unless the CLI requests otherwise. Lipid ligands in
# this benchmark are far larger; <10 heavy atoms is typical for buffer/ion fragments,
# so this floor avoids accidentally picking solvent as the “ligand”.
MIN_LIGAND_HEAVY_ATOMS = 10

# Vina emits 20 poses per target in this benchmark; keep a single source of truth so
# RMSD, contact extraction, and metrics all stay in sync if this ever changes.
VINA_MAX_POSES = 20
