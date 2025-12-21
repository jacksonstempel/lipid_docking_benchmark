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
    # Common modified residues seen in PDB files (often marked HETATM but part of the polymer)
    "MSE": "M",  # selenomethionine
    "CSO": "C",  # cysteine sulfinic acid
    "OCS": "C",  # cysteine sulfonic acid
}

WATER_RES_NAMES = {"HOH", "H2O", "WAT"}

# Small ligands inflate reporting noise, so the default pipeline drops any with fewer
# heavy atoms than this constant unless the CLI requests otherwise. Lipid ligands in
# this benchmark are far larger; <10 heavy atoms is typical for buffer/ion fragments,
# so this floor avoids accidentally picking solvent as the “ligand”.
MIN_LIGAND_HEAVY_ATOMS = 10

# Vina is requested to emit up to this many poses per target. In practice some targets
# may yield fewer poses; treat this as an upper bound rather than a guarantee.
VINA_MAX_POSES = 20
