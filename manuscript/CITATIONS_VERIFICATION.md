# Citation Verification Log

This document lists every reference in `references.bib` with a resolvable link and a short proof statement, so each entry can be manually checked against the primary record.

Verification was performed against Crossref (`https://api.crossref.org/works/<doi>`) and, where available, the publisher page. Author lists, titles, journals, volumes, issues, pages, and years reflect the primary record as returned by Crossref on 2026-04-09.

Two issues from the pre-verification bib were corrected as part of this pass:

- `wohlwend2024boltz` — author list had been partially hallucinated (e.g., "Murat, Welling", "Weinstein, Hannah", "Swanson, Wojciech"). Corrected from the bioRxiv record.
- `wang2016comprehensive` — first author was listed as "Zhihai Wang"; the primary record lists "Zhe Wang".
- `escriba2008membranes` — two author first names were incorrect ("Lucía" Sánchez-Magraner, "Alicia M." Fernández). Corrected to "Lissete" and "Asia M." per Crossref.
- `malinina2006gltp` — author list was missing Malakhova, Kanack, Lu, and Abagyan; first author first name was "Larisa" rather than the correct "Lucy"; pages were listed as `1996--2011` rather than the correct article number `e362`.
- `sridhar2017computational` — no such article could be located in Crossref, PubMed, Google Scholar, or Springer (the cited "Methods in Molecular Biology, vol. 1529, pp. 317--330, 2017" slot belongs to different content). The entry has been removed and the sentence in the Introduction that relied on it has been reworded to stand on its own rather than make a literature claim.

---

## Verified entries

### 1. `wohlwend2024boltz` — Boltz-1 preprint
- **Title:** Boltz-1: Democratizing Biomolecular Interaction Modeling
- **Authors:** Wohlwend, Corso, Passaro, Getz, Reveiz, Leidal, Swiderski, Atkinson, Portnoi, Chinn, Silterra, Jaakkola, Barzilay
- **Venue:** bioRxiv, 2024, 2024.11.19.624167
- **DOI:** [10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167)
- **Verified against:** Crossref + bioRxiv landing page.

### 2. `vanmeer2008membrane`
- **Title:** Membrane lipids: where they are and how they behave
- **Authors:** van Meer, Voelker, Feigenson
- **Venue:** Nature Reviews Molecular Cell Biology 9(2):112–124, 2008
- **DOI:** [10.1038/nrm2330](https://doi.org/10.1038/nrm2330)
- **Verified against:** Crossref.

### 3. `escriba2008membranes`
- **Title:** Membranes: a meeting point for lipids, proteins and therapies
- **Authors:** Escribá, González-Ros, Goñi, Kinnunen, Vigh, Sánchez-Magraner (Lissete), Fernández (Asia M.), Busquets, Horváth, Barceló-Coblijn
- **Venue:** Journal of Cellular and Molecular Medicine 12(3):829–875, 2008
- **DOI:** [10.1111/j.1582-4934.2008.00281.x](https://doi.org/10.1111/j.1582-4934.2008.00281.x)
- **Verified against:** Crossref.

### 4. `kitchen2004docking`
- **Title:** Docking and scoring in virtual screening for drug discovery: methods and applications
- **Authors:** Kitchen, Decornez, Furr, Bajorath
- **Venue:** Nature Reviews Drug Discovery 3(11):935–949, 2004
- **DOI:** [10.1038/nrd1549](https://doi.org/10.1038/nrd1549)
- **Verified against:** Crossref.

### 5. `trott2010autodock`
- **Title:** AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading
- **Authors:** Trott, Olson
- **Venue:** Journal of Computational Chemistry 31(2):455–461, 2010
- **DOI:** [10.1002/jcc.21334](https://doi.org/10.1002/jcc.21334)
- **Verified against:** Crossref.

### 6. `jumper2021highly`
- **Title:** Highly accurate protein structure prediction with AlphaFold
- **Authors:** Jumper, Evans, Pritzel, Green, ... (28 authors total; "and others" in the bib)
- **Venue:** Nature 596(7873):583–589, 2021
- **DOI:** [10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
- **Verified against:** Crossref.

### 7. `abramson2024accurate`
- **Title:** Accurate structure prediction of biomolecular interactions with AlphaFold 3
- **Authors:** Abramson, Adler, Dunger, Evans, Green, ... ("and others" in the bib)
- **Venue:** Nature 630(8016):493–500, 2024
- **DOI:** [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)
- **Verified against:** Crossref.

### 8. `wang2016comprehensive`
- **Title:** Comprehensive evaluation of ten docking programs on a diverse set of protein–ligand complexes: the prediction accuracy of sampling power and scoring power
- **Authors:** Wang (Zhe), Sun, Yao, Li (Dan), Xu, Li (Youyong), Tian, Hou
- **Venue:** Physical Chemistry Chemical Physics 18(18):12964–12975, 2016
- **DOI:** [10.1039/C6CP01555G](https://doi.org/10.1039/C6CP01555G)
- **Verified against:** Crossref.

### 9. `su2019comparative`
- **Title:** Comparative assessment of scoring functions: the CASF-2016 update
- **Authors:** Su, Yang, Du, Feng, Liu, Li, Wang
- **Venue:** Journal of Chemical Information and Modeling 59(2):895–913, 2019
- **DOI:** [10.1021/acs.jcim.8b00545](https://doi.org/10.1021/acs.jcim.8b00545)
- **Verified against:** Crossref.

### 10. `berman2000protein`
- **Title:** The Protein Data Bank
- **Authors:** Berman, Westbrook, Feng, Gilliland, Bhat, Weissig, Shindyalov, Bourne
- **Venue:** Nucleic Acids Research 28(1):235–242, 2000
- **DOI:** [10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235)
- **Verified against:** Crossref. (Crossref returns only the first author; the full author list is visible at the publisher page.)

### 11. `rdkit` — software citation
- **Title:** RDKit: Open-source cheminformatics
- **URL:** [https://www.rdkit.org](https://www.rdkit.org)
- **Note:** Software project, no DOI. Matches common RDKit citation practice.

### 12. `cock2009biopython`
- **Title:** Biopython: freely available Python tools for computational molecular biology and bioinformatics
- **Authors:** Cock, Antao, Chang, Chapman, Cox, Dalke, Friedberg, Hamelryck, Kauff, Wilczynski, de Hoon
- **Venue:** Bioinformatics 25(11):1422–1423, 2009
- **DOI:** [10.1093/bioinformatics/btp163](https://doi.org/10.1093/bioinformatics/btp163)
- **Verified against:** Crossref.

### 13. `pandamap` — software citation
- **Title:** PandaMap: Protein–ligand interaction analysis
- **URL:** [https://github.com/chopralab/pandamap](https://github.com/chopralab/pandamap)
- **Note:** Software project, no DOI.

### 14. `mirdita2022colabfold`
- **Title:** ColabFold: making protein folding accessible to all
- **Authors:** Mirdita, Schütze, Moriwaki, Heo, Ovchinnikov, Steinegger
- **Venue:** Nature Methods 19(6):679–682, 2022
- **DOI:** [10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
- **Verified against:** Crossref.

### 15. `passaro2025boltz2` — Boltz-2 preprint
- **Title:** Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction
- **Authors:** Passaro, Corso, Wohlwend, Reveiz, Thaler, Somnath, Getz, Portnoi, Roy, Stark, Kwabi-Addo, Beaini, Jaakkola, Barzilay
- **Venue:** bioRxiv, 2025, 2025.06.14.659707
- **DOI:** [10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)
- **Verified against:** Crossref.

### 16. `goddard2018chimerax`
- **Title:** UCSF ChimeraX: Meeting modern challenges in visualization and analysis
- **Authors:** Goddard, Huang, Meng, Pettersen, Couch, Morris, Ferrin
- **Venue:** Protein Science 27(1):14–25, 2018
- **DOI:** [10.1002/pro.3235](https://doi.org/10.1002/pro.3235)
- **Verified against:** Crossref.

### 17. `mcnutt2025gnina13`
- **Title:** GNINA 1.3: the next increment in molecular docking with deep learning
- **Authors:** McNutt, Li, Meli, Aggarwal, Koes
- **Venue:** Journal of Cheminformatics 17(1):28, 2025
- **DOI:** [10.1186/s13321-025-00973-x](https://doi.org/10.1186/s13321-025-00973-x)
- **Verified against:** Crossref.

### 18. `masters2025physics`
- **Title:** Investigating whether deep learning models for co-folding learn the physics of protein–ligand interactions
- **Authors:** Masters, Mahmoud, Lill
- **Venue:** Nature Communications 16:8854, 2025
- **DOI:** [10.1038/s41467-025-63947-5](https://doi.org/10.1038/s41467-025-63947-5)
- **Verified against:** Crossref.

### 19. `hunte2008lipids`
- **Title:** Lipids and membrane protein structures
- **Authors:** Hunte, Richers
- **Venue:** Current Opinion in Structural Biology 18(4):406–411, 2008
- **DOI:** [10.1016/j.sbi.2008.03.008](https://doi.org/10.1016/j.sbi.2008.03.008)
- **Verified against:** Crossref.

### 20. `malinina2006gltp`
- **Title:** The Liganding of Glycolipid Transfer Protein Is Controlled by Glycolipid Acyl Structure
- **Authors:** Malinina (Lucy), Malakhova, Kanack, Lu, Abagyan, Brown, Patel
- **Venue:** PLoS Biology 4(11):e362, 2006
- **DOI:** [10.1371/journal.pbio.0040362](https://doi.org/10.1371/journal.pbio.0040362)
- **Verified against:** Crossref and PLOS landing page.

### 21. `fantini2013cholesterol`
- **Title:** How Cholesterol Interacts with Membrane Proteins: An Exploration of Cholesterol-Binding Sites Including CRAC, CARC, and Tilted Domains
- **Authors:** Fantini, Barrantes
- **Venue:** Frontiers in Physiology 4:31, 2013
- **DOI:** [10.3389/fphys.2013.00031](https://doi.org/10.3389/fphys.2013.00031)
- **Verified against:** Crossref.

---

## Removed entries

### `sridhar2017computational` — **not found, removed**
- **Previously cited as:** Sridhar, Ross, Sherborne. "Computational approaches for lipid docking." Methods in Molecular Biology 1529:317–330 (2017).
- **Search outcome:** No matching record in Crossref, PubMed, Google Scholar, or the Springer Methods in Molecular Biology series index. The Methods in Molecular Biology volume 1529 page 317–330 slot corresponds to unrelated content. The entry appears to be fabricated and has been deleted from `references.bib`. The single sentence in the Introduction that cited it has been reworded as a standalone observation rather than a literature claim.
