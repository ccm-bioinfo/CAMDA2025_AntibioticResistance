# CAMDA 2026 — Antimicrobial resistance prediction

## Data

Zenodo record [19824655](https://zenodo.org/records/19824655) (CC-BY-4.0):
gene-level pangenome features (~29K genomes × ~7M features, sparse counts) with
MIC / phenotype metadata (CLSI 2026).

| File | Size |
|---|---|
| `pangenome.npz` | 1.76 GB — sparse count matrix |
| `pangenome.features.txt` | 205 MB — feature names |
| `pangenome.metadata.csv` | 4.6 MB — antibiotic, species, MIC, phenotype |
| `pangenome.json` | 0.5 MB |

## Locations on Alnitak

| Path | Content |
|---|---|
| `/files2/camda2026/data/` | Shared Zenodo data (MD5-verified 2026-10-28) |
| `/files2/camda2026/<user>/` | Personal results |

Scripts live only in this repo (`camda2026/scripts/`); no data under `/home`.

## Scripts

- `scripts/01-download-zenodo.sh` — download and MD5-verify the Zenodo files.

## Data exploration

### *Escherichia* in `pangenome.metadata.csv`

Each row is a genome × antibiotic pair, so one genome can appear more than once.

- **4,419 rows / 3,891 unique genomes** with `gtdb_genus == Escherichia`
- GTDB species: *E. coli* 4,417 · *E. marmotae* 1 · *E. ruysiae* 1
  (original `species` column: *E. coli* 4,418 · *Escherichia* sp. 3)

| Antibiotic | Dataset | Resistant | Intermediate | Susceptible |
|---|---|---:|---:|---:|
| ciprofloxacin | c24_balance | 404 | – | 425 |
| ciprofloxacin | c24_test c24_balance | 23 | – | 3 |
| ciprofloxacin | c24_test | 56 | 19 | 94 |
| gentamicin | c25_balance | 740 | 129 | 1,261 |
| gentamicin | c25_test | 487 | 1 | 20 |
| gentamicin | c25_train c25_balance | 9 | 3 | 4 |
| gentamicin | c25_train | 142 | 8 | 332 |
| meropenem | c24_train | 28 | – | 231 |

Phenotype = `adjusted_phenotype` (CLSI 2026). Notes: gentamicin `c25_test` is
heavily imbalanced (487 R vs 20 S); meropenem has few resistant genomes; some
genomes belong to more than one dataset — check for train/test leakage.

Reproduce on Alnitak:

```bash
cd /files2/camda2026/data
awk -F, 'NR>1 && $9=="Escherichia" {print $4" | "$2" | "$12}' pangenome.metadata.csv | sort | uniq -c
```
