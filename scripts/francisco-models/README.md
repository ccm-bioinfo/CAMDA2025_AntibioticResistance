# francisco-models

Phenotype (R/S) and MIC prediction models for the CAMDA 2026 AMR challenge
(data: Zenodo record 19824655). Author: Francisco Santiago.

Pipeline: pangenome counts → TF-IDF → SVD (LSA) / NMF → train/test split →
classifiers (RF, KNN, SVC, logistic regression, …) → metrics and plots.

## Function modules

| File | Description |
|---|---|
| `new_camda_functions.py` | Current module: data loading (`get_data2`), filtering, TF-IDF/SVD/NMF transforms, training and evaluation. Used by `Model_maker.py`. |
| `camda_functions_2026.py` | Earlier 2026 version of the module. Used by the `*26.py` scripts. |
| `camda_functions.py` | 2025-era module, still imported by the MIC ordinal notebook. |

## Scripts

| File | Description |
|---|---|
| `Model_maker.py` | Full phenotype pipeline for one antibiotic and (optionally) one genus. `python -u Model_maker.py --antibiotico meropenem --genero Klebsiella` (`--genero ninguno` for antibiotic only). |
| `caz26.py`, `cip26.py`, `ery26.py`, `gen26.py`, `mer26.py`, `tet26.py` | Per-antibiotic training scripts (ceftazidime, ciprofloxacin, erythromycin, gentamicin, meropenem, tetracycline). |

## Notebooks

| File | Description |
|---|---|
| `phenotype-pipeline-test.ipynb` | Step-by-step test of the phenotype pipeline (`new_camda_functions`). |
| `phenotype-models-space.ipynb` | Phenotype models by genus / antibiotic. |
| `mic-ordinal-acinetobacter-strict-lsa.ipynb` | MIC estimation with ordinal classification (Acinetobacter, strict, LSA). |
| `lasso-phenotype-prediction.ipynb` | LASSO for phenotype prediction. Imports `camda_functions_FRANCISCO` (2025 module, not included here). |
| `bash-assistant.ipynb` | Small Claude-based bash helper; needs `ANTHROPIC_API_KEY` in the environment. |

## Outputs

Scripts write relative to the working directory: `data_models/` (TF-IDF/SVD/NMF
pickles), `modelos_fenotipo/` (trained models) and `resultados_fenotipo/<label>/`
(logs, metrics, plots). Run them from `/files2/camda2026/francisco/` on Alnitak,
not from `/home`. Input data: `/files2/camda2026/data/`.
