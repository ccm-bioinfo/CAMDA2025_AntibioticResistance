# 🧬 Core/Non-Core Family Filter

This folder contains a script to identify **core and non-core gene families** in pangenome datasets and filter the data accordingly. It is part of the larger analysis pipeline for the CAMDA 2024–2025 Challenge Bacterial Resistance.

## 📁 Contents

- `core_family_analysis.py`: main script with CLI support to analyze and filter pangenome data.
- Example outputs: `frequencies/`, `not_core/`, `pangenomes_filtered/` (created after running).

## ⚙️ Requirements

- Python 3.7+
- Packages:
  - `pandas`

## 🚀 Usage

Make the script executable (only once):

```bash
chmod +x core_family_analysis.py
```
### 1. Analyze core and non-core families

This command will:
- Compute frequency of each gene family
- Define core families based on a threshold (default 80%)
- Save non-core families in `not_core/`

```bash
./core_family_analysis.py analyze path/to/pangenome_<bacteria>_train.csv output/ --threshold 1.0
```

### 2. Filter pangenomes using non-core families

This will keep only:
- The first 11 metadata columns
- The non-core families previously identified

```bash
./core_family_analysis.py filter path/to/pangenome_<bacteria>_train.csv path/to/pangenome_<bacteria>_test.csv output/
```

## 📂 Output structure

```text
output/
├── frequencies/
│   └── family_frequency_<bacteria>.csv
├── not_core/
│   └── not_core_families_<bacteria>.csv
└── pangenomes_filtered/
    ├── pangenome_<bacteria>_train.csv
    └── pangenome_<bacteria>_test.csv
```


## 👩‍💻 Author

This tool was developed by [Haydeé Contreras Peruyero](https://github.com/HaydeePeruyero) as part of the pangenome analysis pipeline for antimicrobial resistance studies in the CAMDA context.

---

🧪 *Feel free to integrate this module with your own workflows or expand it for additional filtering logic.*
