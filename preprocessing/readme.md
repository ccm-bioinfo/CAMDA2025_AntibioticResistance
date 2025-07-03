# ⚙️ CAMDA25 Preprocessing

Documentation for the preprocessing stage of CAMDA25.



## 📊 Data Access

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15617329.svg)](https://doi.org/10.5281/zenodo.15617329)

All relevant datasets for machine learning tasks have been submitted to Zenodo
(link above) as gzipped CSV files. The filename template is
`species_antibiotic_source_subset.csv.gz`, where:

- **`species`** is the bacterium species of the dataset:
- **`antibiotic`** is the antibiotic against which the samples are tested:
`ceftazidime`, `erythromycin`, `gentamicin`, `tetracycline`.
- **`source`** refers to the method with which the dataset was produced:
  - `strict` — resistome annotations with perfect and strict cut-offs
  - `loose` — resistome annotations with loose cut-offs
  - `pangenome` — full set of gene families
  - `kmers` — 31-mer hash counts with a sampling rate of 1/10
- **`subset`** indicates whether the table comprises the CAMDA25 training or
testing sets: `train`, `test`.

## 🔀 Pipeline

In order to run the pipeline, follow these steps:

1.  Install [Docker Engine or Desktop](https://docs.docker.com/engine/install/)
on your system.
2. Open a terminal, and run the following commands from the `preprocessing/`
directory:

```shell
# 🛠️ Build container
docker build -t camda25 .

# 📦 Start container
docker run --rm -itv "$(pwd):/ext" camda25

# 👤 Login to BV-BRC and type password (required to download genomes)
p3-login username

# 🚀 Run pipeline
make &>> main.log
```

## 🗃️ File Layout

By the end of the entire preprocessing pipeline, the file layout will look
something like this:

```text
├─ 📁 data/             # Full datasets (very large, ignored by git)
│  ├─ 📁 annotations/   # Genome annotations with Prokka
│  ├─ 📁 genomes/       # CAMDA Genomes
│  ├─ 📁 localDB/       # CARD
│  ├─ 📁 pangenomes/    # Raw pangenomes with PPanGGOLiN
│  ├─ 📁 resistance/    # Resistome annotations with RGI
│  ├─ 📁 sketches/      # Sourmash sketches
│  │  ├─ 📁 deep/       # Sketches with abundances and sampling rate of 1/10
│  │  └─ 📁 shallow/    # Sketches with sampling rate of 1/1000
│  ├─ 📁 tables/        # Tables for ML model training (see Data Access)
│  └─ 📃 gtdb.sbt.zip   # Sourmash GTDB index file
├─ 📂 env/              # Conda environment definition files
│  ├─ 📃 base.yml       # Main environment
│  └─ 📃 rgi.yml        # RGI environment
├─ 📂 metadata/         # Miscellaneous metadata
│  ├─ 📃 ani.csv        # ANI-based taxonomical assignment
│  ├─ 📃 test.csv       # CAMDA25 testing metadata
│  └─ 📃 train.csv      # CAMDA25 training metadata
├─ 📂 src/              # Source code (see Scripts)
├─ 📃 .dockerignore     # Avoid Docker reading the whole data/ directory
├─ 📃 .gitignore        # Do not upload the data/ directory to git
├─ 📃 Dockerfile        # Docker image definition
├─ 📃 Makefile          # Pipeline definition
└─ 📃 readme.md         # Documentation
```

## 💻 Scripts

### 1. `download_genomes.py`

```text
usage: download_genomes.py [-h] [-o path]

Read genome IDs from stdin and save them gzipped into a directory. Valid
identifiers include those that start with 'GCA_', 'ENA_' followed by a BioSample
accession or 'BVBRC_' followed by a BV-BRC genome accession.

options:
  -h, --help         show this help message and exit
  -o, --output path  output directory (default: .)

example: download_genomes.py -o genomes <<< ENA_SAMEA800315
```

### 2. `compute_sketches.py`

```text
usage:
```

### 3. `get_gtdb_ani.py`

```text
usage: get_gtdb_ani.py [-h] [-d path]

Read Sourmash signature files from stdin, get the closest reference genome for
each input from GTDB using ANI, and print a CSV file of four columns: 'input',
'reference_genome', 'reference_taxonomy', and 'ani'.

options:
  -h, --help           show this help message and exit
  -d, --database path  path to GTDB index file; if one is not found in the
                       specified location, it will be automatically downloaded
                       for you (default: 'gtdb.sbt.zip')

example: get_gtdb_ani.py -d gtdb.sbt.zip <<< genomes/ENA_SAMEA800315.fa.gz
```
