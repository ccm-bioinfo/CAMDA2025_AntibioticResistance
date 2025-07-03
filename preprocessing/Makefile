SHELL := bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

all: sketch-genomes
.PHONY: all

download-genomes:
> @mkdir -p data/genomes
> @(cut -f 18 -d, metadata/train.csv && cut -f 8 -d, metadata/test.csv) | \
    sort | uniq | python3 src/download_genomes.py -o data/genomes
.PHONY: download-genomes

sketch-genomes: download-genomes
> @mkdir -p data/sketches/shallow
> @basename -as .fa.gz data/genomes/*.fa.gz | xargs -I {} -P 10 | \
    sourmash sketch dna -o data/sketches/shallow/{}.sig data/genomes/{}.fa.gz
.PHONY: sketch-genomes

metadata/ani.csv: sketch-genomes
> @find data/sketches/shallow -name '*.sig' | \
    python3 src/get_gtdb_ani.py -d data/gtdb.sbt.zip > metadata/ani.csv
