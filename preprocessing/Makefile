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

all: download-genomes
.PHONY: all

download-genomes:
> @mkdir -p data/genomes/camda
> @(cut -f 18 -d, metadata/train.csv && cut -f 8 -d, metadata/test.csv) | \
    python3 src/download_genomes.py -o data/genomes/camda
.PHONY: download-genomes
