#!/bin/bash
{
# Anton Pashkov 2025-06-18
# Download reference genomes for alignment-based assembly
# Saves the genomes in ../../genome_reassembly/reference/
# Requires:
# - ncbi-datasets-cli (any version, conda-forge)

# Set Bash strict mode
set -euo pipefail
IFS=$'\t\n'

# Change to git base dir (CAMDA2025_AntibioticResistance)
cd "$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"

# Create output directories
mkdir -p genome_reassembly/reference

# Download reference genomes
#datasets download genome accession --include genome,gbff \
#  --filename genome_reassembly/reference/ncbi.zip \
#  GCA_022869665.1 GCA_000006945.2 GCA_000008865.2 \
#  GCA_000013425.1 GCA_001457635.1 GCA_000009085.1 \
#  GCA_013030075.1 GCA_009035845.1 GCA_000006765.1

# Extract genomes
unzip -d genome_reassembly/reference/ncbi \
  genome_reassembly/reference/ncbi.zip

basename -a genome_reassembly/reference/ncbi/ncbi_dataset/data/GC* | \
while read -r base; do
  mv -v genome_reassembly/reference/ncbi/ncbi_dataset/data/${base}/*.fna \
    genome_reassembly/reference/${base}.fna
  mv -v genome_reassembly/reference/ncbi/ncbi_dataset/data/${base}/*.gbff \
    genome_reassembly/reference/${base}.gbk
done

rm -vrf genome_reassembly/reference/ncbi*

exit 0
}
