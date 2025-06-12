# Metadata description

### 1. Resistance categories metadata ([ResistanceCategories.tsv](https://raw.githubusercontent.com/ccm-bioinfo/CAMDA2025_AntibioticResistance/refs/heads/main/metadata/ResistanceCategories.tsv))

Includes extra information regarding the ARO ids, i.e., unique numbers that
identify genes conferring antibiotic resistance from
[CARD](https://card.mcmaster.ca/). Contains the following columns:

- **ARO** - ARO id.
- **Gene Name** - common name for the gene.
- **Antibiotic** - name of the antibiotic for which the gene confers resistance.
  When a gene protects against multiple antibiotics, a comma-separated list is
  written instead. Can be empty if the antibiotic is unknown.
- **AMR Gene Family** - gene family which the gene belongs to. A gene can belong
  to multiple families, and a comma-separated list of families is written in
  such cases.
- **Drug Class** - type of substance that the gene produces. A gene can produce
  many types of substances, so a comma-separated list of substances is used in
  those cases.
- **Resistance Mechanism** - method which the gene uses to protect the organism
  against the antibiotic. Some genes have multiple resistance mechanisms, and,
  just like with previous columns, a comma-separated list of mechanisms is
  written for them.
