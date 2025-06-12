# CAMDA: Antibiotic Resistance

This repository contains the organized and structured information for the **CAMDA Hackathon** competition using the **2024 and 2025 datasets**, focused on **Antibiotic Resistance**. It includes scripts, analyses, visualizations, and relevant documentation developed during the event, aiming to facilitate understanding and reproducibility of the work carried out by the team.

## General information

The dataset contains 9 bacterial species encompassing 4 antibiotics:

    - GEN: Gentamicin
    - ERY: Erythromycin
    - TET: Tetracycline
    - CAZ: Ceftazidime

For each bacterial isolate, the SRA ID corresponding to the sequencing reads is available, and inferences can be performed based on:

    - measurement values with typing method and measurement signs ('=', '>', '<' or '<='). Please note that these inferences are not possible for S. aureus and A. baumannii.
    - phenotypic status ('Susceptible', 'Intermediate' and 'Resistant'), all obtained using the latest CLSI versions (R package 'AMR' v2.1.1).

Some additional metadata are provided when available (publication ID, isolation_source, isolation_country, collection_date).

The total number of isolates for model training is 6,144, subdivided as follows: 

Species                     Antibiotic  Susceptible  Intermediate  Resistant	Notes
-----------------------------------------------------------------------------------------------------------------
Klebsiella pneumoniae      	GEN          350          150         350
Salmonella enterica        	GEN          350           21         350
Escherichia coli           	GEN          345           19         154
Staphylococcus aureus       ERY          334           46         265   		no MIC value in testing dataset
Streptococcus pneumoniae   	ERY          350           53         350
Campylobacter jejuni      	TET          211            0         326
Neisseria gonorrhoeae       TET          271          150         350
Acinetobacter baumannii    	CAZ          277          150         350   		no MIC value in testing dataset
Pseudomonas aeruginosa     	CAZ          228           95         249



## Team 2025: 

- Anton Pashkov
- Francisco Santiago Nieto
- Johana Atenea Carreón Baltazar
- Luis Raúl Figueroa Martinez
- Víctor Muñíz
- César Aguilar
- Varinia
- Johana Castelo
- Mariana
- Evelia Lorena Coss Navarrete 
- Yesenia Villicaña Molina
- Haydeé Contreras Peruyero
- Nelly Sélem Mojica
