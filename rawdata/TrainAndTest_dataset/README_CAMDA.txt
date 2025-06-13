
The dataset contains 9 bacterial species encompassing 4 antibiotics:
    _ GEN: Gentamicin
    _ ERY: Erythromycin
    _ TET: Tetracycline
    _ CAZ: Ceftazidime

For each bacterial isolate, the SRA ID corresponding to the sequencing reads is available, and inferences can be performed based on:
    _ measurement values with typing method and measurement signs ('=', '>', '<' or '<='). Please note that these inferences are not possible for S. aureus and A. baumannii.
    _ phenotypic status ('Susceptible', 'Intermediate' and 'Resistant'), all obtained using the latest CLSI versions (R package 'AMR' v2.1.1).

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



The testing dataset contains 5,345 entries (max. 700 per species).
Good luck!
