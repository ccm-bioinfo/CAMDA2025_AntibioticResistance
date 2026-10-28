import numpy as np
import scipy as sp
import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split

from camda_functions_2026 import get_tfidf, get_factorization, get_y_ordinal, save_pickle_model,load_from_pickle
from camda_functions_2026 import split_stratified_into_train_val_test, train_and_save_binary_models,evaluate_models, results_report_phenotype,evaluate_models,sparse_fileter
from camda_functions_2026 import get_data2, transforms2, transforms_test2

metadata, counts = get_data2()
ruta_fact = "data_models/"
n_comp = 100

if not os.path.exists(ruta_fact):
    os.makedirs(ruta_fact)

label_file = 'gentamicin'

metadata_gentamicin, counts_gentamicin = sparse_fileter(counts, metadata ,label=label_file)
fidf, tfidf_vect, amr_lsa= \
    transforms2(amr_count=counts_meropenem, data_label=label_file, only_counts = False, exist_fact = False,
                    exist_vect = False, fact_path = ruta_fact, n_comp = n_comp)
Y= get_y_ordinal(metadata=metadata_gentamicin, ordinal=False)