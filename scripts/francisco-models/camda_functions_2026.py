#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:52:14 2023

@author: victor
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier

from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
# Hacer compatible la red con sckitlearn
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from scikeras.wrappers import KerasClassifier





import random 
#from keras.utils import to_categorical
'''
Representacion de los datos basado en conteos de genes
'''

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF
import pickle

def get_data(data_path = 'https://zenodo.org/records/15930378/files/', data_file = 'default', ty = 'csv', zipped = True):
    
    data_url = data_path + data_file + '.' + ty
    comp = 'infer'
    if zipped:
        data_url = data_url + '.bz2'
        comp = 'bz2'

    #print(data_url)
    # Read the TSV file directly from the URL into a pandas DataFrame
    if ty == 'tsv':
        data_df = pd.read_csv(data_url, sep='\t', low_memory=False, compression=comp, index_col=0)
    else:
        data_df = pd.read_csv(data_url, low_memory=False, compression=comp, index_col=0)

    # quitar NaN
    data_df = data_df.dropna()

    metadata = data_df.iloc[:,:10]
    counts = data_df.iloc[:,10:]

    return metadata, counts



def get_tfidf(data):
    """
    Ajusta y transforma una matriz de conteos usando TF-IDF.

    Parameters
    ----------
    data : csr_matrix o pd.DataFrame
        Matriz de conteos.

    Returns
    -------
    tfidf_transformer : TfidfTransformer
        Transformador TF-IDF ajustado.

    X_tfidf : csr_matrix
        Matriz TF-IDF en formato sparse.
    """

    tfidf_transformer = TfidfTransformer()

    # ajustar modelo
    tfidf_transformer.fit(data)

    # transformar
    X_tfidf = tfidf_transformer.transform(data)

    # IMPORTANTE:
    # NO usar .toarray()

    return tfidf_transformer, X_tfidf

# realiza factorizacion SVD o NMF
def get_factorization(data, n_comp=100, nmf=True):
    if nmf==False:
        fact_model = TruncatedSVD(n_components=n_comp)
        fact_model.fit(data)
    else:
        fact_model = NMF(n_components = n_comp, init=None, max_iter=12000)
        fact_model.fit(data)
    
    return fact_model

def save_pickle_model(model_obj, file_path, model_name):    
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name,'wb') as file:
        pickle.dump(model_obj,file)

def load_from_pickle(file_path, model_name):
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name,'rb') as file:
        fact_model = pickle.load(file)
    
    return fact_model


'''
Reducción y selección de características.
Se probarán diferentes métodos para seleccionar características relevantes. 
Los criterios de relevancia son diferentes para cada método.
Se usarán todos los datos como entrenamiento.
'''
def get_reduced_df(df, varnames, importances_df):
    var_to_drop = [elem for idx, elem in enumerate(varnames) if elem not in importances_df.index]
    reduced_df = df.drop(var_to_drop, axis=1)
    return reduced_df

# selección de características con random forests
# El criterio para la selección de características, es la reducción del índice de impureza de Gini
def rf_features(X_df, y, num_features = 100, n_trees=500, min_samples_split=3):
    rf = RandomForestClassifier(n_estimators=n_trees, min_samples_split=3, n_jobs=-1, random_state=42)
    rf.fit(X_df,y)
    varnames = X_df.columns.tolist()
    importances = rf.feature_importances_
    indices = np.flip(np.argsort(importances))[:num_features]
    fnames = [varnames[i] for i in indices]
    rf_importances = pd.Series(importances[indices], index=fnames)
    
    return rf_importances
    
    
    
# Recursive feature elimination
# En RFE, las variables "importantes" son las que quedan después de elimnar las "menos importantes". 
# El criterio para eliminar un conjunto de variables en cada recursión, 
# son los pesos del modelo base (estimator). En éste caso, se usa una SVM con un kernel lineal, 
# para mayor facilidad y porque es en éste caso, la solución está definida en el espacio original 
# de las variables. No estoy seguro si RFE sea válido (o cómo se haga) al usar un kernel no lineal. 
# Es necesario checarlo.... También puede usarse algún otro estimator.

def rfe_features(X_df, y, num_features = 100, steps = 50):
    svc = SVC(kernel="linear", C=1)
    #clf = LogisticRegression()
    rfe = RFE(estimator=svc, step=steps, n_features_to_select=num_features, verbose=False)
    rfe.fit(X_df, y)
    varnames = X_df.columns.tolist()
    rfe_indices = rfe.get_support(indices=True)
    fnames = [varnames[i] for i in rfe_indices]
    rfe_importances = pd.Series(rfe.ranking_[rfe_indices], index=fnames)
    
    return rfe_importances


def split_stratified_into_train_val_test(X, y, frac_train=0.6, frac_val=0.15, frac_test=0.25, std = True, 
                                         two_subsets=False, random_state=None):
    '''
    Splits a dataset into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in y (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    X : numpy dataframe of covariates
    y : numpy array of responses
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''
    
    if round(frac_train + frac_val + frac_test,10) != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    # Split original dataframe into temp and test dataframes.
    #x_train, x_temp, y_train, y_temp = train_test_split(X, y, stratify=y, test_size=(1.0 - frac_train), random_state=random_state)
    x_temp, x_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=(1.0 - (frac_train+frac_val)), random_state=random_state)
    scaler = None
    if std:
        # standardize train_val (temp) and test data
        scaler = StandardScaler()
        x_temp = scaler.fit_transform(x_temp)
        x_test = scaler.transform(x_test)
        
    # weights for class imbalance (https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
    class_w = compute_class_weight('balanced',classes=np.unique(y_temp),y=y_temp)
    # the latter is equivalent to:
    # unique, class_counts = np.unique(y_temp, return_counts=True)
    # class_w = sum(class_counts)/(len(unique)*class_counts)    
    if two_subsets:        
        x_train = x_temp
        y_train = y_temp
        x_val = None
        y_val = None
        #return x_train, y_train, x_test, y_test, class_w, scaler
    else:
        # Split the temp dataframe into train and val dataframes.
        relative_frac_val = frac_val / (frac_train + frac_val)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, stratify=y_temp, 
                                                          test_size=relative_frac_val, random_state=random_state)
        #assert len(df_input) == len(df_train) + len(df_val) + len(df_test)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, class_w, scaler


def get_y_ordinal(metadata, ordinal=True):
    
    if ordinal==True:
        # MIC recategorizado, es decir, se juntan algunos valores, sobre todo de concentraciones pequeñas
        str_mic = [str(val) for val in metadata['adjusted_mic']]

        # categorización de la respuesta
        #y = str_mic.astype('category')
        le = LabelEncoder()
        le.fit(str_mic)
        #le.classes_
        y_cat = le.transform(str_mic)

        Y = pd.DataFrame({'MIC':metadata['mic'], 'MIC_recat':metadata['adjusted_mic'],
                          'MIC_str':str_mic, 'MIC_cat':y_cat})

        return Y, le
    else:
    
        Y = metadata["phenotype"]
    return Y
    

def transforms(amr_count, data_label, only_counts = False, exist_fact = False, exist_vect = False,
                    fact_path = os.getcwd(), n_comp = 100):

    if not os.path.exists(fact_path):
        os.makedirs(fact_path)

    
    #Y = metadata["assigned_phenotype"]
 
    # realiza TF-IDF y factorizacion
    if only_counts is False:
        if exist_vect is False:
            tfidf_vect, tfidf = get_tfidf(amr_count)
            # save tfidf model
            pkl_file = 'tfidf_'+ data_label + '.pkl'
            save_pickle_model(tfidf_vect, fact_path, pkl_file)
        else:
            pkl_file = 'tfidf_'+ data_label + '.pkl'
            tfidf_vect = load_from_pickle(fact_path, pkl_file)
            X = tfidf_vect.transform(amr_count)
            tfidf = X.toarray()

        tfidf = pd.DataFrame(tfidf,columns=tfidf_vect.get_feature_names_out(),index=amr_count.index)

        # si no hay un modelo de factorización guardado, obtiene uno y lo guarda
        if exist_fact is False:
            svd_fact = get_factorization(data=tfidf, n_comp=n_comp, nmf=False)
            nmf_fact = get_factorization(data=tfidf, n_comp=n_comp, nmf=True)
            pkl_file = 'nmf_'+ data_label + '.pkl'
            save_pickle_model(nmf_fact, fact_path, pkl_file)
            pkl_file = 'svd_' + data_label + '.pkl'
            save_pickle_model(svd_fact, fact_path, pkl_file)
        else: 
            # lsa
            pkl_file = 'svd_' + data_label + '.pkl'
            svd_fact = load_from_pickle(fact_path, pkl_file)
            # nmf
            pkl_file = 'nmf_' + data_label + '.pkl'
            nmf_fact = load_from_pickle(fact_path, pkl_file)

        amr_lsa = svd_fact.transform(tfidf)
        amr_nmf = nmf_fact.transform(tfidf)
        amr_lsa = pd.DataFrame(amr_lsa,index=amr_count.index)
        amr_nmf = pd.DataFrame(amr_nmf,index=amr_count.index)
        return tfidf, tfidf_vect, amr_lsa, amr_nmf
    else:
        # si no hay un modelo de factorización guardado, obtiene uno y lo guarda
        if exist_fact is False:
            svd_fact = get_factorization(data=amr_count, n_comp=n_comp, nmf=False)
            nmf_fact = get_factorization(data=amr_count, n_comp=n_comp, nmf=True)
            pkl_file = 'nmf_count_'+ data_label + '.pkl'
            save_pickle_model(nmf_fact, fact_path, pkl_file)
            pkl_file = 'svd_count_' + data_label + '.pkl'
            save_pickle_model(svd_fact, fact_path, pkl_file)
        else: 
            # lsa
            pkl_file = 'svd_count_' + data_label + '.pkl'
            svd_fact = load_from_pickle(fact_path, pkl_file)
            # nmf
            pkl_file = 'nmf_count_' + data_label + '.pkl'
            nmf_fact = load_from_pickle(fact_path, pkl_file)

        amr_lsa = svd_fact.transform(amr_count)
        amr_nmf = nmf_fact.transform(amr_count)
        amr_lsa = pd.DataFrame(amr_lsa,index=amr_count.index)
        amr_nmf = pd.DataFrame(amr_nmf,index=amr_count.index)
        return amr_count, amr_lsa, amr_nmf
    
    



# obtener las transformaciones y reduccion de dimensión para datos de test
def transforms_test(amr_count, data_label, only_counts = False, fact_path = os.getcwd()):

    if only_counts is False:
        # lsa
        pkl_file = 'svd_' + data_label + '.pkl'
        svd_fact = load_from_pickle(fact_path, pkl_file)
        # nmf
        pkl_file = 'nmf_' + data_label + '.pkl'
        nmf_fact = load_from_pickle(fact_path, pkl_file)
        # realiza TF-IDF y factorizacion
        pkl_file = 'tfidf_'+ data_label + '.pkl'
        tfidf_vect = load_from_pickle(fact_path, pkl_file)
        X = tfidf_vect.transform(amr_count)
        tfidf = X.toarray()
        the_data = pd.DataFrame(tfidf,columns=tfidf_vect.get_feature_names_out(),index=amr_count.index)

        amr_lsa = svd_fact.transform(the_data)
        amr_nmf = nmf_fact.transform(the_data)
        amr_lsa = pd.DataFrame(amr_lsa,index=amr_count.index)
        amr_nmf = pd.DataFrame(amr_nmf,index=amr_count.index)
    else:
        the_data = amr_count
        # lsa
        pkl_file = 'svd_count_' + data_label + '.pkl'
        svd_fact = load_from_pickle(fact_path, pkl_file)
        # nmf
        pkl_file = 'nmf_count_' + data_label + '.pkl'
        nmf_fact = load_from_pickle(fact_path, pkl_file)
        amr_lsa = svd_fact.transform(the_data)
        amr_nmf = nmf_fact.transform(the_data)
        amr_lsa = pd.DataFrame(amr_lsa,index=amr_count.index)
        amr_nmf = pd.DataFrame(amr_nmf,index=amr_count.index)

    return the_data, amr_lsa, amr_nmf

from sklearn.model_selection import GridSearchCV

# obtiene parámetros óptimos del clasificador binario con grid search y CV
def search_params_binary(clf, X, y, param_grid, cv=5, scoring='accuracy'):
    # Realizar Grid Search y Cross Validation
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_model, best_params, best_score


def train_and_save_binary_models(X_train, y_train, label_file, output_dir="modelos_fenotipo"):
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Definir clasificadores y grids
    models_param_grid = {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [50, 100, 200],
             "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5,10]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7],
             "weights": ["uniform", "distance"]}
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10],
             "solver": ["liblinear", "lbfgs"]}
        ),
        "SVC": (
            SVC(probability=True, random_state=42),
            {"C": [0.1, 1, 10],
             "kernel": ["linear", "rbf"],
             "gamma": ["scale", "auto"]}
        ),
        "LASSO": (
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1, 10, 100],   # fuerza de regularización
            "solver": ["liblinear", "saga"] # solvers que soportan L1
            }
        )#,
        #"XGBoost": (
        #XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        #    {"n_estimators": [100, 200, 300],
        #    "max_depth": [3, 5, 7],
        #    "learning_rate": [0.01, 0.1, 0.2],
        #    "subsample": [0.8, 1.0],
        #    "colsample_bytree": [0.8, 1.0]}
    #)
    }

    resultados = []

    # Entrenar cada modelo con GridSearchCV
    for nombre, (modelo, param_grid) in models_param_grid.items():
        print(f"Entrenando {nombre}...")
        best_model, best_params, best_score = search_params_binary(modelo, X_train, y_train, param_grid, cv=5, scoring='accuracy')
        
        # Guardar modelo en pickle
        modelo_path = os.path.join(output_dir, f"{nombre}_{label_file}.pkl")
        with open(modelo_path, "wb") as f:
            pickle.dump(best_model, f)

        resultados.append((nombre, best_params, best_score))

    # Guardar parámetros en archivo de texto
    resultados_path = os.path.join(output_dir, "parametros_optimos.txt")
    with open(resultados_path, "w") as f:
        for nombre, params, score in resultados:
            f.write(f"Modelo: {nombre}_{label_file}\n")
            f.write(f"Mejores parámetros: {params}\n")
            f.write(f"Mejor score (CV): {score:.4f}\n")
            f.write("-"*40 + "\n")

    #print(f"Modelos y parámetros guardados en '{output_dir}'.")

    return resultados







# obtiene parámetros óptimos del clasificador ordinal con grid search y CV
def search_params_ordinal(ordinal_cl, X, y, param_grid, cv=5, scoring = 'accuracy'):
    # Realizar Grid Search y Cross Validation
    grid_search = GridSearchCV(ordinal_cl,
                            param_grid = param_grid,
                            cv = cv,
                            scoring = 'accuracy',
                            n_jobs = -1
                            )

    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return best_params, best_score

# obtiene estadísticas del desempeño del clasificador ordinal con validación cruzada
# RECORDATORIO PARA MI: cambiar el nombre de la función, porque realmente se puede usar con cualquier modelo, no
# solamente el que se adapta para clasificación ordinal...
def ordinal_clas_stat(ordinal_cl, X, y, score, rand_state = None, n_repeats=5, n_split = 5):
    # Realizar la validación cruzada repetida
    cv_scores = []
    for _ in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=rand_state)
        scores = cross_val_score(ordinal_cl, X, y, cv=cv, scoring=score)
        cv_scores.extend(-scores)

    # Calcular media y desviación estándar de los puntajes
    mean_score = sum(cv_scores) / len(cv_scores)
    std_score = np.std(cv_scores)
    return mean_score, std_score, cv_scores

def ordinal_clas_stat2(ordinal_cl, X, y, rand_state = None, n_repeats=5, n_split = 5):
    # Realizar la validación cruzada repetida
    mae_cv_scores = []
    rmse_cv_scores = []
    for _ in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=rand_state)
        scs = ('neg_mean_absolute_error', 'neg_root_mean_squared_error') # MAE y RMSE
        scores = cross_validate(ordinal_cl, X, y, cv=cv, scoring=scs)
        mae_cv_scores.extend(-scores['test_neg_mean_absolute_error'])
        rmse_cv_scores.extend(-scores['test_neg_root_mean_squared_error'])

    # Calcular media y desviación estándar de los puntajes
    mean_mae = np.mean(mae_cv_scores)
    mean_rmse = np.mean(rmse_cv_scores)
    std_mae = np.std(mae_cv_scores)
    std_rmse = np.std(rmse_cv_scores)
    return mean_mae, mean_rmse, std_mae, std_rmse
    
def std_cv_scores(model, X, y, rand_state = None, n_repeats=5, n_split = 5):
    # Realizar la validación cruzada repetida
    mae_cv_scores = []
    rmse_cv_scores = []
    for _ in range(n_repeats):
        cv = KFold(n_splits=n_split, shuffle=True, random_state=rand_state)
        scs = ('neg_mean_absolute_error', 'neg_root_mean_squared_error') # MAE y RMSE
        scores = cross_validate(model, X, y, cv=cv, scoring=scs)
        mae_cv_scores.extend(-scores['test_neg_mean_absolute_error'])
        rmse_cv_scores.extend(-scores['test_neg_root_mean_squared_error'])

    # Calcular media y desviación estándar de los puntajes
    mean_mae = np.mean(mae_cv_scores)
    mean_rmse = np.mean(rmse_cv_scores)
    std_mae = np.std(mae_cv_scores)
    std_rmse = np.std(rmse_cv_scores)
    return mean_mae, mean_rmse, std_mae, std_rmse

def std_cv_scores(model, X, y, rand_state = None, n_repeats=5, n_split = 5):
    # Realizar la validación cruzada repetida
    mae_cv_scores = []
    rmse_cv_scores = []
    for _ in range(n_repeats):
        cv = KFold(n_splits=n_split, shuffle=True, random_state=rand_state)
        scs = ('neg_mean_absolute_error', 'neg_root_mean_squared_error') # MAE y RMSE
        scores = cross_validate(ordinal_cl, X, y, cv=cv, scoring=scs)
        mae_cv_scores.extend(-scores['test_neg_mean_absolute_error'])
        rmse_cv_scores.extend(-scores['test_neg_root_mean_squared_error'])

    # Calcular media y desviación estándar de los puntajes
    mean_mae = np.mean(mae_cv_scores)
    mean_rmse = np.mean(rmse_cv_scores)
    std_mae = np.std(mae_cv_scores)
    std_rmse = np.std(rmse_cv_scores)
    return mean_mae, mean_rmse, std_mae, std_rmse

def ordinal_encoding_nn(label, num_clases):
    ord_enc_y = []
    for k in label:
        shape_vector = np.zeros(num_clases)  
        shape_vector[:k+1] = 1 
        ord_enc_y.append(shape_vector)
    ord_enc_y = np.array(ord_enc_y)
    return ord_enc_y

def prediction2label_nn(pred: np.ndarray):
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1


def cross_val_metrics_nn(model, X, y, cv):
    kf = KFold(n_splits=cv)
    mae_scores = []
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16, verbose=0)

        y_pred = prediction2label_nn(model.predict(X_val_fold))
        y_val_lab = prediction2label_nn(y_val_fold)
        mae = mean_absolute_error(y_val_lab, y_pred)
        rmse = root_mean_squared_error(y_val_lab, y_pred)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        # Calcular los valores promedio y desviación estándar
        mae_mean = np.mean(mae_scores)
        mae_std = np.std(mae_scores)
        rmse_mean = np.mean(rmse_scores)
        rmse_std = np.std(rmse_scores)

    return mae_mean, mae_std, rmse_mean, rmse_std

def cross_val_metrics_nn_standard(model, X, y, cv):
    kf = KFold(n_splits=cv)
    mae_scores = []
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16, verbose=0)

        y_pred = np.argmax(model.predict(X_val_fold), axis=-1)
        mae = mean_absolute_error(y_val_fold, y_pred)
        rmse = root_mean_squared_error(y_val_fold, y_pred)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        # Calcular los valores promedio y desviación estándar
        mae_mean = np.mean(mae_scores)
        mae_std = np.std(mae_scores)
        rmse_mean = np.mean(rmse_scores)
        rmse_std = np.std(rmse_scores)

    return mae_mean, mae_std, rmse_mean, rmse_std

def results_report(y_test, y_hat, label_enc):
    mae = mean_absolute_error(y_test, y_hat)
    rmse = root_mean_squared_error(y_test, y_hat)
    print('MAE: ', np.round(mae,3), '\nRMSE: ',np.round(rmse,3))

    # Calcular la matriz de confusión
    print(metrics.classification_report(label_enc.inverse_transform(y_test), label_enc.inverse_transform(y_hat)))

    cm = metrics.confusion_matrix(label_enc.inverse_transform(y_test), label_enc.inverse_transform(y_hat), normalize = 'true')
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=.8)
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap='bwr', vmin=0, vmax=1,
                    xticklabels=label_enc.inverse_transform(np.unique(y_test)), 
                    yticklabels=label_enc.inverse_transform(np.unique(y_test)))
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    #plt.title('Confusion matrix')
    plt.xlabel('Predicted MIC', fontsize=10)
    plt.ylabel('True MIC', fontsize=10)
    plt.show()


resistance_thresholds = {
    "Klebsiella": 4,
    "Escherichia": 4,
    "Salmonella": 2,
    "Streptococcus": 1,
    "Staphylococcus": 2,
    "Pseudomonas": 8,
    "Acinetobacter": 8,
    "Campylobacter": 4,
    "Neisseria": 1
}
def assign_phenotype(data_mic, spec_col = 'gtdb_species', mic_col = 'MIC_pred', thresholds=resistance_thresholds):
    feno = []
    for index, row in data_mic.iterrows(): 
        specie = row[spec_col].split()[0]
        mic = row[mic_col]

        if pd.isna(mic) or pd.isna(specie):
            val = 'Unknown'
        else:
            threshold = thresholds.get(specie)
            val = 'Resistant' if float(mic) >= threshold else 'Susceptible'
        
        feno.append(val)
    
    return feno



def results_report_phenotype(y_test, y_hat):
    # Calcular la matriz de confusión
    print(metrics.classification_report(y_test, y_hat))

    cm = metrics.confusion_matrix(y_test, y_hat, normalize = 'true')
    plt.figure(figsize=(5, 4))
    sns.set(font_scale=.8)
    ax = sns.heatmap(cm, annot=True, fmt='.2f', cmap='bwr', vmin=0, vmax=1,
                        xticklabels=np.unique(y_test), 
                        yticklabels=np.unique(y_test))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    #plt.title('Confusion matrix')
    plt.xlabel('Predicted Phenotype', fontsize=10)
    plt.ylabel('True Phenotype', fontsize=10)
    plt.show()
    
def evaluate_models(X_test, y_test, data_label, models_dir="modelos_fenotipo"):
    
    resultados = []

    # Buscar todos los archivos .pkl en el directorio
    for file in os.listdir(models_dir):
        if file.endswith(data_label+".pkl"):
            modelo_path = os.path.join(models_dir, file)
            nombre_modelo = file.replace(".pkl", "")

            # Cargar modelo
            with open(modelo_path, "rb") as f:
                modelo = pickle.load(f)

            # Predecir
            y_pred = modelo.predict(X_test)

            # Calcular métricas (usando weighted y macro)
            metrics = {
                "Modelo": nombre_modelo,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 (weighted)": f1_score(y_test, y_pred, average="weighted"),
                "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
                "Precision (weighted)": precision_score(y_test, y_pred, average="weighted"),
                "Recall (weighted)": recall_score(y_test, y_pred, average="weighted")
            }
            resultados.append(metrics)
            results_report_phenotype(y_test, y_pred)

    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Guardar resultados en archivo de texto
    resultados_path = os.path.join(models_dir, "scores_modelos.txt")
    with open(resultados_path, "w") as f:
        f.write(df_resultados.to_string(index=False))

    print(f"Scores guardados en '{resultados_path}'.")
    
    
    return df_resultados



from scipy.sparse import csr_matrix
def sparse_fileter (counts, metadata,label,antibiotic = True):
    """
    counts: a sparse matrix
    metadata: csv metadata
    label: antibiotic
    """
    if antibiotic == True:
        metadata_filtered = metadata[metadata["antibiotic"]==label]
        metadata_filtered = metadata_filtered[metadata_filtered["phenotype"]!='intermediate']
    
    else:   
    
        metadata_filtered = metadata[metadata["gtdb_genus"]==label]
        metadata_filtered = metadata_filtered[metadata_filtered["phenotype"]!='intermediate']
    
    # Reconstruir matriz sparse desde el npz
    counts_matrix = csr_matrix(
        (counts["data"], counts["indices"], counts["indptr"]),
        shape=counts["shape"]
    )
    # Obtener posiciones numéricas de las filas filtradas
    rows_to_keep = np.where(metadata.index.isin(metadata_filtered.index))[0]

    # Filtrar matriz sparse
    counts_filtered = counts_matrix[rows_to_keep]
    
    # filtrar columnas de zeros
    #cols_nonzero = np.array((counts_filtered != 0).sum(axis=0)).ravel() > 0

    #counts_filtered_reduced = counts_filtered[:, cols_nonzero]
    
    return  metadata_filtered, counts_filtered#_reduced


from urllib.request import urlopen
from io import BytesIO


def get_data2(data_path = 'https://zenodo.org/records/19824655/files/'):
    
    data_url = data_path + 'pangenome.npz?download=1'
    meta_url = data_path + 'pangenome.metadata.csv'
    
    #metadata
    metadata = pd.read_csv(meta_url, low_memory=False,  index_col=0)
    #counts
    response = urlopen(data_url)
    counts = np.load(BytesIO(response.read()))
    return metadata, counts


def transforms2(amr_count,
               data_label,
               only_counts=False,
               exist_fact=False,
               exist_vect=False,
               fact_path=os.getcwd(),
               n_comp=100):

    """
    Realiza transformaciones TF-IDF + reducción dimensional (SVD/NMF)
    sobre una matriz de conteos AMR.

    La función está optimizada para trabajar con matrices sparse
    (`csr_matrix`) sin densificarlas innecesariamente.

    Parameters
    ----------
    amr_count : csr_matrix o pd.DataFrame
        Matriz de conteos AMR.

    data_label : str
        Etiqueta usada para guardar/cargar modelos.

    only_counts : bool, default=False
        Si True, omite TF-IDF y trabaja directamente con los conteos.

    exist_fact : bool, default=False
        Si True, carga modelos SVD/NMF previamente guardados.

    exist_vect : bool, default=False
        Si True, carga un transformador TF-IDF previamente guardado.

    fact_path : str
        Ruta donde guardar/cargar modelos.

    n_comp : int, default=100
        Número de componentes para SVD y NMF.

    Returns
    -------
    if only_counts=False:
        X_tfidf : sparse matrix
        tfidf_vect : TfidfTransformer
        amr_lsa : pd.DataFrame
        amr_nmf : pd.DataFrame

    if only_counts=True:
        amr_count : original input
        amr_lsa : pd.DataFrame
        amr_nmf : pd.DataFrame
    """

    if not os.path.exists(fact_path):
        os.makedirs(fact_path)

    # ==========================================================
    # PIPELINE CON TF-IDF
    # ==========================================================
    if not only_counts:

        # ------------------------------------------------------
        # Obtener o cargar TF-IDF
        # ------------------------------------------------------
        if not exist_vect:

            tfidf_vect, X_tfidf = get_tfidf(amr_count)
            print("tfidf hecho")
            

            pkl_file = f'tfidf_{data_label}.pkl'
            save_pickle_model(tfidf_vect, fact_path, pkl_file)

        else:

            pkl_file = f'tfidf_{data_label}.pkl'
            tfidf_vect = load_from_pickle(fact_path, pkl_file)

            # IMPORTANTE:
            # NO usar .toarray()
            X_tfidf = tfidf_vect.transform(amr_count)

        # ------------------------------------------------------
        # Obtener o cargar modelos de factorización
        # ------------------------------------------------------
        if not exist_fact:
            print("inicia svd")

            svd_fact = get_factorization(
                data=X_tfidf,
                n_comp=n_comp,
                nmf=False
            )
            print("terminó svd")

            nmf_fact = get_factorization(
                data=X_tfidf,
                n_comp=n_comp,
                nmf=True
            )

            save_pickle_model(
                nmf_fact,
                fact_path,
                f'nmf_{data_label}.pkl'
            )

            save_pickle_model(
                svd_fact,
                fact_path,
                f'svd_{data_label}.pkl'
            )

        else:

            svd_fact = load_from_pickle(
                fact_path,
                f'svd_{data_label}.pkl'
            )

            nmf_fact = load_from_pickle(
                fact_path,
                f'nmf_{data_label}.pkl'
            )

        # ------------------------------------------------------
        # Aplicar transformaciones
        # ------------------------------------------------------
        print("amr inicia")

        amr_lsa = svd_fact.transform(X_tfidf)
        amr_nmf = nmf_fact.transform(X_tfidf)

        # ------------------------------------------------------
        # Convertir SOLO resultados reducidos a DataFrame
        # ------------------------------------------------------
        if hasattr(amr_count, "index"):
            idx = amr_count.index
        else:
            idx = None

        amr_lsa = pd.DataFrame(amr_lsa, index=idx)
        amr_nmf = pd.DataFrame(amr_nmf, index=idx)

        return X_tfidf, tfidf_vect, amr_lsa, amr_nmf

    # ==========================================================
    # PIPELINE SOLO CON CONTEOS
    # ==========================================================
    else:

        if not exist_fact:

            svd_fact = get_factorization(
                data=amr_count,
                n_comp=n_comp,
                nmf=False
            )

            nmf_fact = get_factorization(
                data=amr_count,
                n_comp=n_comp,
                nmf=True
            )

            save_pickle_model(
                nmf_fact,
                fact_path,
                f'nmf_count_{data_label}.pkl'
            )

            save_pickle_model(
                svd_fact,
                fact_path,
                f'svd_count_{data_label}.pkl'
            )

        else:

            svd_fact = load_from_pickle(
                fact_path,
                f'svd_count_{data_label}.pkl'
            )

            nmf_fact = load_from_pickle(
                fact_path,
                f'nmf_count_{data_label}.pkl'
            )

        # ------------------------------------------------------
        # Aplicar transformaciones
        # ------------------------------------------------------
        amr_lsa = svd_fact.transform(amr_count)
        amr_nmf = nmf_fact.transform(amr_count)

        # ------------------------------------------------------
        # Índices opcionales
        # ------------------------------------------------------
        if hasattr(amr_count, "index"):
            idx = amr_count.index
        else:
            idx = None

        amr_lsa = pd.DataFrame(amr_lsa, index=idx)
        amr_nmf = pd.DataFrame(amr_nmf, index=idx)

        return amr_count, amr_lsa, amr_nmf

def transforms_test2(amr_count, data_label, only_counts = False, fact_path = os.getcwd()):

    if only_counts is False:
        # lsa
        pkl_file = 'svd_' + data_label + '.pkl'
        svd_fact = load_from_pickle(fact_path, pkl_file)
        # nmf
        #pkl_file = 'nmf_' + data_label + '.pkl'
        #nmf_fact = load_from_pickle(fact_path, pkl_file)
        # realiza TF-IDF y factorizacion
        pkl_file = 'tfidf_'+ data_label + '.pkl'
        tfidf_vect = load_from_pickle(fact_path, pkl_file)
        X = tfidf_vect.transform(amr_count)
        tfidf = X.toarray()
        the_data = pd.DataFrame(tfidf,columns=tfidf_vect.get_feature_names_out(),index=amr_count.index)

        amr_lsa = svd_fact.transform(the_data)
        #amr_nmf = nmf_fact.transform(the_data)
        amr_lsa = pd.DataFrame(amr_lsa,index=amr_count.index)
        #amr_nmf = pd.DataFrame(amr_nmf,index=amr_count.index)
    else:
        the_data = amr_count
        # lsa
        pkl_file = 'svd_count_' + data_label + '.pkl'
        svd_fact = load_from_pickle(fact_path, pkl_file)
        # nmf
        pkl_file = 'nmf_count_' + data_label + '.pkl'
        nmf_fact = load_from_pickle(fact_path, pkl_file)
        amr_lsa = svd_fact.transform(the_data)
        #amr_nmf = nmf_fact.transform(the_data)
        amr_lsa = pd.DataFrame(amr_lsa,index=amr_count.index)
        #amr_nmf = pd.DataFrame(amr_nmf,index=amr_count.index)

    return the_data, amr_lsa#, amr_nmf
