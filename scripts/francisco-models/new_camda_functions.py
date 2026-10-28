#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funciones para el pipeline de predicción de fenotipo (resistente / susceptible):

    1. Carga de datos ............ get_data2
    2. Filtrado .................. sparse_fileter, make_label, available_combinations
    3. Representación ............ get_tfidf, get_factorization, transforms2, transforms_test2
    4. Respuesta y partición ..... get_y_ordinal, split_stratified_into_train_val_test
    5. Entrenamiento ............. search_params_binary, train_and_save_binary_models
    6. Evaluación ................ results_report_phenotype, evaluate_models
    Utilidades ................... save_pickle_model, load_from_pickle

Basado en camda_functions_2026.py (autor original: victor) y en las funciones
de modelos_fenotipo_space.ipynb, sin duplicados.
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD, NMF


# ==========================================================================
# Utilidades: guardar / cargar objetos con pickle
# ==========================================================================

def save_pickle_model(model_obj, file_path, model_name):
    """
    Guarda cualquier objeto de Python (modelo, transformador, etc.) en un
    archivo pickle.

    Parameters
    ----------
    model_obj : objeto a guardar.
    file_path : str, carpeta destino.
    model_name : str, nombre del archivo (p. ej. 'svd_erythromycin.pkl').
    """
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name, 'wb') as file:
        pickle.dump(model_obj, file)


def load_from_pickle(file_path, model_name):
    """
    Carga un objeto guardado previamente con save_pickle_model.

    Parameters
    ----------
    file_path : str, carpeta donde está el archivo.
    model_name : str, nombre del archivo .pkl.

    Returns
    -------
    El objeto guardado.
    """
    pkl_name = os.path.join(file_path, model_name)
    with open(pkl_name, 'rb') as file:
        fact_model = pickle.load(file)

    return fact_model


# ==========================================================================
# 1. Carga de datos
# ==========================================================================

def get_data2(data_path='/files2/camda2026/data/'):
    """
    Lee los datos del pangenoma CAMDA 2026 desde una carpeta local del servidor.

    Parameters
    ----------
    data_path : str
        Carpeta que contiene 'pangenome.npz' y 'pangenome.metadata.csv'.

    Returns
    -------
    metadata : pd.DataFrame
        Una fila por muestra (genoma + antibiótico), indexada por 'accession'.
        OJO: el índice tiene valores repetidos, porque el mismo genoma aparece
        una vez por cada antibiótico con el que se probó.
    counts : numpy NpzFile
        Componentes de la matriz sparse de conteos ('data', 'indices',
        'indptr', 'shape'). La fila i corresponde a la fila i de metadata.
        Los arreglos se leen del disco hasta que se usan.
    """
    data_file = os.path.join(data_path, 'pangenome.npz')
    meta_file = os.path.join(data_path, 'pangenome.metadata.csv')

    # metadata
    metadata = pd.read_csv(meta_file, low_memory=False, index_col=0)
    # counts
    counts = np.load(data_file)
    return metadata, counts


# ==========================================================================
# 2. Filtrado por antibiótico y género
# ==========================================================================

def available_combinations(metadata, intermediate='resistant', drop_test=True):
    """
    Muestra cuántas muestras hay por combinación antibiótico / género, con el
    mismo criterio que usa sparse_fileter. Sirve para elegir qué modelos
    tiene sentido entrenar. No modifica metadata.

    Parameters
    ----------
    metadata : pd.DataFrame (salida de get_data2).
    intermediate : 'resistant' o 'drop'
        'resistant' -> cuenta las 'intermediate' como resistentes.
        'drop'      -> no las cuenta.
    drop_test : bool, si True no cuenta las filas con 'test' en 'dataset'.

    Returns
    -------
    pd.DataFrame con columnas antibiotic, gtdb_genus, n, resistant, susceptible,
    ordenado de mayor a menor n.
    """
    # mismo criterio que sparse_fileter (no modifica metadata)
    if drop_test:
        metadata = metadata[~metadata["dataset"].str.contains("test", na=False)]

    # trabajar sobre una copia de la columna, nunca sobre la tabla original
    fenotipo = _recode_intermediate(metadata["adjusted_phenotype"], intermediate)

    # tabla de conteos por fenotipo para cada antibiótico/género (los NaN se ignoran)
    tabla = (metadata[["antibiotic", "gtdb_genus"]].assign(phenotype=fenotipo)
             .groupby(["antibiotic", "gtdb_genus", "adjusted_phenotype"]).size()
             .unstack(fill_value=0))
    tabla.insert(0, "n", tabla.sum(axis=1))
    return tabla.sort_values("n", ascending=False).reset_index()


def _recode_intermediate(phenotype, intermediate='resistant'):
    """
    Devuelve una COPIA de la columna phenotype con las muestras 'intermediate'
    recodificadas ('resistant') o marcadas como NaN para descartarlas ('drop').
    """
    if intermediate not in ("resistant", "drop"):
        raise ValueError("intermediate debe ser 'resistant' o 'drop'.")
    nuevo = "resistant" if intermediate == "resistant" else np.nan
    return phenotype.replace("intermediate", nuevo)


def make_label(antibiotic, genus=None):
    """
    Construye la etiqueta con la que se nombran los archivos de un modelo.

    Ejemplos: make_label('erythromycin', 'Streptococcus') -> 'erythromycin_Streptococcus'
              make_label('erythromycin')                  -> 'erythromycin'
    """
    return antibiotic if genus is None else f"{antibiotic}_{genus}"


def sparse_fileter(counts, metadata, antibiotic, genus=None, intermediate='resistant', drop_test=True):
    """
    Filtra metadata y matriz de conteos para quedarse con UN antibiótico y,
    opcionalmente, UN género.

    Uso:
        # solo antibiótico
        md, X = sparse_fileter(counts, metadata, antibiotic='erythromycin')
        # antibiótico + género
        md, X = sparse_fileter(counts, metadata, antibiotic='erythromycin',
                               genus='Streptococcus')

    Parameters
    ----------
    counts : NpzFile o dict con 'data', 'indices', 'indptr', 'shape'
        Matriz de conteos (salida de get_data2).
    metadata : pd.DataFrame
        Metadata alineada fila por fila con counts.
    antibiotic : str
        Valor de la columna 'antibiotic' a conservar.
    genus : str o None, default=None
        Valor de la columna 'gtdb_genus' a conservar. None = no filtrar por género.
    intermediate : 'resistant' (default) o 'drop'
        Qué hacer con las muestras de fenotipo 'intermediate':
        'resistant' -> se cuentan como resistentes (problema binario).
        'drop'      -> se eliminan.
        El cambio se hace sobre una copia; metadata NO se modifica.
        Las muestras sin fenotipo (NaN) se eliminan siempre.
    drop_test : bool, default=True
        Si True elimina las filas cuya columna 'dataset' contiene 'test'
        (p. ej. 'c25_test', 'c25_train c25_test'). False = conservarlas.

    Returns
    -------
    metadata_filtered : pd.DataFrame
        Copia filtrada. 'phenotype' ya viene recodificada y la columna
        'phenotype_original' conserva el valor original.
    counts_filtered : csr_matrix
        Misma cantidad de filas y mismo orden que metadata_filtered.
    """
    # Máscara booleana POR POSICIÓN. No se usa el índice 'accession' porque
    # está repetido (un genoma aparece una vez por antibiótico) y filtrar por
    # nombre traería filas de otros antibióticos.
    mask = metadata["antibiotic"] == antibiotic
    if genus is not None:
        mask &= metadata["gtdb_genus"] == genus
    # RESTRICCIÓN TEST: excluye las filas con 'test' en la columna dataset
    # (incluye etiquetas combinadas como 'c25_train c25_test'). Para entrenar
    # con ellas, llama con drop_test=False o borra estas dos líneas.
    if drop_test:
        mask &= ~metadata["dataset"].str.contains("test", na=False)
    # fenotipo recodificado (copia; la tabla original no se toca)
    fenotipo = _recode_intermediate(metadata["adjusted_phenotype"], intermediate)
    # muestras sin fenotipo (o 'intermediate' con 'drop') no sirven para entrenar ni evaluar
    mask &= fenotipo.notna()
    mask = mask.to_numpy()

    if not mask.any():
        generos = sorted(metadata.loc[metadata["antibiotic"] == antibiotic, "gtdb_genus"].dropna().unique())
        raise ValueError(f"No hay muestras para antibiotic='{antibiotic}', genus='{genus}'. "
                         f"Géneros disponibles para ese antibiótico: {generos}")

    # Reconstruir matriz sparse desde el npz
    counts_matrix = csr_matrix(
        (counts["data"], counts["indices"], counts["indptr"]),
        shape=tuple(counts["shape"])
    )
    if counts_matrix.shape[0] != len(metadata):
        raise ValueError(f"counts tiene {counts_matrix.shape[0]} filas y metadata {len(metadata)}; "
                         "no están alineados.")

    # Filtrar filas (mismas posiciones en metadata y en la matriz)
    rows_to_keep = np.where(mask)[0]
    metadata_filtered = metadata.iloc[rows_to_keep].copy()
    metadata_filtered["phenotype_original"] = metadata_filtered["adjusted_phenotype"]
    metadata_filtered["adjusted_phenotype"] = fenotipo.iloc[rows_to_keep].to_numpy()
    counts_filtered = counts_matrix[rows_to_keep]

    # Con un solo antibiótico cada genoma debería aparecer una vez; si no,
    # el mismo genoma podría caer en train y en test a la vez.
    n_rep = metadata_filtered.index.duplicated().sum()
    if n_rep:
        warnings.warn(f"{n_rep} accessions repetidos tras el filtrado ({antibiotic}, {genus}).")

    return metadata_filtered, counts_filtered


# ==========================================================================
# 3. Representación: TF-IDF + factorización (SVD / NMF)
# ==========================================================================

def _row_index(amr_count, index=None):
    """
    Decide el índice de filas de las matrices reducidas (amr_lsa / amr_nmf).
    Prioridad: el `index` dado -> amr_count.index (DataFrame) -> None (0..n-1).
    Verifica que el índice tenga tantas filas como la matriz.
    """
    if index is None:
        return amr_count.index if hasattr(amr_count, "index") else None
    if len(index) != amr_count.shape[0]:
        raise ValueError(f"index tiene {len(index)} elementos y la matriz {amr_count.shape[0]} filas.")
    return index


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


def get_factorization(data, n_comp=100, nmf=True):
    """
    Ajusta un modelo de reducción de dimensión sobre la matriz (sparse o densa).

    Parameters
    ----------
    data : matriz de entrada (conteos o TF-IDF).
    n_comp : int, número de componentes.
    nmf : bool
        True  -> NMF (factorización no negativa; puede tardar mucho).
        False -> TruncatedSVD (LSA).

    Returns
    -------
    fact_model : modelo ajustado (usar fact_model.transform(X)).
    """
    if nmf == False:
        fact_model = TruncatedSVD(n_components=n_comp)
        fact_model.fit(data)
    else:
        fact_model = NMF(n_components=n_comp, init=None, max_iter=12000)
        fact_model.fit(data)

    return fact_model


def transforms2(amr_count,
                data_label,
                only_counts=False,
                exist_fact=False,
                exist_vect=False,
                fact_path=os.getcwd(),
                n_comp=100,
                index=None):
    """
    Realiza transformaciones TF-IDF + reducción dimensional (SVD/NMF)
    sobre una matriz de conteos AMR, y guarda los modelos ajustados en
    fact_path como tfidf_<label>.pkl, svd_<label>.pkl y nmf_<label>.pkl
    (o svd_count_/nmf_count_ si only_counts=True).

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
        Si True, carga modelos SVD/NMF previamente guardados en lugar de ajustarlos.
    exist_vect : bool, default=False
        Si True, carga un transformador TF-IDF previamente guardado.
    fact_path : str
        Ruta donde guardar/cargar modelos.
    n_comp : int, default=100
        Número de componentes para SVD y NMF.
    index : array-like o None
        Índice para las filas de amr_lsa / amr_nmf; normalmente
        metadata_filtered.index (accessions), para que X y Y queden con el
        mismo índice. Si es None se usa amr_count.index (si es DataFrame)
        o 0..n-1 (si es matriz sparse).

    Returns
    -------
    if only_counts=False:
        X_tfidf : sparse matrix
        tfidf_vect : TfidfTransformer
        amr_lsa : pd.DataFrame  (muestras x n_comp, SVD)
        amr_nmf : pd.DataFrame  (muestras x n_comp, NMF)

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
            svd_fact = get_factorization(data=X_tfidf, n_comp=n_comp, nmf=False)
            save_pickle_model(svd_fact, fact_path, f'svd_{data_label}.pkl')
            print("terminó svd")

            print("inicia nmf")
            nmf_fact = get_factorization(data=X_tfidf, n_comp=n_comp, nmf=True)
            save_pickle_model(nmf_fact, fact_path, f'nmf_{data_label}.pkl')
            print("terminó nmf")

        else:
            svd_fact = load_from_pickle(fact_path, f'svd_{data_label}.pkl')
            nmf_fact = load_from_pickle(fact_path, f'nmf_{data_label}.pkl')

        # ------------------------------------------------------
        # Aplicar transformaciones
        # ------------------------------------------------------
        print("amr inicia")

        amr_lsa = svd_fact.transform(X_tfidf)
        amr_nmf = nmf_fact.transform(X_tfidf)

        # ------------------------------------------------------
        # Convertir SOLO resultados reducidos a DataFrame
        # ------------------------------------------------------
        idx = _row_index(amr_count, index)

        amr_lsa = pd.DataFrame(amr_lsa, index=idx)
        amr_nmf = pd.DataFrame(amr_nmf, index=idx)

        return X_tfidf, tfidf_vect, amr_lsa, amr_nmf

    # ==========================================================
    # PIPELINE SOLO CON CONTEOS
    # ==========================================================
    else:

        if not exist_fact:
            svd_fact = get_factorization(data=amr_count, n_comp=n_comp, nmf=False)
            save_pickle_model(svd_fact, fact_path, f'svd_count_{data_label}.pkl')

            nmf_fact = get_factorization(data=amr_count, n_comp=n_comp, nmf=True)
            save_pickle_model(nmf_fact, fact_path, f'nmf_count_{data_label}.pkl')

        else:
            svd_fact = load_from_pickle(fact_path, f'svd_count_{data_label}.pkl')
            nmf_fact = load_from_pickle(fact_path, f'nmf_count_{data_label}.pkl')

        # ------------------------------------------------------
        # Aplicar transformaciones
        # ------------------------------------------------------
        amr_lsa = svd_fact.transform(amr_count)
        amr_nmf = nmf_fact.transform(amr_count)

        # ------------------------------------------------------
        # Índices opcionales
        # ------------------------------------------------------
        idx = _row_index(amr_count, index)

        amr_lsa = pd.DataFrame(amr_lsa, index=idx)
        amr_nmf = pd.DataFrame(amr_nmf, index=idx)

        return amr_count, amr_lsa, amr_nmf


def transforms_test2(amr_count, data_label, only_counts=False, fact_path=os.getcwd(), index=None):
    """
    Aplica a datos nuevos los modelos TF-IDF / SVD / NMF guardados por
    transforms2 (no reajusta nada). Trabaja con la matriz sparse sin densificarla.

    Parameters
    ----------
    amr_count : csr_matrix, conteos de las muestras nuevas (mismas columnas).
    data_label : str, la misma etiqueta usada en transforms2.
    only_counts : bool, True si los modelos se ajustaron sin TF-IDF.
    fact_path : str, carpeta donde están los .pkl.
    index : array-like o None, índice para las filas (ver transforms2).

    Returns
    -------
    the_data : matriz TF-IDF sparse (o los conteos si only_counts=True)
    amr_lsa  : pd.DataFrame
    amr_nmf  : pd.DataFrame
    """
    prefix = 'count_' if only_counts else ''

    svd_fact = load_from_pickle(fact_path, f'svd_{prefix}{data_label}.pkl')
    nmf_fact = load_from_pickle(fact_path, f'nmf_{prefix}{data_label}.pkl')

    if only_counts:
        the_data = amr_count
    else:
        tfidf_vect = load_from_pickle(fact_path, f'tfidf_{data_label}.pkl')
        # IMPORTANTE:
        # NO usar .toarray()
        the_data = tfidf_vect.transform(amr_count)

    idx = _row_index(amr_count, index)
    amr_lsa = pd.DataFrame(svd_fact.transform(the_data), index=idx)
    amr_nmf = pd.DataFrame(nmf_fact.transform(the_data), index=idx)

    return the_data, amr_lsa, amr_nmf


# ==========================================================================
# 4. Respuesta y partición de datos
# ==========================================================================

def get_y_ordinal(metadata, ordinal=True):
    """
    Extrae la variable respuesta de la metadata.

    Parameters
    ----------
    metadata : pd.DataFrame (ya filtrada).
    ordinal : bool
        False -> devuelve la columna 'phenotype' (resistant / susceptible).
        True  -> devuelve el MIC recategorizado ('adjusted_mic') codificado
                 como clases ordinales, junto con el LabelEncoder.

    Returns
    -------
    ordinal=False : Y (pd.Series)
    ordinal=True  : (Y, le) donde Y es un DataFrame con columnas
                    MIC, MIC_recat, MIC_str, MIC_cat y le el LabelEncoder.
    """
    if ordinal == True:
        # MIC recategorizado, es decir, se juntan algunos valores, sobre todo de concentraciones pequeñas
        str_mic = [str(val) for val in metadata['adjusted_mic']]

        # categorización de la respuesta
        le = LabelEncoder()
        le.fit(str_mic)
        y_cat = le.transform(str_mic)

        Y = pd.DataFrame({'MIC': metadata['mic'], 'MIC_recat': metadata['adjusted_mic'],
                          'MIC_str': str_mic, 'MIC_cat': y_cat})

        return Y, le
    else:
        Y = metadata["phenotype"]
    return Y


def split_stratified_into_train_val_test(X, y, frac_train=0.6, frac_val=0.15, frac_test=0.25, std=True,
                                         two_subsets=False, random_state=None):
    """
    Divide los datos en entrenamiento / validación / prueba, estratificando
    por y (cada subconjunto conserva la proporción de clases).

    Uso típico (solo train/test 80/20, sin estandarizar):
        X_train, y_train, _, _, X_test, y_test, class_w, _ = \
            split_stratified_into_train_val_test(X, Y, .80, 0, .20, std=False,
                                                 two_subsets=True, random_state=42)

    Parameters
    ----------
    X : DataFrame o array de covariables.
    y : Series o array de respuestas.
    frac_train, frac_val, frac_test : float
        Proporciones de cada subconjunto; deben sumar 1.0.
    std : bool
        Si True estandariza (StandardScaler ajustado en train+val).
    two_subsets : bool
        Si True no se separa validación (x_val, y_val = None).
    random_state : int o None, semilla para train_test_split.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test, class_w, scaler
        class_w son pesos 'balanced' por clase (para datos desbalanceados);
        scaler es None si std=False.
    """
    if round(frac_train + frac_val + frac_test, 10) != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' %
                         (frac_train, frac_val, frac_test))

    # Separar prueba (test) del resto (temp = train + val)
    x_temp, x_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=(1.0 - (frac_train + frac_val)),
                                                      random_state=random_state)
    scaler = None
    if std:
        # estandarizar con parámetros de train+val y aplicarlos a test
        scaler = StandardScaler()
        x_temp = scaler.fit_transform(x_temp)
        x_test = scaler.transform(x_test)

    # pesos por clase para desbalance
    # (https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
    class_w = compute_class_weight('balanced', classes=np.unique(y_temp), y=y_temp)

    if two_subsets:
        x_train = x_temp
        y_train = y_temp
        x_val = None
        y_val = None
    else:
        # Separar validación del conjunto temp
        relative_frac_val = frac_val / (frac_train + frac_val)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, stratify=y_temp,
                                                          test_size=relative_frac_val, random_state=random_state)

    return x_train, y_train, x_val, y_val, x_test, y_test, class_w, scaler


# ==========================================================================
# 5. Entrenamiento de modelos de fenotipo
# ==========================================================================

def search_params_binary(clf, X, y, param_grid, cv=5, scoring='accuracy'):
    """
    Busca los mejores hiperparámetros de un clasificador con GridSearchCV
    (validación cruzada de cv particiones, en paralelo con todos los núcleos).

    Returns
    -------
    best_model : clasificador reajustado con los mejores parámetros en todo X.
    best_params : dict con los mejores parámetros.
    best_score : float, score promedio de CV de esos parámetros.
    """
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


def train_and_save_binary_models(X_train, y_train, label_file, output_dir="modelos_fenotipo",
                                 params_dir=None, cv=5, scoring='accuracy'):
    """
    Entrena 5 clasificadores (RandomForest, KNN, LogisticRegression, SVC, LASSO),
    cada uno con búsqueda de hiperparámetros por GridSearchCV, y los guarda.

    Archivos generados:
        <output_dir>/<Modelo>_<label_file>.pkl          un pickle por modelo
        <params_dir>/parametros_optimos_<label_file>.txt  mejores parámetros y score CV

    Parameters
    ----------
    X_train, y_train : datos de entrenamiento.
    label_file : str, etiqueta para nombrar los archivos (ver make_label).
    output_dir : str, carpeta para los modelos .pkl.
    params_dir : str o None, carpeta para el txt de parámetros (None = output_dir).
    cv : int, número de particiones de validación cruzada.
    scoring : str, métrica para elegir hiperparámetros (p. ej. 'accuracy', 'balanced_accuracy').

    Returns
    -------
    lista de tuplas (nombre, mejores_parámetros, mejor_score_cv).
    """
    os.makedirs(output_dir, exist_ok=True)
    params_dir = params_dir or output_dir
    os.makedirs(params_dir, exist_ok=True)

    # Clasificadores y sus rejillas de hiperparámetros
    models_param_grid = {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [50, 100, 200],
             "max_depth": [None, 10, 20],
             "min_samples_split": [2, 5, 10]}
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
             "solver": ["liblinear", "saga"]  # solvers que soportan L1
             }
        ),
    }

    resultados = []

    # Entrenar cada modelo con GridSearchCV y guardarlo
    for nombre, (modelo, param_grid) in models_param_grid.items():
        print(f"Entrenando {nombre}...")
        best_model, best_params, best_score = search_params_binary(modelo, X_train, y_train, param_grid,
                                                                   cv=cv, scoring=scoring)

        modelo_path = os.path.join(output_dir, f"{nombre}_{label_file}.pkl")
        with open(modelo_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"    {nombre}: score CV ({scoring}) = {best_score:.4f} | {best_params}")
        resultados.append((nombre, best_params, best_score))

    # Guardar parámetros óptimos (un archivo por etiqueta, para no sobrescribir otros modelos)
    resultados_path = os.path.join(params_dir, f"parametros_optimos_{label_file}.txt")
    with open(resultados_path, "w") as f:
        for nombre, params, score in resultados:
            f.write(f"Modelo: {nombre}_{label_file}\n")
            f.write(f"Mejores parámetros: {params}\n")
            f.write(f"Mejor score (CV, {scoring}): {score:.4f}\n")
            f.write("-" * 40 + "\n")

    return resultados


# ==========================================================================
# 6. Evaluación de modelos
# ==========================================================================

def results_report_phenotype(y_test, y_hat, title=None, save_path=None, show=True):
    """
    Imprime el reporte de clasificación (precision, recall, f1 por clase) y
    dibuja la matriz de confusión normalizada por fila (proporción de cada
    clase verdadera que se predijo como cada clase).

    Parameters
    ----------
    y_test : valores verdaderos.
    y_hat : valores predichos.
    title : str o None, título de la gráfica.
    save_path : str o None, si se da, guarda la figura ahí (p. ej. '.png').
    show : bool, si True muestra la figura (útil en notebook).

    Returns
    -------
    report : str, el reporte de clasificación en texto.
    """
    report = metrics.classification_report(y_test, y_hat, zero_division=0)
    print(report)

    # Matriz de confusión con las clases de y_test y y_hat, en el mismo orden en ejes
    labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_hat)]))
    cm = metrics.confusion_matrix(y_test, y_hat, labels=labels, normalize='true')

    fig = plt.figure(figsize=(5, 4))
    sns.set(font_scale=.8)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='bwr', vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if title:
        plt.title(title, fontsize=10)
    plt.xlabel('Predicted Phenotype', fontsize=10)
    plt.ylabel('True Phenotype', fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return report


def evaluate_models(X_test, y_test, data_label, models_dir="modelos_fenotipo",
                    results_dir=None, show_plots=True):
    """
    Evalúa en el conjunto de prueba todos los modelos guardados como
    <models_dir>/<Modelo>_<data_label>.pkl.

    Para cada modelo calcula Accuracy, F1 (weighted y macro), Precision y
    Recall (weighted), imprime el reporte y la matriz de confusión.

    Archivos generados en results_dir (o models_dir si es None):
        scores_<data_label>.txt / .csv            tabla de métricas de todos los modelos
        reporte_<Modelo>_<data_label>.txt         classification report por modelo
        matriz_confusion_<Modelo>_<data_label>.png  (solo si results_dir no es None)

    Parameters
    ----------
    X_test, y_test : datos de prueba (misma representación usada al entrenar).
    data_label : str, etiqueta de los modelos a evaluar.
    models_dir : str, carpeta con los .pkl.
    results_dir : str o None, carpeta para métricas y gráficas.
    show_plots : bool, mostrar las gráficas (True en notebook, False en terminal).

    Returns
    -------
    df_resultados : pd.DataFrame con una fila por modelo.
    """
    out_dir = results_dir or models_dir
    os.makedirs(out_dir, exist_ok=True)

    resultados = []
    sufijo = f"_{data_label}.pkl"

    # Buscar los modelos de esta etiqueta
    for file in sorted(os.listdir(models_dir)):
        if not file.endswith(sufijo):
            continue
        modelo_path = os.path.join(models_dir, file)
        nombre_modelo = file[:-len(".pkl")]

        # Cargar modelo
        with open(modelo_path, "rb") as f:
            modelo = pickle.load(f)

        # Predecir
        y_pred = modelo.predict(X_test)

        # Calcular métricas (usando weighted y macro)
        fila = {
            "Modelo": nombre_modelo,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 (weighted)": f1_score(y_test, y_pred, average="weighted"),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
            "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, y_pred, average="weighted")
        }
        resultados.append(fila)

        # Reporte + matriz de confusión
        print(f"===== {nombre_modelo} =====")
        fig_path = os.path.join(results_dir, f"matriz_confusion_{nombre_modelo}.png") if results_dir else None
        report = results_report_phenotype(y_test, y_pred, title=nombre_modelo,
                                          save_path=fig_path, show=show_plots)
        with open(os.path.join(out_dir, f"reporte_{nombre_modelo}.txt"), "w") as f:
            f.write(report)

    if not resultados:
        raise FileNotFoundError(f"No se encontraron modelos '*{sufijo}' en '{models_dir}'.")

    # Tabla de métricas
    df_resultados = pd.DataFrame(resultados)

    resultados_path = os.path.join(out_dir, f"scores_{data_label}.txt")
    with open(resultados_path, "w") as f:
        f.write(df_resultados.to_string(index=False))
    df_resultados.to_csv(os.path.join(out_dir, f"scores_{data_label}.csv"), index=False)

    print(f"Scores guardados en '{resultados_path}'.")

    return df_resultados
