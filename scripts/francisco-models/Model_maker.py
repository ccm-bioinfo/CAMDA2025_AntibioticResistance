"""
Pipeline completo de predicción de fenotipo para UN antibiótico y (opcionalmente) UN género:

    carga -> filtrado -> TF-IDF + SVD/NMF -> partición 80/20
          -> entrenamiento (5 clasificadores) -> evaluación (métricas + gráficas)

Uso:
    python -u Streptococcus26.py                                   # erythromycin + Streptococcus
    python -u Streptococcus26.py --antibiotico meropenem --genero Klebsiella
    python -u Streptococcus26.py --antibiotico meropenem --genero ninguno   # solo antibiótico

Salidas:
    data_models/                          tfidf_/svd_/nmf_<etiqueta>.pkl (se reutilizan si ya existen)
    modelos_fenotipo/                     <Modelo>_<etiqueta>_<lsa|nmf>.pkl
    resultados_fenotipo/<etiqueta>/       log, métricas, reportes, gráficas, parámetros óptimos
"""
import sys
import time
import logging
import argparse
import threading
import faulthandler
import traceback
import os

# Backend sin pantalla: las gráficas se guardan a archivo (necesario dentro de screen)
import matplotlib
matplotlib.use("Agg")

import pandas as pd

from new_camda_functions import (get_data2, sparse_fileter, make_label, transforms2, get_y_ordinal,
                                 split_stratified_into_train_val_test, train_and_save_binary_models,
                                 evaluate_models)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Entrena modelos de fenotipo para un antibiótico/género.")
parser.add_argument("--antibiotico", default="erythromycin")
parser.add_argument("--genero", default="Streptococcus", help="'ninguno' para filtrar solo por antibiótico")
parser.add_argument("--n_comp", type=int, default=100, help="componentes de SVD y NMF")
args = parser.parse_args()

ANTIBIOTICO = args.antibiotico
GENERO = None if args.genero.lower() in ("ninguno", "none", "") else args.genero
N_COMP = args.n_comp
LABEL = make_label(ANTIBIOTICO, GENERO)        # p. ej. 'erythromycin_Streptococcus'

RUTA_FACT = "data_models/"                                   # TF-IDF / SVD / NMF
RUTA_MODELOS = "modelos_fenotipo/"                           # clasificadores entrenados
RUTA_RESULTADOS = os.path.join("resultados_fenotipo", LABEL)  # métricas, gráficas, log
for d in (RUTA_FACT, RUTA_MODELOS, RUTA_RESULTADOS):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Monitoreo: log con hora, latido periódico y volcado de traceback si se cuelga
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(RUTA_RESULTADOS, "ejecucion.log")
HEARTBEAT_SEG = 60        # cada cuánto imprimir "sigo vivo"
TRACEBACK_SEG = 30 * 60   # cada cuánto volcar en qué línea va cada hilo

_log_fh = open(LOG_FILE, "a", buffering=1)


class Tee:
    """Duplica lo que se imprime (print, warnings, sklearn) a pantalla y al log."""
    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        _log_fh.write(s)

    def flush(self):
        self.stream.flush()
        _log_fh.flush()


sys.stdout = Tee(sys.__stdout__)
sys.stderr = Tee(sys.__stderr__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PID %(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fenotipo")

T0 = time.time()
paso_actual = "inicio"


def memoria_mb():
    try:
        with open("/proc/self/status") as f:
            for linea in f:
                if linea.startswith("VmRSS:"):
                    return int(linea.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def transcurrido():
    s = int(time.time() - T0)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def latido():
    while True:
        time.sleep(HEARTBEAT_SEG)
        log.info(f"[latido] vivo | paso: {paso_actual} | transcurrido {transcurrido()} | RAM {memoria_mb():.0f} MB")


class Paso:
    """Registra inicio, fin, duración y memoria de cada etapa."""
    def __init__(self, nombre):
        self.nombre = nombre

    def __enter__(self):
        global paso_actual
        paso_actual = self.nombre
        self.t = time.time()
        log.info(f">>> INICIA: {self.nombre} | RAM {memoria_mb():.0f} MB")

    def __exit__(self, exc_type, exc, tb):
        dur = time.time() - self.t
        if exc_type is None:
            log.info(f"<<< TERMINA: {self.nombre} en {dur:.1f} s | RAM {memoria_mb():.0f} MB")
        else:
            log.error(f"!!! FALLÓ: {self.nombre} tras {dur:.1f} s: {exc_type.__name__}: {exc}")
        return False


# Traceback de todos los hilos si hay un crash fatal (segfault, etc.) y
# periódicamente, para ver en qué línea está si parece colgado.
faulthandler.enable(file=_log_fh)
faulthandler.dump_traceback_later(TRACEBACK_SEG, repeat=True, file=_log_fh)

threading.Thread(target=latido, daemon=True).start()
log.info(f"Script iniciado: antibiótico={ANTIBIOTICO} | género={GENERO} | n_comp={N_COMP} | etiqueta={LABEL}")
log.info(f"Log en {os.path.abspath(LOG_FILE)}")

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
try:
    with Paso("get_data2 (carga de datos)"):
        metadata, counts = get_data2()
        log.info(f"    metadata: {metadata.shape} | counts: {tuple(counts['shape'])}")

    with Paso(f"sparse_fileter ({LABEL})"):
        metadata_f, counts_f = sparse_fileter(counts, metadata, antibiotic=ANTIBIOTICO, genus=GENERO)
        del counts  # ya no se necesita la matriz completa
        log.info(f"    metadata filtrada: {metadata_f.shape} | counts filtrados: {counts_f.shape}")

    # Si los modelos de representación ya existen, se cargan en lugar de reajustarlos
    exist_vect = os.path.exists(os.path.join(RUTA_FACT, f"tfidf_{LABEL}.pkl"))
    exist_fact = exist_vect and all(os.path.exists(os.path.join(RUTA_FACT, f"{m}_{LABEL}.pkl"))
                                    for m in ("svd", "nmf"))
    with Paso(f"transforms2 (TF-IDF + SVD + NMF, n_comp={N_COMP}; "
              f"tfidf {'cargado' if exist_vect else 'nuevo'}, svd/nmf {'cargados' if exist_fact else 'nuevos'})"):
        X_tfidf, tfidf_vect, amr_lsa, amr_nmf = \
            transforms2(amr_count=counts_f, data_label=LABEL, only_counts=False, exist_fact=exist_fact,
                        exist_vect=exist_vect, fact_path=RUTA_FACT, n_comp=N_COMP,
                        index=metadata_f.index)
        del X_tfidf, counts_f

    with Paso("get_y_ordinal"):
        Y = get_y_ordinal(metadata=metadata_f, ordinal=False)
        log.info(f"    Y: {Y.shape} | clases: {Y.value_counts().to_dict()}")
        # X y Y deben tener las mismas muestras en el mismo orden
        assert amr_lsa.index.equals(Y.index) and amr_nmf.index.equals(Y.index), "X y Y desalineados"

    todos = []
    for rep, X in (("lsa", amr_lsa), ("nmf", amr_nmf)):
        label_rep = f"{LABEL}_{rep}"

        with Paso(f"partición 80/20 ({rep})"):
            X_train, y_train, _, _, X_test, y_test, class_w, _ = \
                split_stratified_into_train_val_test(X, Y, .80, 0, .20, std=False,
                                                     two_subsets=True, random_state=42)
            log.info(f"    train: {X_train.shape} | test: {X_test.shape}")

        with Paso(f"entrenamiento ({rep})"):
            train_and_save_binary_models(X_train, y_train, label_rep, output_dir=RUTA_MODELOS,
                                         params_dir=RUTA_RESULTADOS)

        with Paso(f"evaluación ({rep})"):
            scores = evaluate_models(X_test, y_test, data_label=label_rep, models_dir=RUTA_MODELOS,
                                     results_dir=RUTA_RESULTADOS, show_plots=False)
            scores.insert(1, "Representación", rep)
            log.info(f"    métricas en test ({rep}):\n{scores.to_string(index=False)}")
            todos.append(scores)

    resumen = pd.concat(todos, ignore_index=True).sort_values("F1 (macro)", ascending=False)
    resumen.to_csv(os.path.join(RUTA_RESULTADOS, f"resumen_{LABEL}.csv"), index=False)
    log.info(f"Resumen (ordenado por F1 macro):\n{resumen.to_string(index=False)}")
    log.info(f"Resultados en {os.path.abspath(RUTA_RESULTADOS)}")

    log.info(f"=== SCRIPT COMPLETADO OK en {transcurrido()} ===")
except BaseException:
    log.error(f"=== SCRIPT ABORTADO en paso '{paso_actual}' tras {transcurrido()} ===\n{traceback.format_exc()}")
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
