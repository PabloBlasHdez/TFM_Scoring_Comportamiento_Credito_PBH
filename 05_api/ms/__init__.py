# ============================================================
# ms/__init__.py — inicializa la app FastAPI y carga el modelo (Taiwan)
# ============================================================

import os
import logging
import joblib
from fastapi import FastAPI

app = FastAPI(
    title="API de Scoring Crediticio (Taiwan)",
    description=(
        "Servicio REST que expone el scorecard de comportamiento del TFM. "
        "Devuelve la probabilidad de retraso en el mes siguiente, la puntuacion, "
        "la decision de gestion y la perdida esperada IFRS 9 por stage.\n\n"
        "**Las cifras de ECL son ilustrativas y no aptas para provisionar**: "
        "ver el campo `aviso` de cada respuesta."
    ),
    version="2.0.0",
)

log = logging.getLogger("scoring_api")

# Ruta absoluta de la carpeta 'ms', para encontrar siempre los ficheros
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path  = os.path.join(BASE_DIR, "model", "tarjeta_taiwan.pkl")
config_path = os.path.join(BASE_DIR, "model", "tarjeta_taiwan_config.pkl")

model  = joblib.load(model_path)
config = joblib.load(config_path)

# Si el config es de una version anterior no se arranca la API.
faltan = [k for k in ("CCF", "pd_estado_12m", "pd_estado_vida") if k not in config]
if faltan:
    raise RuntimeError(
        f"tarjeta_taiwan_config.pkl no contiene los parametros IFRS 9 {faltan}: "
        "es de una version anterior. Ejecuta el notebook y copia en ms/model/ "
        "el modelo y el config regenerados.")
