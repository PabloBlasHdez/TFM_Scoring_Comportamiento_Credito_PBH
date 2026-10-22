# ============================================================
# ms/functions.py - logica de prediccion del servicio (Taiwan)
# ============================================================

import pandas as pd

from ms import model, config
# Columnas, variables derivadas y logica IFRS 9: las mismas que el notebook y
# los agentes, desde el modulo compartido.
from ms.config_riesgo import (COLUMNAS_ENTRADA, crear_derivadas,
                              LGD_DEFECTO, AVISO_ECL, estado_mora, asignar_stage,
                              pd_a_horizonte, ead_con_ccf, calcular_ecl)

PROB_CORTE      = config["Prob_Corte"]
LGD             = config.get("LGD", LGD_DEFECTO)
CCF             = config["CCF"]
PD_ESTADO_12M   = config["pd_estado_12m"]
PD_ESTADO_VIDA  = config["pd_estado_vida"]


def construir_dataframe(input_dict: dict) -> pd.DataFrame:
    """Dataframe de una fila para el scorecard: datos de la peticion mas las
    variables derivadas."""
    fila = {col: input_dict.get(col, None) for col in COLUMNAS_ENTRADA}
    entrada = pd.DataFrame([fila], dtype=float)
    return crear_derivadas(entrada)[model.binning_process_.variable_names]


def predict(input_dict: dict) -> dict:
    """Aplica el modelo y devuelve PD, score, decision y perdida esperada."""
    df = construir_dataframe(input_dict)

    pd_valor   = float(model.predict_proba(df)[:, 1][0])
    score      = float(model.score(df)[0])
    decision_n = 1 if pd_valor >= PROB_CORTE else 0
    # En scoring de comportamiento la "decision" es de gestion del riesgo:
    #   1 -> titular en riesgo (revisar limite / seguimiento)
    #   0 -> titular sano
    decision   = "RIESGO" if decision_n == 1 else "SANO"

    # IFRS 9 con la logica compartida
    pay_1   = int(input_dict["PAY_1"])
    saldo   = float(input_dict["BILL_AMT1"])
    limite  = float(input_dict["LIMIT_BAL"])
    estado  = int(estado_mora(pay_1))
    stage   = int(asignar_stage(pay_1, pd_valor))
    pd_12m  = float(pd_a_horizonte(estado, PD_ESTADO_12M))
    pd_vida = float(pd_a_horizonte(estado, PD_ESTADO_VIDA))
    ead     = float(ead_con_ccf(saldo, limite, CCF))
    ecl     = float(calcular_ecl(pd_12m, pd_vida, stage, LGD, ead))

    return {
        "decision":     decision,
        "decision_n":   decision_n,
        "PD":           round(pd_valor, 4),
        "PD_12m":       round(pd_12m, 4),
        "PD_vida":      round(pd_vida, 4),
        "score":        round(score, 2),
        "EAD_sin_CCF":  round(max(saldo, 0.0), 2),
        "EAD":          round(ead, 2),
        "ECL":          round(ecl, 2),
        "Stage":        stage,
        "aviso":        AVISO_ECL,
    }


def get_model_response(input) -> dict:
    """Recibe el objeto Input de Pydantic y devuelve la respuesta del modelo."""
    if hasattr(input, "model_dump"):        # Pydantic v2
        input_dict = input.model_dump()
    elif hasattr(input, "dict"):            # Pydantic v1
        input_dict = input.dict()
    else:
        input_dict = dict(input)
    return predict(input_dict)
