"""
tools.py - Tools del simulador de riesgo de crédito de tarjetas (Taiwan)

Tools que el agente puede invocar:
  1. scoring_credito    → PD, score, nivel de riesgo, ECL, Stage IFRS 9 y límite recomendado
  2. recomendar_credito → KNN sobre titulares sanos con comportamiento similar
  3. guardar_solicitud  → persiste la consulta de forma seudonimizada, previo consentimiento

Modelo de COMPORTAMIENTO: evalúa el riesgo de mora del mes siguiente de un titular
que YA posee tarjeta. No evalúa solicitudes de crédito nuevas, para las que haría
falta un modelo de admisión: la acción que propone es de gestión sobre una tarjeta
en vigor (mantener o revisar el límite).

Nota sobre variables protegidas:
  La variable SEX se EXCLUYE por cumplimiento (Directiva 2004/113/CE).
  No se solicita ni se usa en el scoring.

Autor: Pablo Blas Hernández
"""

import sqlite3
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from langchain.tools import tool
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Fuente unica de verdad: columnas, LGD por defecto, umbrales y logica de stage.
from config_riesgo import (COLUMNAS_ENTRADA, crear_derivadas,
                           LGD_DEFECTO, AVISO_ECL, estado_mora, asignar_stage,
                           pd_a_horizonte, ead_con_ccf, calcular_ecl)


# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════════════════════
DB_PATH     = "riesgo_crediticio_taiwan.db"
MODELO_PATH = "tarjeta_taiwan.pkl"
CONFIG_PATH = "tarjeta_taiwan_config.pkl"

modelo = joblib.load(MODELO_PATH)
config = joblib.load(CONFIG_PATH)

PROB_CORTE  = config["Prob_Corte"]
LGD         = config.get("LGD", LGD_DEFECTO)

# Parametros IFRS 9 que estima el notebook
faltan_config = [k for k in ("CCF", "pd_estado_12m", "pd_estado_vida") if k not in config]
if faltan_config:
    raise RuntimeError(
        "tarjeta_taiwan_config.pkl no contiene los parametros IFRS 9 "
        f"{faltan_config}: es de una version anterior. Ejecuta el notebook y "
        "copia aqui el config regenerado.")
CCF             = config["CCF"]
PD_ESTADO_12M   = config["pd_estado_12m"]
PD_ESTADO_VIDA  = config["pd_estado_vida"]

# Edad minima legal para contratar credito: no se concede a menores por
# incapacidad de obrar, con independencia de la PD estimada por el modelo.
EDAD_MINIMA_LEGAL = 18


# ════════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# (columnas, derivadas y logica IFRS 9 se importan de config_riesgo)
# ════════════════════════════════════════════════════════════════════


def cargar_historico() -> pd.DataFrame:
    """Carga el histórico usando la vista v_scoring del esquema SQL.

    El JOIN entre cliente, tarjeta_credito e historial_pagos vive en el DDL
    (Anexo 4), no duplicado aquí: la vista es el único punto donde se define
    la tabla plana que consume el modelo.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT idTarjeta AS idCliente, LIMIT_BAL, AGE, EDUCATION, MARRIAGE,
               PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
               BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
               PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, y
        FROM v_scoring
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


# Nombres legibles de las variables del modelo. El agente NUNCA muestra al
# cliente el nombre tecnico: usa esta traduccion.
NOMBRES_LEGIBLES = {
    "PAY_1": "el estado de pago del ultimo mes",
    "PAY_2": "el estado de pago de hace dos meses",
    "PAY_3": "el estado de pago de hace tres meses",
    "PAY_4": "el estado de pago de hace cuatro meses",
    "PAY_5": "el estado de pago de hace cinco meses",
    "PAY_6": "el estado de pago de hace seis meses",
    "PAY_AMT1": "el importe pagado el ultimo mes",
    "PAY_AMT2": "el importe pagado hace dos meses",
    "PAY_AMT3": "el importe pagado hace tres meses",
    "PAY_AMT4": "el importe pagado hace cuatro meses",
    "PAY_AMT5": "el importe pagado hace cinco meses",
    "PAY_AMT6": "el importe pagado hace seis meses",
    "LIMIT_BAL": "el limite de credito de tu tarjeta",
    "AGE": "la edad",
    "EDUCATION": "el nivel de estudios",
    # Variables que el IV suele descartar: se nombran por si llegan a seleccionarse.
    "MARRIAGE": "el estado civil",
    "BILL_AMT1": "el importe de la ultima factura",
    "BILL_AMT2": "el importe de la factura de hace dos meses",
    "BILL_AMT3": "el importe de la factura de hace tres meses",
    "BILL_AMT4": "el importe de la factura de hace cuatro meses",
    "BILL_AMT5": "el importe de la factura de hace cinco meses",
    "BILL_AMT6": "el importe de la factura de hace seis meses",
    # Variables derivadas
    "util_limite": "la parte de tu limite que tienes dispuesta",
    "util_limite_media": "la parte de tu limite que has tenido dispuesta de media en el semestre",
    "esfuerzo_pago": "la parte de la ultima factura que has pagado",
    "esfuerzo_pago_medio": "la parte de tus facturas que has pagado en el semestre",
    "tendencia_retraso": "la evolucion de tus retrasos en el semestre",
    "meses_en_retraso": "el numero de meses con retraso en el semestre",
}


def factores_decision(df: pd.DataFrame, top_k: int = 3) -> list:
    """Devuelve los factores que MAS empujan la PD del cliente, calculados
    sobre el propio scorecard (no interpretados por el modelo de lenguaje).

    La scorecard es una regresion logistica sobre valores WOE, de modo que
        logit(PD) = intercepto + sum_j (coef_j * woe_j)
    y la contribucion de cada variable es exactamente coef_j * woe_j. Las
    contribuciones POSITIVAS son las que elevan la probabilidad de impago.

    Solo se devuelven variables que estan realmente en el modelo: las que el
    BinningProcess descarto por IV bajo no pueden aparecer aqui, porque no
    intervienen en el calculo de la PD.
    """
    try:
        bp = modelo.binning_process_
        woe = bp.transform(df, metric="woe")
        coefs = modelo.estimator_.coef_[0]
        if len(coefs) != woe.shape[1]:
            return []

        contribuciones = []
        for var, coef, valor_woe in zip(woe.columns, coefs, woe.iloc[0].values):
            aporte = float(coef) * float(valor_woe)
            contribuciones.append({
                "variable": var,
                "descripcion": NOMBRES_LEGIBLES.get(var, var),
                "aporte": round(aporte, 4),
                "sentido": "aumenta el riesgo" if aporte > 0 else "reduce el riesgo",
                "dato_facilitado": df.iloc[0].get(var) is not None
                                   and not pd.isna(df.iloc[0].get(var)),
            })

        # Orden descendente: primero lo que mas eleva la PD.
        contribuciones.sort(key=lambda c: c["aporte"], reverse=True)
        return [c for c in contribuciones if c["aporte"] > 0][:top_k]
    except Exception:
        # Si la version de optbinning no expone estos atributos, se prefiere
        # no dar ningun factor antes que dar uno inventado.
        return []


def construir_dataframe(datos: dict) -> pd.DataFrame:
    """Dataframe de una fila para el scorecard: datos del titular mas las
    variables derivadas, que se calculan aqui igual que en el entrenamiento."""
    fila = {col: datos.get(col, None) for col in COLUMNAS_ENTRADA}
    entrada = pd.DataFrame([fila], dtype=float)
    return crear_derivadas(entrada)[modelo.binning_process_.variable_names]


# ════════════════════════════════════════════════════════════════════
# TOOL 2 — SCORING
# ════════════════════════════════════════════════════════════════════
@tool
def scoring_credito(
    LIMIT_BAL: Optional[float] = None,
    PAY_1: Optional[int] = None, PAY_2: Optional[int] = None, PAY_3: Optional[int] = None,
    PAY_4: Optional[int] = None, PAY_5: Optional[int] = None, PAY_6: Optional[int] = None,
    BILL_AMT1: Optional[float] = None,
    AGE: Optional[int] = None, EDUCATION: Optional[int] = None, MARRIAGE: Optional[int] = None,
    BILL_AMT2: Optional[float] = None, BILL_AMT3: Optional[float] = None,
    BILL_AMT4: Optional[float] = None, BILL_AMT5: Optional[float] = None,
    BILL_AMT6: Optional[float] = None,
    PAY_AMT1: Optional[float] = None, PAY_AMT2: Optional[float] = None,
    PAY_AMT3: Optional[float] = None, PAY_AMT4: Optional[float] = None,
    PAY_AMT5: Optional[float] = None, PAY_AMT6: Optional[float] = None,
) -> dict:
    """Calcula PD, score, nivel de riesgo, ECL, Stage IFRS 9 y el limite recomendado
    para una tarjeta YA EN VIGOR (scoring de comportamiento).

    No evalua una solicitud de credito nueva: estima si el titular entrara en mora el
    mes siguiente y, a partir de ahi, si procede mantener o revisar su limite actual.

    Solo son imprescindibles DOS datos: el limite actual (LIMIT_BAL) y el estado de
    pago del ultimo mes (PAY_1). Todo lo demas es opcional: si el cliente no lo
    facilita, la tool lo infiere o lo trata como missing (WOE neutro), de modo que la
    evaluacion SIEMPRE se completa y el cliente recibe su resultado.

    Las facturas de los seis meses (BILL_AMT1..6) alimentan tres variables
    derivadas del modelo: la utilizacion del limite y el esfuerzo de pago. Si el
    titular no las facilita, esas variables quedan como ausentes.

    Devuelve limite_actual y limite_recomendado para construir el mensaje final.
    """
    # Solo son imprescindibles el limite actual de la tarjeta y el estado de pago
    # mas reciente. Si falta cualquier otro dato lo inferimos (abajo), de modo que la
    # evaluacion SIEMPRE se completa y el cliente recibe su decision. Antes se
    # exigian los seis meses y el saldo, y bastaba un "prefiero no contestar"
    # para que el asistente se quedase sin poder responder.
    faltan = []
    if LIMIT_BAL is None:
        faltan.append("el limite de credito de su tarjeta")
    if PAY_1 is None:
        faltan.append("el estado de pago del ultimo mes")
    if faltan:
        return {
            "error": "faltan_datos",
            "faltan": faltan,
            "mensaje": ("Aun no puedo evaluar la tarjeta: necesito "
                        + " y ".join(faltan) + ". Pideselo al cliente y vuelve "
                        "a llamar a scoring_credito."),
        }

    # Meses de pago no facilitados: se propaga el ultimo valor conocido, que es
    # el supuesto mas razonable sobre el comportamiento reciente del cliente.
    estados_pago = [PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6]
    ultimo_estado = PAY_1
    for i in range(len(estados_pago)):
        if estados_pago[i] is None:
            estados_pago[i] = ultimo_estado
        else:
            ultimo_estado = estados_pago[i]
    PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 = estados_pago

    # Sin saldo declarado, la utilizacion del limite queda como ausente en el
    # modelo y la exposicion se calcula sobre un saldo dispuesto nulo (solo
    # cuenta la parte del limite libre que podria llegar a disponer).
    saldo_ead = 0.0 if BILL_AMT1 is None else float(BILL_AMT1)

    datos = {
        "LIMIT_BAL": LIMIT_BAL, "AGE": AGE, "EDUCATION": EDUCATION, "MARRIAGE": MARRIAGE,
        "PAY_1": PAY_1, "PAY_2": PAY_2, "PAY_3": PAY_3,
        "PAY_4": PAY_4, "PAY_5": PAY_5, "PAY_6": PAY_6,
        "BILL_AMT1": BILL_AMT1, "BILL_AMT2": BILL_AMT2, "BILL_AMT3": BILL_AMT3,
        "BILL_AMT4": BILL_AMT4, "BILL_AMT5": BILL_AMT5, "BILL_AMT6": BILL_AMT6,
        "PAY_AMT1": PAY_AMT1, "PAY_AMT2": PAY_AMT2, "PAY_AMT3": PAY_AMT3,
        "PAY_AMT4": PAY_AMT4, "PAY_AMT5": PAY_AMT5, "PAY_AMT6": PAY_AMT6,
    }

    # Regla de negocio: se rechaza a los menores ANTES de puntuar. La denegacion
    # es por incapacidad legal para contratar, no por riesgo (no depende de la PD).
    if AGE is not None and AGE < EDAD_MINIMA_LEGAL:
        return {
            "decision":   "DENEGADO",
            "decision_n": 1,
            "motivo":     (f"Solicitante menor de {EDAD_MINIMA_LEGAL} anos: no se "
                           "puede conceder credito por incapacidad legal para contratar."),
            "datos":      datos,
        }

    df = construir_dataframe(datos)

    pd_valor   = float(modelo.predict_proba(df)[:, 1][0])
    score      = float(modelo.score(df)[0])
    decision   = "RIESGO" if pd_valor >= PROB_CORTE else "SANO"
    decision_n = 1 if pd_valor >= PROB_CORTE else 0
    # IFRS 9: stage por los criterios de retraso de la norma, PD a 12 meses y a
    # toda la vida desde la cadena de Markov, y EAD con factor de conversion.
    estado     = int(estado_mora(PAY_1))
    stage      = int(asignar_stage(PAY_1, pd_valor))
    pd_12m     = float(pd_a_horizonte(estado, PD_ESTADO_12M))
    pd_vida    = float(pd_a_horizonte(estado, PD_ESTADO_VIDA))
    ead        = float(ead_con_ccf(saldo_ead, LIMIT_BAL, CCF))
    ecl        = float(calcular_ecl(pd_12m, pd_vida, stage, LGD, ead))

    # Limite actualmente concedido al titular (variable LIMIT_BAL del modelo)
    limite_actual = float(LIMIT_BAL)

    if decision_n == 0:
        # SANO: el comportamiento de pago no justifica revisar el limite.
        limite_recomendado = limite_actual
        accion_recomendada = "mantener"
    else:
        # RIESGO: en scoring de comportamiento la accion de gestion no es denegar,
        # sino reducir la exposicion. El ajuste es proporcional a la PD estimada:
        # a mayor probabilidad de impago, mayor reduccion del limite.
        # Es un criterio de gestion propuesto para este TFM, no una regla
        # regulatoria: la politica real de revision de limites es del negocio.
        limite_recomendado = round(limite_actual * (1.0 - pd_valor), 2)
        accion_recomendada = "revisar a la baja"

    return {
        "PD":                 round(pd_valor, 4),
        "score":              round(score, 2),
        "decision":           decision,
        "decision_n":         decision_n,
        # Informacion de provision de la entidad: NO se comunica al titular.
        "PD_12m":             round(pd_12m, 4),
        "PD_vida":            round(pd_vida, 4),
        "EAD":                round(ead, 2),
        "ECL":                round(ecl, 2),
        "Stage":              stage,
        "aviso_ecl":          AVISO_ECL,
        "limite_actual":      round(limite_actual, 2),
        "limite_recomendado": round(limite_recomendado, 2),
        "accion_recomendada": accion_recomendada,
        # Unicos motivos que el asistente puede comunicar al cliente. Salen del
        # scorecard, no de la interpretacion del modelo de lenguaje.
        "factores_principales": factores_decision(df),
        # Variables del modelo que el cliente no facilito: entran como missing
        # con WOE neutro, de modo que la PD es una estimacion con informacion
        # parcial. Sirve para avisar al cliente de la limitacion.
        "datos_no_facilitados": [
            NOMBRES_LEGIBLES.get(v, v)
            for v in ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                      "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                      "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                      "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                      "AGE", "EDUCATION")
            if datos.get(v) is None
        ],
        # El saldo en bruto no entra en el modelo, pero si a traves de los
        # cocientes derivados (utilizacion del limite, esfuerzo de pago).
        "nota_bill_amt": ("El importe de la factura en bruto no interviene en la "
                          "probabilidad de impago; si lo hacen la utilizacion del "
                          "limite y el esfuerzo de pago, que se calculan con el. "
                          "Solo pueden citarse como motivo si aparecen en "
                          "factores_principales."),
        "datos":              datos,
    }


# ════════════════════════════════════════════════════════════════════
# TOOL 3 — RECOMENDADOR (perfiles de clientes sanos similares)
# ════════════════════════════════════════════════════════════════════
@tool
def recomendar_credito(LIMIT_BAL: Optional[float] = None, AGE: Optional[int] = None,
                       PAY_1: Optional[int] = None, PAY_2: Optional[int] = None,
                       PAY_3: Optional[int] = None,
                       BILL_AMT1: Optional[float] = None, PAY_AMT1: Optional[float] = None,
                       top_k: int = 3) -> dict:
    """Para un titular en RIESGO, busca los k titulares SANOS mas parecidos de la
    cartera historica (KNN) usando SUS PROPIOS datos como referencia (no hace falta
    ningun identificador). Sirve para orientarle sobre que patron de pago diferencia
    a los titulares de bajo riesgo con un perfil similar al suyo.
    """
    historico = cargar_historico()

    sanos = historico[historico["y"] == 0].copy()
    if sanos.empty:
        return {"error": "No hay clientes sanos en el historico."}

    cols_knn = ["LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "PAY_3",
                "BILL_AMT1", "PAY_AMT1"]

    # Referencia: los datos que el propio cliente ha aportado en la conversacion
    ref = {"LIMIT_BAL": LIMIT_BAL, "AGE": AGE, "PAY_1": PAY_1, "PAY_2": PAY_2,
           "PAY_3": PAY_3, "BILL_AMT1": BILL_AMT1, "PAY_AMT1": PAY_AMT1}
    ref_df = pd.DataFrame([{c: ref.get(c) for c in cols_knn}])

    # Imputar con la mediana de la cartera sana, no con 0: un 0 en LIMIT_BAL o en AGE
    # se convierte en un valor extremo al estandarizar y los "vecinos similares" dejan de serlo.
    medianas = sanos[cols_knn].median()
    scaler = StandardScaler()
    X_sanos = scaler.fit_transform(sanos[cols_knn].fillna(medianas))
    x_ref = scaler.transform(ref_df.fillna(medianas))

    k = min(top_k, len(sanos))
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_sanos)
    dist, idx = knn.kneighbors(x_ref)

    vecinos = sanos.iloc[idx[0]][cols_knn].copy()
    vecinos["distancia"] = dist[0].round(3)

    return {
        "vecinos_sanos": vecinos.to_dict(orient="records"),
        "mensaje": f"{k} clientes sanos con perfil similar encontrados."
    }


# ════════════════════════════════════════════════════════════════════
@tool
def guardar_solicitud(datos: dict, resultado_scoring: dict) -> dict:
    """Guarda la consulta de forma SEUDONIMIZADA en la base de datos, SOLO si el cliente
    ha dado su consentimiento explicito. El registro NO contiene nombre, DNI ni ningun
    dato identificativo: la base de datos asigna un ID autogenerado (idCliente) al
    insertar. Llama a esta tool UNICAMENTE cuando el cliente confirme que desea guardar
    su consulta; si no lo desea, no la llames.

    'datos' contiene las variables de la tarjeta (las de COLUMNAS_ENTRADA).
    'resultado_scoring' es la salida de scoring_credito; se recibe para trazabilidad, no se almacena
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Salvaguarda: no persistir menores (coherente con el CHECK edad>=18 del
    # esquema SQL). Evita un error de integridad y deja un mensaje claro.
    edad = datos.get("AGE")
    if edad is not None and edad < EDAD_MINIMA_LEGAL:
        conn.close()
        return {"ok": False,
                "motivo": (f"No se guarda: solicitante menor de {EDAD_MINIMA_LEGAL} "
                           "anos (incapacidad legal para contratar credito).")}

    try:
        # 1. Insertar cliente
        cur.execute(
            "INSERT INTO cliente (edad, educacion, estado_civil) VALUES (?, ?, ?)",
            (datos.get("AGE"), datos.get("EDUCATION"), datos.get("MARRIAGE"))
        )
        id_cliente = cur.lastrowid

        # 2. Insertar tarjeta. El desenlace (default_mes_sig) queda a NULL: 
        # la variable objetivo es la mora del mes siguiente, que aun no se ha observado. 
        # Guardar aqui la prediccion convertiria una salida del modelo en etiqueta de entrenamiento.
        cur.execute(
            "INSERT INTO tarjeta_credito (idCliente, limite_credito, default_mes_sig) "
            "VALUES (?, ?, ?)",
            (id_cliente, datos.get("LIMIT_BAL"), None)
        )
        id_tarjeta = cur.lastrowid

        # 3. Insertar historial de pagos
        cur.execute(
            """INSERT INTO historial_pagos
               (idTarjeta, estado_pago_1, estado_pago_2, estado_pago_3,
                estado_pago_4, estado_pago_5, estado_pago_6,
                factura_1, factura_2, factura_3, factura_4, factura_5, factura_6,
                pago_1, pago_2, pago_3, pago_4, pago_5, pago_6)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (id_tarjeta,
             datos.get("PAY_1"), datos.get("PAY_2"), datos.get("PAY_3"),
             datos.get("PAY_4"), datos.get("PAY_5"), datos.get("PAY_6"),
             datos.get("BILL_AMT1"), datos.get("BILL_AMT2"), datos.get("BILL_AMT3"),
             datos.get("BILL_AMT4"), datos.get("BILL_AMT5"), datos.get("BILL_AMT6"),
             datos.get("PAY_AMT1"), datos.get("PAY_AMT2"), datos.get("PAY_AMT3"),
             datos.get("PAY_AMT4"), datos.get("PAY_AMT5"), datos.get("PAY_AMT6"))
        )

        conn.commit()
        return {"guardado": True, "idCliente": id_cliente, "idTarjeta": id_tarjeta,
                "mensaje": (f"Consulta guardada con referencia #{id_cliente} "
                            f"(ID autogenerado, sin datos identificativos). El "
                            f"desenlace queda pendiente de observacion.")}
    except Exception as e:
        conn.rollback()
        return {"guardado": False, "error": str(e)}
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════════════
# LISTA DE TOOLS que importa chatbot.py
# ════════════════════════════════════════════════════════════════════
TOOLS = [scoring_credito, recomendar_credito, guardar_solicitud]
