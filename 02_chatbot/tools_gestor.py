"""
tools_gestor.py - Tools del asistente de cartera para el gestor de riesgos

Tools que el agente puede invocar:
  1. resumen_cartera     -> clientes, exposicion y provision, con filtros
  2. top_clientes        -> ranking por ECL o por PD
  3. detalle_cliente     -> ficha de un titular y factores que elevan su PD
  4. simular_pago        -> contrafactual: PD y ECL si el titular se pone al corriente

Diferencia con el simulador dirigido al titular (Anexo 2): aqui el gestor NO
introduce las variables. Los datos ya estan en la base, de modo que las tools
leen de cartera_scoring y el agente solo traduce la pregunta a una consulta.

Principio de diseno heredado del simulador: el modelo de lenguaje NO calcula.
Las tools ejecutan consultas PARAMETRIZADAS (no SQL generado por el LLM) y
devuelven cifras ya calculadas y formateadas; el LLM solo elige la tool y
redacta. Un LLM escribiendo SQL libre puede producir una consulta valida que
responde a otra pregunta, y ese numero parece verificable sin serlo.

Autor: Pablo Blas Hernandez
"""

import sqlite3
import unicodedata
from typing import Optional

import joblib
import pandas as pd
from langchain.tools import tool

from config_riesgo import (crear_derivadas, LGD_DEFECTO,
                           AVISO_ECL, estado_mora, asignar_stage,
                           pd_a_horizonte, ead_con_ccf, calcular_ecl)

DB_PATH     = "riesgo_crediticio_taiwan.db"
MODELO_PATH = "tarjeta_taiwan.pkl"
CONFIG_PATH = "tarjeta_taiwan_config.pkl"
TABLA       = "cartera_scoring"
TABLA_ALERTAS = "alertas_recobro"
MES_ALERTAS = "agosto, el ultimo mes con codificacion consistente"

modelo = joblib.load(MODELO_PATH)
config = joblib.load(CONFIG_PATH)

PROB_CORTE = config["Prob_Corte"]
# Criterio del punto de corte: "coste" o "neutro" (los config antiguos no lo traen)
CRITERIO_CORTE = config.get("criterio_corte", "neutro")
COSTE_FP       = config.get("coste_fp")
LGD        = config.get("LGD", LGD_DEFECTO)

faltan_config = [k for k in ("CCF", "pd_estado_12m", "pd_estado_vida") if k not in config]
if faltan_config:
    raise RuntimeError(
        "tarjeta_taiwan_config.pkl no contiene los parametros IFRS 9 "
        f"{faltan_config}: es de una version anterior. Ejecuta el notebook y "
        "copia aqui el config regenerado.")
CCF             = config["CCF"]
PD_ESTADO_12M   = config["pd_estado_12m"]
PD_ESTADO_VIDA  = config["pd_estado_vida"]

NOMBRES_LEGIBLES = {
    "PAY_1": "estado de pago del ultimo mes",
    "PAY_2": "estado de pago de hace dos meses",
    "PAY_3": "estado de pago de hace tres meses",
    "PAY_4": "estado de pago de hace cuatro meses",
    "PAY_5": "estado de pago de hace cinco meses",
    "PAY_6": "estado de pago de hace seis meses",
    "PAY_AMT1": "importe pagado el ultimo mes",
    "PAY_AMT2": "importe pagado hace dos meses",
    "PAY_AMT3": "importe pagado hace tres meses",
    "PAY_AMT4": "importe pagado hace cuatro meses",
    "PAY_AMT5": "importe pagado hace cinco meses",
    "PAY_AMT6": "importe pagado hace seis meses",
    "LIMIT_BAL": "limite de credito concedido",
    "AGE": "edad",
    "EDUCATION": "nivel de estudios",
    # Variables que el IV suele descartar: se nombran por si llegan a seleccionarse.
    "MARRIAGE": "estado civil",
    "BILL_AMT1": "importe de la ultima factura",
    "BILL_AMT2": "importe de la factura de hace dos meses",
    "BILL_AMT3": "importe de la factura de hace tres meses",
    "BILL_AMT4": "importe de la factura de hace cuatro meses",
    "BILL_AMT5": "importe de la factura de hace cinco meses",
    "BILL_AMT6": "importe de la factura de hace seis meses",
    # Variables derivadas
    "util_limite": "utilizacion del limite en el ultimo mes",
    "util_limite_media": "utilizacion media del limite en el semestre",
    "esfuerzo_pago": "parte pagada de la ultima factura",
    "esfuerzo_pago_medio": "parte pagada de las facturas del semestre",
    "tendencia_retraso": "evolucion del retraso en el semestre",
    "meses_en_retraso": "meses con retraso en el semestre",
}


# ════════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ════════════════════════════════════════════════════════════════════
def conectar() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def formatear_importe(x: float) -> str:
    """Importe con formato espanol y en NT$. Se devuelve ya formateado porque el
    LLM leia el punto decimal como separador de miles."""
    s = f"{x:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return f"{s} NT$"


def preparar_fila(crudo: pd.DataFrame) -> pd.DataFrame:
    """Anade las variables derivadas a una fila de v_scoring (la vista solo
    tiene las originales)."""
    return crear_derivadas(crudo)[modelo.binning_process_.variable_names]


def construir_filtros(stage, pd_min, pd_max, limite_min, solo_riesgo):
    """Construye el WHERE con parametros ligados. Nunca interpola valores."""
    cond, params = [], []
    if stage is not None:
        cond.append("Stage = ?");       params.append(int(stage))
    if pd_min is not None:
        cond.append("PD >= ?");         params.append(float(pd_min))
    if pd_max is not None:
        cond.append("PD <= ?");         params.append(float(pd_max))
    if limite_min is not None:
        cond.append("LIMIT_BAL >= ?");  params.append(float(limite_min))
    if solo_riesgo:
        cond.append("PD >= ?");         params.append(float(PROB_CORTE))
    where = (" WHERE " + " AND ".join(cond)) if cond else ""
    return where, params


def factores_decision(df_modelo: pd.DataFrame, top_k: int = 3) -> list:
    """Contribucion de cada variable a la PD, calculada sobre el scorecard.

    La scorecard es una regresion logistica sobre valores WOE, de modo que
        logit(PD) = intercepto + sum_j (coef_j * woe_j)
    y la contribucion de cada variable es exactamente coef_j * woe_j. Las
    positivas son las que elevan la probabilidad de impago.

    Si la version de optbinning no expone estos atributos se devuelve una lista
    vacia: mejor ningun factor que uno inventado.
    """
    try:
        bp = modelo.binning_process_
        woe = bp.transform(df_modelo, metric="woe")
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
            })
        contribuciones.sort(key=lambda c: c["aporte"], reverse=True)
        return [c for c in contribuciones if c["aporte"] > 0][:top_k]
    except Exception:
        return []


def hay_alertas(conn) -> bool:
    """Comprueba si el precalculo ha cargado las alertas de recobro."""
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (TABLA_ALERTAS,)).fetchone() is not None


def calcular_ifrs9(pd_valor: float, pay_1: int, saldo: float, limite: float) -> dict:
    """Stage, PD a horizonte, EAD y ECL de un titular, con la logica compartida."""
    estado = int(estado_mora(pay_1))
    stage  = int(asignar_stage(pay_1, pd_valor))
    pd_12m = float(pd_a_horizonte(estado, PD_ESTADO_12M))
    pd_vid = float(pd_a_horizonte(estado, PD_ESTADO_VIDA))
    ead    = float(ead_con_ccf(saldo, limite, CCF))
    ecl    = float(calcular_ecl(pd_12m, pd_vid, stage, LGD, ead))
    return {"stage": stage, "pd_12m": pd_12m, "pd_vida": pd_vid, "ead": ead, "ecl": ecl}


# ════════════════════════════════════════════════════════════════════
# TOOL 1 - RESUMEN DE CARTERA
# ════════════════════════════════════════════════════════════════════
@tool
def resumen_cartera(stage: Optional[int] = None,
                    pd_min: Optional[float] = None,
                    pd_max: Optional[float] = None,
                    limite_min: Optional[float] = None,
                    solo_riesgo: bool = False) -> dict:
    """Numero de clientes, exposicion (EAD) y provision (ECL) de un segmento.

    Responde a preguntas agregadas sobre la cartera: cuantos titulares superan
    el corte, cuanta provision concentra un stage, que peso tiene un segmento.

    Args:
        stage: filtra por stage IFRS 9 (1, 2 o 3). None = todos.
        pd_min: PD minima incluida.
        pd_max: PD maxima incluida.
        limite_min: solo titulares con limite concedido >= este valor.
        solo_riesgo: True para quedarse con los que superan el punto de corte.
    """
    where, params = construir_filtros(stage, pd_min, pd_max, limite_min, solo_riesgo)
    conn = conectar()
    q = f"""SELECT COUNT(*) AS clientes, SUM(EAD) AS ead, SUM(ECL) AS ecl,
                   AVG(PD) AS pd_media, AVG(PD_12m) AS pd12, AVG(PD_vida) AS pdv
            FROM {TABLA}{where}"""
    fila = pd.read_sql(q, conn, params=params).iloc[0]
    total = pd.read_sql(f"SELECT COUNT(*) AS n, SUM(ECL) AS ecl FROM {TABLA}",
                        conn).iloc[0]
    conn.close()

    if not fila["clientes"]:
        return {"clientes": 0, "mensaje": "Ningun titular cumple esos filtros."}

    ead, ecl = float(fila["ead"] or 0), float(fila["ecl"] or 0)
    return {
        "clientes": int(fila["clientes"]),
        "pct_cartera": round(100.0 * fila["clientes"] / total["n"], 2),
        "EAD_total": formatear_importe(ead),
        "ECL_total": formatear_importe(ecl),
        "cobertura_pct": round(100.0 * ecl / ead, 2) if ead else 0.0,
        "pct_ECL_cartera": round(100.0 * ecl / total["ecl"], 2),
        "PD_mes_siguiente_media": round(float(fila["pd_media"]), 4),
        "PD_12m_media": round(float(fila["pd12"]), 4),
        "PD_vida_media": round(float(fila["pdv"]), 4),
        "corte_aplicado": round(PROB_CORTE, 3),
        "aviso": AVISO_ECL,
    }


# ════════════════════════════════════════════════════════════════════
# TOOL 2 - RANKING DE CLIENTES
# ════════════════════════════════════════════════════════════════════
@tool
def top_clientes(n: int = 20,
                 ordenar_por: str = "ECL",
                 stage: Optional[int] = None,
                 limite_min: Optional[float] = None) -> dict:
    """Titulares mas relevantes de un segmento, ordenados por ECL o por PD.

    Ordenar por ECL y no por PD es lo util para priorizar recobro: un titular
    con PD 0,8 y saldo 2.000 cuesta menos que uno con PD 0,3 y saldo 50.000.

    Args:
        n: cuantos titulares devolver (maximo 50).
        ordenar_por: "ECL" (coste esperado) o "PD" (probabilidad de impago).
        stage: filtra por stage IFRS 9.
        limite_min: solo titulares con limite concedido >= este valor.
    """
    campo = "ECL" if str(ordenar_por).upper() == "ECL" else "PD"
    n = max(1, min(int(n), 50))

    where, params = construir_filtros(stage, None, None, limite_min, False)
    conn = conectar()
    q = f"""SELECT idTarjeta, PD, score, EAD, ECL, Stage, decision,
                   LIMIT_BAL, limite_recomendado, PAY_1
            FROM {TABLA}{where}
            ORDER BY {campo} DESC LIMIT ?"""
    df = pd.read_sql(q, conn, params=params + [n])
    conn.close()

    df["PD"] = df["PD"].round(4)
    df["score"] = df["score"].round(1)
    for c in ["EAD", "ECL", "LIMIT_BAL", "limite_recomendado"]:
        df[c] = df[c].apply(formatear_importe)

    return {
        "ordenado_por": campo,
        "devueltos": len(df),
        "clientes": df.to_dict(orient="records"),
        "aviso": AVISO_ECL,
    }


# ════════════════════════════════════════════════════════════════════
# TOOL 3 - FICHA DE UN TITULAR
# ════════════════════════════════════════════════════════════════════
@tool
def detalle_cliente(id_tarjeta: int) -> dict:
    """Ficha de riesgo de un titular concreto, con los factores que elevan su PD.

    Los factores NO los interpreta el modelo de lenguaje: se calculan en codigo
    como el producto del coeficiente por el valor WOE de cada tramo.

    Args:
        id_tarjeta: identificador del titular en la cartera.
    """
    conn = conectar()
    df = pd.read_sql(f"SELECT * FROM {TABLA} WHERE idTarjeta = ?",
                     conn, params=[int(id_tarjeta)])
    if df.empty:
        conn.close()
        return {"error": f"No existe el titular {id_tarjeta} en la cartera."}

    crudo = pd.read_sql("SELECT * FROM v_scoring WHERE idTarjeta = ?",
                        conn, params=[int(id_tarjeta)])
    recobro = None
    if hay_alertas(conn):
        a = pd.read_sql(f"SELECT estrategia, accion_gestor, PD_no_recupera FROM {TABLA_ALERTAS} "
                        "WHERE idTarjeta = ?", conn, params=[int(id_tarjeta)])
        if not a.empty:
            recobro = {"estrategia": a.iloc[0]["estrategia"],
                       "accion_gestor": a.iloc[0]["accion_gestor"],
                       "PD_no_recupera": round(float(a.iloc[0]["PD_no_recupera"]), 4),
                       "mes_referencia": MES_ALERTAS}
    conn.close()

    f = df.iloc[0]
    return {
        "idTarjeta": int(f["idTarjeta"]),
        "PD_mes_siguiente": round(float(f["PD"]), 4),
        "PD_12m": round(float(f["PD_12m"]), 4),
        "PD_vida": round(float(f["PD_vida"]), 4),
        "score": round(float(f["score"]), 2),
        "decision": f["decision"],
        "Stage": int(f["Stage"]),
        "EAD": formatear_importe(float(f["EAD"])),
        "ECL": formatear_importe(float(f["ECL"])),
        "limite_actual": formatear_importe(float(f["LIMIT_BAL"])),
        "limite_recomendado": formatear_importe(float(f["limite_recomendado"])),
        "estado_pago_ultimo_mes": int(f["PAY_1"]),
        "factores_que_elevan_la_PD": factores_decision(preparar_fila(crudo)),
        "recobro": recobro,
        "aviso": AVISO_ECL,
    }


# ════════════════════════════════════════════════════════════════════
# TOOL 4 - CONTRAFACTUAL
# ════════════════════════════════════════════════════════════════════
@tool
def simular_pago(id_tarjeta: int, nuevo_pay_1: int = 0) -> dict:
    """Recalcula PD, stage y ECL de un titular con otro estado de pago reciente.

    Al cambiar PAY_1 se recalculan tambien las derivadas que dependen de el,
    como los meses en retraso.

    Args:
        id_tarjeta: identificador del titular.
        nuevo_pay_1: estado de pago simulado (0 = al corriente, 1 = un mes de
            retraso, y asi sucesivamente; -1 y -2 son pago anticipado o sin consumo).
    """
    conn = conectar()
    crudo = pd.read_sql("SELECT * FROM v_scoring WHERE idTarjeta = ?",
                        conn, params=[int(id_tarjeta)])
    conn.close()
    if crudo.empty:
        return {"error": f"No existe el titular {id_tarjeta} en la cartera."}

    fila = crudo.iloc[0]
    saldo, limite = float(fila["BILL_AMT1"]), float(fila["LIMIT_BAL"])

    pd_actual = float(modelo.predict_proba(preparar_fila(crudo))[:, 1][0])
    simulado = crudo.copy()
    simulado["PAY_1"] = int(nuevo_pay_1)
    pd_sim = float(modelo.predict_proba(preparar_fila(simulado))[:, 1][0])

    act = calcular_ifrs9(pd_actual, int(fila["PAY_1"]), saldo, limite)
    sim = calcular_ifrs9(pd_sim, int(nuevo_pay_1), saldo, limite)
    return {
        "idTarjeta": int(id_tarjeta),
        "PAY_1_actual": int(fila["PAY_1"]),
        "PAY_1_simulado": int(nuevo_pay_1),
        "PD_actual": round(pd_actual, 4),
        "PD_simulada": round(pd_sim, 4),
        "variacion_PD": round(pd_sim - pd_actual, 4),
        "stage_actual": act["stage"],
        "stage_simulado": sim["stage"],
        "ECL_actual": formatear_importe(act["ecl"]),
        "ECL_simulado": formatear_importe(sim["ecl"]),
        "aviso": AVISO_ECL,
    }


# ════════════════════════════════════════════════════════════════════
# TOOL 5 - ALERTAS DE RECOBRO
# ════════════════════════════════════════════════════════════════════
ESTRATEGIAS = {"recordatorio": "Recordatorio", "plan de pagos": "Plan de pagos",
               "gestion especializada": "Gestion especializada"}


def normalizar(texto: str) -> str:
    """Minusculas y sin tildes, para aceptar 'Gestión especializada' o 'gestion especializada'."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.lower().strip()


@tool
def alertas_recobro(estrategia: Optional[str] = None, n: int = 15) -> dict:
    """Titulares en mora con su estrategia de recobro, ordenados por saldo en riesgo.

    Salen del modelo de recobro: para cada titular en mora, la probabilidad de que no
    se recupere el mes siguiente, la estrategia que le corresponde, la accion para el
    gestor y el mensaje para el titular. El saldo en riesgo es esa probabilidad por el
    saldo, y combina lo probable con lo que hay en juego.

    Args:
        estrategia: "Recordatorio", "Plan de pagos" o "Gestion especializada". None = todas.
        n: cuantos titulares devolver (maximo 50).
    """
    conn = conectar()
    if not hay_alertas(conn):
        conn.close()
        return {"error": "No hay alertas de recobro cargadas: copia alertas_recobro.xlsx a "
                         "02_chatbot y ejecuta precalcular_cartera.py."}

    filtro, params = "", []
    if estrategia:
        elegida = ESTRATEGIAS.get(normalizar(estrategia))
        if elegida is None:
            conn.close()
            return {"error": f"Estrategia desconocida: {estrategia}. "
                             f"Usa una de: {', '.join(ESTRATEGIAS.values())}."}
        filtro, params = " WHERE estrategia = ?", [elegida]

    resumen = pd.read_sql(f"SELECT estrategia, COUNT(*) AS titulares, "
                          f"SUM(saldo_en_riesgo) AS saldo_en_riesgo FROM {TABLA_ALERTAS} "
                          f"GROUP BY estrategia", conn)
    n = max(1, min(int(n), 50))
    df = pd.read_sql(f"SELECT prioridad, idTarjeta, PD_no_recupera, saldo, saldo_en_riesgo, "
                     f"estrategia, accion_gestor, mensaje_titular FROM {TABLA_ALERTAS}{filtro} "
                     f"ORDER BY saldo_en_riesgo DESC LIMIT ?", conn, params=params + [n])
    conn.close()

    df["PD_no_recupera"] = df["PD_no_recupera"].round(4)
    for c in ("saldo", "saldo_en_riesgo"):
        df[c] = df[c].apply(formatear_importe)
    resumen["saldo_en_riesgo"] = resumen["saldo_en_riesgo"].apply(formatear_importe)

    return {
        "mes_referencia": MES_ALERTAS,
        "resumen_por_estrategia": resumen.to_dict(orient="records"),
        "devueltos": len(df),
        "titulares": df.to_dict(orient="records"),
        "nota": "La estrategia, la accion y el mensaje los fija el modelo de recobro.",
    }


TOOLS = [resumen_cartera, top_clientes, detalle_cliente, simular_pago, alertas_recobro]
