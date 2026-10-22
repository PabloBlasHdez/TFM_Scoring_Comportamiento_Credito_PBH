"""
precalcular_cartera.py - Puntua la cartera completa y la deja en la base

Lee la vista v_scoring, calcula las variables derivadas, puntua a los 30.000
titulares y guarda en la tabla cartera_scoring la PD, el stage y el ECL.

Se precalcula porque el agente del gestor hace preguntas agregadas sobre toda la
cartera, y puntuarla en cada consulta seria demasiado lento. Equivale al proceso
nocturno que alimenta el cuadro de mando en una entidad.

Se ejecuta una vez, despues de cargar_datos.py y con el config nuevo del notebook:
    python precalcular_cartera.py

Autor: Pablo Blas Hernandez
"""

import os
import sqlite3
import joblib
import pandas as pd

from config_riesgo import (crear_derivadas, LGD_DEFECTO,
                           estado_mora, asignar_stage, pd_a_horizonte,
                           ead_con_ccf, calcular_ecl)

DB_PATH     = "riesgo_crediticio_taiwan.db"
MODELO_PATH = "tarjeta_taiwan.pkl"
CONFIG_PATH = "tarjeta_taiwan_config.pkl"
TABLA       = "cartera_scoring"
ALERTAS_PATH  = "alertas_recobro.xlsx"      # lo genera el notebook de recobro (08_recobro)
TABLA_ALERTAS = "alertas_recobro"


def main() -> None:
    modelo = joblib.load(MODELO_PATH)
    config = joblib.load(CONFIG_PATH)

    faltan = [k for k in ("CCF", "pd_estado_12m", "pd_estado_vida")
              if k not in config]
    if faltan:
        raise RuntimeError(f"El config no tiene los parametros IFRS 9 {faltan}: "
                           "ejecuta el notebook y copia aqui el config regenerado.")

    prob_corte = config["Prob_Corte"]
    lgd        = config.get("LGD", LGD_DEFECTO)

    conn = sqlite3.connect(DB_PATH)

    # La tabla plana la define la vista del DDL, no una consulta duplicada aqui.
    df = pd.read_sql("SELECT * FROM v_scoring", conn)
    print(f"Registros leidos de v_scoring: {len(df)}")

    # Mismas variables derivadas que en el entrenamiento, sobre valores brutos.
    X = crear_derivadas(df)[modelo.binning_process_.variable_names]

    df["PD"]    = modelo.predict_proba(X)[:, 1]
    df["score"] = modelo.score(X)

    # IFRS 9 con la logica compartida (identica a la de la API y los agentes).
    df["estado_mora"] = estado_mora(df["PAY_1"])
    df["Stage"]   = asignar_stage(df["PAY_1"].values, df["PD"].values)
    df["PD_12m"]  = pd_a_horizonte(df["estado_mora"].values, config["pd_estado_12m"])
    df["PD_vida"] = pd_a_horizonte(df["estado_mora"].values, config["pd_estado_vida"])
    df["EAD_sin_CCF"] = df["BILL_AMT1"].clip(lower=0.0)
    df["EAD"] = ead_con_ccf(df["BILL_AMT1"].values, df["LIMIT_BAL"].values, config["CCF"])
    df["ECL"] = calcular_ecl(df["PD_12m"].values, df["PD_vida"].values,
                             df["Stage"].values, lgd, df["EAD"].values)

    df["decision"] = (df["PD"] >= prob_corte).map({True: "RIESGO", False: "SANO"})

    # Limite recomendado: mismo criterio de gestion que la tool del simulador,
    # para que gestor y titular no vean cifras distintas del mismo caso.
    df["limite_recomendado"] = df.apply(
        lambda r: r["LIMIT_BAL"] if r["decision"] == "SANO"
        else round(r["LIMIT_BAL"] * (1.0 - r["PD"]), 2),
        axis=1,
    )

    salida = df[[
        "idTarjeta", "LIMIT_BAL", "AGE", "EDUCATION",
        "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "PAY_AMT1",
        "PD", "PD_12m", "PD_vida", "score", "estado_mora", "Stage",
        "EAD_sin_CCF", "EAD", "ECL", "decision",
        "limite_recomendado", "y",
    ]].copy()

    salida.to_sql(TABLA, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLA}_pd ON {TABLA}(PD)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLA}_stage ON {TABLA}(Stage)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLA}_ecl ON {TABLA}(ECL)")
    conn.commit()

    print(f"Tabla {TABLA} escrita: {len(salida)} filas")
    print(f"Corte aplicado: {prob_corte:.3f} | LGD: {lgd} | CCF: {config['CCF']:.3f}")
    resumen = salida.groupby("Stage").agg(
        clientes=("idTarjeta", "size"),
        EAD_total=("EAD", "sum"),
        ECL_total=("ECL", "sum"),
    )
    resumen["cobertura_pct"] = (100 * resumen["ECL_total"] / resumen["EAD_total"]).round(2)
    print(resumen)

    # Alertas de recobro: se cargan si el Excel del notebook de recobro esta en la carpeta
    if os.path.exists(ALERTAS_PATH):
        alertas = pd.read_excel(ALERTAS_PATH).rename(columns={"ID": "idTarjeta"})
        alertas.to_sql(TABLA_ALERTAS, conn, if_exists="replace", index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLA_ALERTAS}_estrategia "
                     f"ON {TABLA_ALERTAS}(estrategia)")
        conn.commit()
        print(f"\nTabla {TABLA_ALERTAS} escrita: {len(alertas)} alertas")
        print(alertas["estrategia"].value_counts().to_string())
    else:
        print(f"\nNo se encuentra {ALERTAS_PATH}: el agente funcionara sin alertas de recobro")

    conn.close()


if __name__ == "__main__":
    main()
