"""
cargar_datos.py — Puebla la base de datos SQLite desde el CSV de Taiwan

Crea las 3 tablas (cliente, tarjeta_credito, historial_pagos) y la vista
v_scoring, e inserta los 30.000 registros del dataset.

El ESQUEMA (DDL) NO se duplica aqui: se lee del fichero de referencia
`riesgo_crediticio_taiwan.sql`, que es la unica fuente de verdad del modelo
de datos. Asi el .sql y la base .db no pueden divergir.

Uso:
    python cargar_datos.py

Genera el fichero: riesgo_crediticio_taiwan.db
"""

import os
import sqlite3
import pandas as pd

CSV_PATH    = "../06_data/UCI_Credit_Card.csv"
DB_PATH     = "riesgo_crediticio_taiwan.db"
SCHEMA_PATH = "riesgo_crediticio_taiwan.sql"

# ── 1. Cargar y limpiar el CSV (igual que el notebook) ───────────────
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"default.payment.next.month": "y"})
df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
df["MARRIAGE"]  = df["MARRIAGE"].replace(0, 3)
df = df.rename(columns={"PAY_0": "PAY_1"})

# ── 2. Crear la base de datos ejecutando el esquema de referencia ────
# El DDL vive en riesgo_crediticio_taiwan.sql (unica fuente de verdad).
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema_sql = f.read()

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON;")
conn.executescript(schema_sql)
cur = conn.cursor()

# ── 3. Insertar los registros (por lotes, dentro de una transaccion) ─
clientes, tarjetas, historiales = [], [], []
for idx, row in enumerate(df.itertuples(index=False), start=1):
    r = row._asdict()
    clientes.append((idx, int(r["AGE"]), int(r["EDUCATION"]), int(r["MARRIAGE"])))
    tarjetas.append((idx, idx, float(r["LIMIT_BAL"]), int(r["y"])))
    historiales.append((
        idx,
        int(r["PAY_1"]), int(r["PAY_2"]), int(r["PAY_3"]),
        int(r["PAY_4"]), int(r["PAY_5"]), int(r["PAY_6"]),
        float(r["BILL_AMT1"]), float(r["BILL_AMT2"]), float(r["BILL_AMT3"]),
        float(r["BILL_AMT4"]), float(r["BILL_AMT5"]), float(r["BILL_AMT6"]),
        float(r["PAY_AMT1"]), float(r["PAY_AMT2"]), float(r["PAY_AMT3"]),
        float(r["PAY_AMT4"]), float(r["PAY_AMT5"]), float(r["PAY_AMT6"]),
    ))

cur.executemany(
    "INSERT INTO cliente (idCliente, edad, educacion, estado_civil) VALUES (?,?,?,?)",
    clientes)
cur.executemany(
    "INSERT INTO tarjeta_credito (idTarjeta, idCliente, limite_credito, default_mes_sig) "
    "VALUES (?,?,?,?)",
    tarjetas)
cur.executemany(
    """INSERT INTO historial_pagos
        (idTarjeta, estado_pago_1, estado_pago_2, estado_pago_3, estado_pago_4,
         estado_pago_5, estado_pago_6, factura_1, factura_2, factura_3, factura_4,
         factura_5, factura_6, pago_1, pago_2, pago_3, pago_4, pago_5, pago_6)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    historiales)

conn.commit()

# ── 4. Verificacion ──────────────────────────────────────────────────
n = cur.execute("SELECT COUNT(*) FROM v_scoring").fetchone()[0]
print(f"OK Base de datos creada: {DB_PATH}")
print(f"OK Registros insertados: {n}")
conn.close()
