/* -------------------------------------------------------------------------------------------
Autor: Pablo Blas Hernandez
Nombre de la base de datos: riesgo_crediticio_taiwan  (fichero: riesgo_crediticio_taiwan.db)
Motor: SQLite 3

Descripcion: Base de datos basada en el Default of Credit Card Clients Dataset (UCI, CC BY 4.0)
             30.000 titulares de tarjeta de credito de Taiwan con su comportamiento de pago
             y el resultado (default el mes siguiente: 0/1)

             Scoring de COMPORTAMIENTO: predice si un cliente que YA tiene tarjeta
             entrara en mora el mes siguiente.

             Diseno relacional en 3 tablas:
               - cliente            : datos demograficos (SIN sexo, excluido por fairness)
               - tarjeta_credito    : limite y resultado
               - historial_pagos    : estado de pago, facturas y pagos de los ultimos 6 meses

NOTA DE IMPLEMENTACION:
  Este script es el esquema OPERATIVO real del proyecto y se ejecuta sobre SQLite,
  el mismo motor que utiliza cargar_datos.py para poblar la base y que consumen la
  API (05_api) y el chatbot (02_chatbot). Se elige SQLite por portabilidad: la base
  viaja como un unico fichero .db sin necesidad de servidor, lo que simplifica la
  reproducibilidad del TFM. El DDL es identico al que ejecuta cargar_datos.py.

  Para ejecutarlo de forma independiente:
      sqlite3 riesgo_crediticio_taiwan.db < riesgo_crediticio_taiwan.sql
--------------------------------------------------------------------------------------------*/

-- SQLite exige activar explicitamente la comprobacion de claves foraneas en cada conexion.
PRAGMA foreign_keys = ON;

/* -------------------------------------------------------------------------------------------
Definicion de la estructura de la base de datos
--------------------------------------------------------------------------------------------*/
DROP VIEW  IF EXISTS v_scoring;
DROP TABLE IF EXISTS historial_pagos;
DROP TABLE IF EXISTS tarjeta_credito;
DROP TABLE IF EXISTS cliente;

-- Tabla CLIENTE: datos demograficos del titular
-- NOTA: la variable SEX del dataset original se OMITE deliberadamente
--       por cumplimiento de la Directiva 2004/113/CE (no discriminacion por sexo).
CREATE TABLE cliente (
  idCliente     INTEGER PRIMARY KEY AUTOINCREMENT,
  edad          INTEGER CHECK (edad >= 18),  -- mayoria de edad legal: no se concede credito a menores
  educacion     INTEGER CHECK (educacion BETWEEN 1 AND 4),  -- 1=posgrado,2=univ,3=secundaria,4=otros
  estado_civil  INTEGER CHECK (estado_civil BETWEEN 1 AND 3) -- 1=casado,2=soltero,3=otros
);

-- Tabla TARJETA_CREDITO: limite concedido y resultado observado
CREATE TABLE tarjeta_credito (
  idTarjeta       INTEGER PRIMARY KEY AUTOINCREMENT,
  idCliente       INTEGER NOT NULL,
  limite_credito  REAL NOT NULL CHECK (limite_credito > 0),   -- LIMIT_BAL en NT$
  default_mes_sig INTEGER CHECK (default_mes_sig IN (0,1)), -- variable objetivo; NULL = desenlace aun no observado
  FOREIGN KEY (idCliente) REFERENCES cliente(idCliente)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);

-- Tabla HISTORIAL_PAGOS: comportamiento de los ultimos 6 meses (abril-septiembre 2005)
-- estado_pago_n: -2=sin consumo, -1=al dia, 0=pago minimo, 1..9=meses de retraso
CREATE TABLE historial_pagos (
  idTarjeta     INTEGER PRIMARY KEY,
  -- Estado de pago mensual (estado_pago_1 = PAY_1 = mes mas reciente)
  estado_pago_1 INTEGER NOT NULL,
  estado_pago_2 INTEGER,
  estado_pago_3 INTEGER,
  estado_pago_4 INTEGER,
  estado_pago_5 INTEGER,
  estado_pago_6 INTEGER,
  -- Importe de la factura de cada mes (BILL_AMT)
  factura_1     REAL,
  factura_2     REAL,
  factura_3     REAL,
  factura_4     REAL,
  factura_5     REAL,
  factura_6     REAL,
  -- Importe pagado cada mes (PAY_AMT)
  pago_1        REAL,
  pago_2        REAL,
  pago_3        REAL,
  pago_4        REAL,
  pago_5        REAL,
  pago_6        REAL,
  FOREIGN KEY (idTarjeta) REFERENCES tarjeta_credito(idTarjeta)
    ON DELETE CASCADE
    ON UPDATE CASCADE
);

/* -------------------------------------------------------------------------------------------
Vista para el scoring: reune las 3 tablas en la estructura que espera el modelo
--------------------------------------------------------------------------------------------*/
CREATE VIEW v_scoring AS
SELECT
    t.idTarjeta,
    t.limite_credito         AS LIMIT_BAL,
    c.edad                   AS AGE,
    c.educacion              AS EDUCATION,
    c.estado_civil           AS MARRIAGE,
    h.estado_pago_1          AS PAY_1,
    h.estado_pago_2          AS PAY_2,
    h.estado_pago_3          AS PAY_3,
    h.estado_pago_4          AS PAY_4,
    h.estado_pago_5          AS PAY_5,
    h.estado_pago_6          AS PAY_6,
    h.factura_1              AS BILL_AMT1,
    h.factura_2              AS BILL_AMT2,
    h.factura_3              AS BILL_AMT3,
    h.factura_4              AS BILL_AMT4,
    h.factura_5              AS BILL_AMT5,
    h.factura_6              AS BILL_AMT6,
    h.pago_1                 AS PAY_AMT1,
    h.pago_2                 AS PAY_AMT2,
    h.pago_3                 AS PAY_AMT3,
    h.pago_4                 AS PAY_AMT4,
    h.pago_5                 AS PAY_AMT5,
    h.pago_6                 AS PAY_AMT6,
    t.default_mes_sig        AS y
FROM cliente c
JOIN tarjeta_credito t ON c.idCliente = t.idCliente
JOIN historial_pagos  h ON t.idTarjeta = h.idTarjeta;
