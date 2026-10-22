"""
config_riesgo.py - Parametros y funciones de riesgo compartidos

Lo importan el notebook, la API y los dos agentes, para que todos calculen
igual las variables derivadas, el stage y la perdida esperada.

Los parametros estimados (matriz de la cadena de Markov y CCF) los calcula el
notebook y se guardan en tarjeta_taiwan_config.pkl.

SEX no aparece porque se excluye del modelo (Directiva 2004/113/CE).

Autor: Pablo Blas Hernandez
"""

import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════════
# VARIABLES
# ════════════════════════════════════════════════════════════════════
MESES_PAY  = ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
MESES_BILL = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
              'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
MESES_PAGO = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
              'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Variables que se reciben desde fuera (formulario, API o base de datos)
COLUMNAS_ENTRADA = (['LIMIT_BAL', 'AGE', 'EDUCATION', 'MARRIAGE']
                    + MESES_PAY + MESES_BILL + MESES_PAGO)

VARIABLES_DERIVADAS = [
    'util_limite', 'util_limite_media',
    'esfuerzo_pago', 'esfuerzo_pago_medio',
    'tendencia_retraso', 'meses_en_retraso',
]

# Variables que recibe el scorecard
COLUMNAS_MODELO = COLUMNAS_ENTRADA + VARIABLES_DERIVADAS

# Candidatas que se descartan de entrada porque repiten la informacion de otra:
#   - PAY_2..PAY_6: el historial lo resumen PAY_1 (ultimo mes) y meses_en_retraso
#   - tendencia_retraso: es PAY_1 - PAY_6
#   - util_limite y esfuerzo_pago: las recogen sus medias del semestre
VARIABLES_REDUNDANTES = ['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                         'tendencia_retraso', 'util_limite', 'esfuerzo_pago']


def division_segura(numerador, denominador):
    """Cociente que deja en NaN los casos con denominador nulo o negativo."""
    return numerador / denominador.where(denominador > 0)


def crear_derivadas(datos):
    """Devuelve una copia del dataframe con las seis variables derivadas.

    Se calculan sobre valores brutos, igual al entrenar que al puntuar.
    """
    d = datos.copy()

    # Utilizacion del limite
    d['util_limite'] = division_segura(d['BILL_AMT1'], d['LIMIT_BAL'])
    d['util_limite_media'] = division_segura(d[MESES_BILL].mean(axis=1), d['LIMIT_BAL'])

    # Esfuerzo de pago (el pago de un mes liquida la factura del mes anterior)
    d['esfuerzo_pago'] = division_segura(d['PAY_AMT1'], d['BILL_AMT2'])
    d['esfuerzo_pago_medio'] = division_segura(d[MESES_PAGO[:-1]].sum(axis=1),
                                               d[MESES_BILL[1:]].sum(axis=1))

    # Dinamica del retraso
    d['tendencia_retraso'] = d['PAY_1'] - d['PAY_6']
    d['meses_en_retraso'] = (d[MESES_PAY] >= 1).sum(axis=1)

    return d


# ════════════════════════════════════════════════════════════════════
# PARAMETROS IFRS 9
# ════════════════════════════════════════════════════════════════════
# LGD convencional, ante la falta de datos de recuperacion. En tarjetas sin
# garantia la LGD real suele ser mayor, asi que el ECL tiende a quedarse corto.
LGD_DEFECTO = 0.45

HORIZONTE_12M = 12      # meses, ECL del Stage 1
VIDA_ESPERADA = 36      # meses, vida conductual de una tarjeta (NIIF 9, 5.5.20)

# Criterio cuantitativo de Stage 2. Sin la PD de concesion no se puede medir el
# aumento relativo del riesgo, asi que se usa un umbral absoluto.
UMBRAL_SICR_PD = 0.35

# Costes del falso positivo para el punto de corte (proporcion del saldo). Marcar como
# riesgo a un buen pagador cuesta la gestion del caso y el margen de intereses que se
# deja de ganar al recortarle el limite. El margen es un supuesto.
COSTE_GESTION_FP = 0.02
MARGEN_PERDIDO_FP = 0.08
COSTE_FP = COSTE_GESTION_FP + MARGEN_PERDIDO_FP

# Estados de morosidad (default a 90 dias, articulo 178 del CRR)
CORRIENTE, MORA, DEFAULT = 0, 1, 2

AVISO_ECL = (
    'ECL ilustrativo, no apto para provisionar. La PD de default sale de una '
    'cadena de Markov por tramo de retraso estimada con cinco meses de historial, '
    'la LGD del 45 % es un valor convencional y el CCF se estima con una muestra '
    'pequena. Antes de usarlo en real habria que validarlo de forma independiente.'
)


# ════════════════════════════════════════════════════════════════════
# FUNCIONES IFRS 9
# ════════════════════════════════════════════════════════════════════
def estado_mora(pay):
    """0 = al corriente, 1 = entre 30 y 89 dias, 2 = default (90 dias o mas)."""
    pay = np.asarray(pay)
    return np.where(pay >= 3, DEFAULT, np.where(pay >= 1, MORA, CORRIENTE))


def asignar_stage(pay_1, pd_valor):
    """Stage 3 con 90+ dias de retraso, Stage 2 con mas de 30 dias o PD alta,
    Stage 1 el resto."""
    pay = np.asarray(pay_1)
    prob = np.asarray(pd_valor, dtype=float)
    return np.where(pay >= 3, 3, np.where((pay >= 1) | (prob > UMBRAL_SICR_PD), 2, 1))


def pd_a_horizonte(estado, pd_base_estado):
    """PD de default a un horizonte segun el estado de morosidad.

    pd_base_estado es la PD acumulada de cada estado que sale de la cadena de
    Markov. Es una evaluacion colectiva por tramo de retraso.
    """
    est = np.asarray(estado)
    base = np.asarray(pd_base_estado, dtype=float)
    return np.where(est == DEFAULT, 1.0, base[np.minimum(est, MORA)])


def ead_con_ccf(saldo, limite, ccf):
    """EAD = saldo dispuesto + CCF * limite no dispuesto."""
    saldo = np.clip(np.asarray(saldo, dtype=float), 0, None)
    libre = np.clip(np.asarray(limite, dtype=float) - saldo, 0, None)
    return saldo + ccf * libre


def calcular_ecl(pd_12m, pd_vida, stage, lgd, ead):
    """ECL con la PD a 12 meses en Stage 1 y la de vida en Stages 2 y 3."""
    pd_aplicada = np.where(np.asarray(stage) == 1, pd_12m, pd_vida)
    return pd_aplicada * lgd * np.asarray(ead, dtype=float)
