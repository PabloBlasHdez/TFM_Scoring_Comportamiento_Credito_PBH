# ============================================================
# main.py - API REST de scoring crediticio con FastAPI (Taiwan)
# Modelo: scorecard de comportamiento (retraso de la tarjeta el mes siguiente)
# Dataset: Default of Credit Card Clients (UCI, CC BY 4.0)
#
# Arranque local:
#     uvicorn main:app --reload
# Con autenticacion:
#     export API_KEY="una_clave"        (Windows CMD: set API_KEY=una_clave)
#     uvicorn main:app
# Documentacion interactiva:
#     http://localhost:8000/docs   (boton "Authorize" para introducir la clave)
# ============================================================

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from ms import app, config, log
from ms.functions import get_model_response, CCF, LGD, PROB_CORTE
from ms.config_riesgo import AVISO_ECL, UMBRAL_SICR_PD, VIDA_ESPERADA, HORIZONTE_12M


model_name = "Scorecard Credit Card Default (Taiwan)"
version = "v2.0.0"

# Edad minima legal para contratar credito. No se concede a menores por
# incapacidad de obrar, con independencia de su riesgo estimado (PD). El modelo,
# ademas, se entreno con titulares de 21+ anos, por lo que puntuar a un menor
# quedaria fuera de dominio. Se valida en el endpoint y no en el esquema Pydantic
# para devolver el motivo real en lugar de un error generico de validacion.
EDAD_MINIMA_LEGAL = 18


# ════════════════════════════════════════════════════════════
# Autenticacion por clave en cabecera
# ════════════════════════════════════════════════════════════
# Si la variable de entorno API_KEY esta definida, /info y /predict exigen la
# cabecera X-API-Key. Si no lo esta, la API queda abierta (solo para pruebas en local).
API_KEY = os.environ.get("API_KEY")
cabecera_clave = APIKeyHeader(name="X-API-Key", auto_error=False)

if API_KEY:
    log.warning("Autenticacion ACTIVADA: se exige la cabecera X-API-Key.")
else:
    log.warning("Autenticacion DESACTIVADA: define API_KEY para exigir clave. "
                "No expongas el servicio fuera de local en este modo.")


def verificar_clave(clave: Optional[str] = Security(cabecera_clave)) -> None:
    """Comprueba la clave de la cabecera (compare_digest, como en la documentacion de FastAPI)."""
    if not API_KEY:
        return
    if not clave or not secrets.compare_digest(clave, API_KEY):
        raise HTTPException(status_code=401, detail="Clave de API ausente o no valida.",
                            headers={"WWW-Authenticate": "API-Key"})


# ════════════════════════════════════════════════════════════
# Esquema de entrada — validacion con Pydantic
# NOTA: SEX se EXCLUYE del modelo por cumplimiento (Directiva 2004/113/CE),
# por eso no aparece como variable de entrada.
# Las variables PAY_n son ordinales (-2 a 8): estado de pago de cada mes.
# ════════════════════════════════════════════════════════════
class Input(BaseModel):
    LIMIT_BAL: float = Field(..., gt=0, description="Limite de credito en NT$")
    AGE: int = Field(..., ge=0, le=120, description="Edad del titular en anos")

    # Historial de estado de pago (los mas predictivos)
    PAY_1: int = Field(..., ge=-2, le=9, description="Estado de pago mes -1 (sep). Fija tambien el stage IFRS 9")
    PAY_2: int = Field(..., ge=-2, le=9, description="Estado de pago mes -2 (ago)")
    PAY_3: int = Field(..., ge=-2, le=9, description="Estado de pago mes -3 (jul)")
    PAY_4: int = Field(..., ge=-2, le=9, description="Estado de pago mes -4 (jun)")
    PAY_5: int = Field(..., ge=-2, le=9, description="Estado de pago mes -5 (may)")
    PAY_6: int = Field(..., ge=-2, le=9, description="Estado de pago mes -6 (abr)")

    # Facturas: alimentan las variables derivadas de utilizacion del limite y
    # esfuerzo de pago. Si faltan, esas variables quedan como ausentes.
    BILL_AMT1: float = Field(..., description="Saldo de la ultima factura: entra en las derivadas y fija la exposicion")
    BILL_AMT2: Optional[float] = Field(None, description="Factura mes -2: denominador del esfuerzo de pago")
    BILL_AMT3: Optional[float] = Field(None, description="Factura mes -3")
    BILL_AMT4: Optional[float] = Field(None, description="Factura mes -4")
    BILL_AMT5: Optional[float] = Field(None, description="Factura mes -5")
    BILL_AMT6: Optional[float] = Field(None, description="Factura mes -6")

    # Importes pagados en los ultimos 6 meses
    PAY_AMT1: Optional[float] = None
    PAY_AMT2: Optional[float] = None
    PAY_AMT3: Optional[float] = None
    PAY_AMT4: Optional[float] = None
    PAY_AMT5: Optional[float] = None
    PAY_AMT6: Optional[float] = None

    # Demograficas no protegidas
    EDUCATION: Optional[int] = Field(None, ge=1, le=4, description="1=posgrado,2=univ,3=secundaria,4=otros")
    MARRIAGE: Optional[int] = Field(None, ge=1, le=3, description="1=casado,2=soltero,3=otros")

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 120000,
                "AGE": 34,
                "PAY_1": 0, "PAY_2": 0, "PAY_3": 0,
                "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 3200, "BILL_AMT2": 2800, "BILL_AMT3": 2500,
                "BILL_AMT4": 2100, "BILL_AMT5": 1800, "BILL_AMT6": 1500,
                "PAY_AMT1": 3000, "PAY_AMT2": 2800, "PAY_AMT3": 2400,
                "PAY_AMT4": 2000, "PAY_AMT5": 1800, "PAY_AMT6": 1500,
                "EDUCATION": 2, "MARRIAGE": 1
            }
        }
    }


# ════════════════════════════════════════════════════════════
# Esquema de salida del servicio
# ════════════════════════════════════════════════════════════
class Output(BaseModel):
    decision: str = Field(description="SANO o RIESGO, segun el punto de corte desplegado")
    decision_n: int
    PD: float = Field(description="Probabilidad de retraso en el mes siguiente (scorecard)")
    PD_12m: float = Field(description="Probabilidad de default (90+ dias) a 12 meses")
    PD_vida: float = Field(description="Probabilidad de default (90+ dias) a toda la vida esperada")
    score: float
    EAD_sin_CCF: float = Field(description="Saldo dispuesto")
    EAD: float = Field(description="Exposicion con factor de conversion sobre el limite libre")
    ECL: float = Field(description="Perdida esperada: 12 meses en Stage 1, vida completa en 2 y 3")
    Stage: int = Field(description="Stage IFRS 9 por criterios de retraso")
    aviso: str = Field(description="Alcance y supuestos de las cifras de ECL")


# ════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════
@app.get("/health")
async def service_health():
    """Estado del servicio. Abierto siempre, para comprobaciones de disponibilidad."""
    return {"status": "ok"}


@app.get("/info", dependencies=[Security(verificar_clave)])
async def model_info():
    """Informacion del modelo, de los parametros IFRS 9 y de la autenticacion."""
    return {
        "name": model_name,
        "version": version,
        "autenticacion": "activada" if API_KEY else "DESACTIVADA (solo desarrollo)",
        "punto_corte": round(PROB_CORTE, 4),
        "criterio_corte": config.get("criterio_corte", "neutro"),
        "coste_falso_positivo": config.get("coste_fp"),
        "parametros_ifrs9": {
            "LGD": LGD,
            "CCF": round(CCF, 4),
            "criterio_stage_3": "90 o mas dias de retraso",
            "criterio_stage_2": f"mas de 30 dias de retraso o PD > {UMBRAL_SICR_PD}",
            "horizonte_stage_1_meses": HORIZONTE_12M,
            "vida_esperada_meses": VIDA_ESPERADA,
        },
        "aviso": AVISO_ECL,
    }


@app.post("/predict", response_model=Output, dependencies=[Security(verificar_clave)])
async def model_predict(input: Input):
    """Calcula PD, score, decision, ECL y Stage IFRS 9 para una tarjeta.

    Regla de negocio: se rechaza toda solicitud de un menor de edad ANTES de
    puntuarla. La denegacion es por incapacidad legal para contratar, no por
    riesgo crediticio, de modo que no depende de la PD del modelo.
    """
    if input.AGE < EDAD_MINIMA_LEGAL:
        raise HTTPException(
            status_code=422,
            detail=(f"Solicitante menor de {EDAD_MINIMA_LEGAL} anos: no se puede "
                    "conceder credito por incapacidad legal para contratar."),
        )
    return get_model_response(input)
