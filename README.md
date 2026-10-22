# Scoring de Riesgo de Crédito de Comportamiento - Taiwán

Sistema de extremo a extremo para estimar la probabilidad de impago de titulares de tarjetas de crédito, calcular la pérdida esperada bajo IFRS 9, priorizar las acciones de recobro y desplegar el sistema mediante una API REST, dos agentes conversacionales y un cuadro de mando.

**Trabajo Fin de Máster:** Máster en Big Data, Data Science e Inteligencia Artificial  
**Universidad:** Universidad Complutense de Madrid - NTIC Máster  
**Curso académico:** 2025/2026  
**Autor:** Pablo Blas Hernández

---

## 1. Descripción del proyecto

El proyecto desarrolla un sistema de *behaviour scoring* para estimar el riesgo de impago de titulares de tarjetas de crédito. A partir del historial de pagos y de la información financiera del cliente, el modelo devuelve:

- **PD:** probabilidad de retraso en el pago del mes siguiente, calibrada.
- **Score:** puntuación interpretable mediante una escala PDO/odds.
- **Decisión:** clasificación de riesgo bajo o alto según un punto de corte sensible al coste.
- **PD a 12 meses y a toda la vida:** estimación del riesgo de *default* a 90 días mediante una cadena de Markov.
- **Stage IFRS 9:** clasificación en Stage 1, Stage 2 o Stage 3.
- **ECL:** pérdida crediticia esperada, calculada con la exposición ajustada mediante un factor de conversión de crédito (CCF).

El sistema incorpora un segundo modelo de **recobro**, dirigido a titulares que ya se encuentran en mora. Este estima la probabilidad de que no se recuperen al mes siguiente y asigna una estrategia de gestión: recordatorio, plan de pagos o gestión especializada.

El proyecto no es un modelo de admisión. Evalúa tarjetas ya concedidas y en vigor. Por tanto, las decisiones propuestas están relacionadas con el seguimiento del riesgo y la revisión de la exposición, no con conceder o denegar nuevas solicitudes de crédito.

## 2. Resultados del modelo

### 2.1. Scorecard principal (*champion*)

| Métrica | Train | Test |
|---|---:|---:|
| AUC-ROC | 0,7763 | 0,7594 (IC 95 %: 0,7433–0,7745) |
| Gini | 0,5527 | 0,5188 |
| KS | 41,49 % | 39,49 % |
| Brier score | — | 0,1390 |
| AUC en validación cruzada 5-fold | 0,7730 ± 0,0121 | — |
| PSI (train frente a test) | — | 0,0030 |

El modelo final utiliza **13 variables**, con los coeficientes en el sentido esperado. Incluye variables derivadas que resumen la persistencia de los retrasos, la utilización media del límite y el esfuerzo de pago.

La validación desplazada en el tiempo mantiene el AUC entre 0,860 y 0,875 y una calibración con una PD media de 0,151 frente a una tasa observada de 0,146.

### 2.2. Modelo challenger

Un modelo XGBoost sin restricciones, entrenado sobre las 28 variables candidatas, alcanza un AUC de 0,7858. La diferencia frente al scorecard es de **+0,026**, con un intervalo bootstrap emparejado de +0,020 a +0,033.

El scorecard se mantiene como modelo principal por su interpretabilidad y auditabilidad. El XGBoost presenta un Brier score de 0,177, frente a 0,139 en el scorecard, por lo que su mayor poder discriminatorio se acompaña de una calibración inferior.

### 2.3. Punto de corte

El punto de corte se obtiene mediante la regla de decisión de Bayes:

\[
PD > \frac{c}{LGD+c}
\]

donde \(c\) representa el coste de marcar como riesgo a un buen pagador.

Con el coste supuesto de un 2 % por gestión y un 8 % por margen perdido, el corte resultante es 0,182. Frente al corte neutro de 0,221, permite detectar 67 impagos adicionales a cambio de revisar a 262 buenos pagadores más, con un coste total prácticamente igual.

### 2.4. Pérdida esperada bajo IFRS 9

Los siguientes resultados corresponden a los 6.000 titulares del conjunto de test:

| Stage | Clientes | EAD (NT$) | ECL (NT$) | Cobertura |
|---|---:|---:|---:|---:|
| 1 | 4.598 | 443.294.664 | 13.509.725 | 3,05 % |
| 2 | 1.329 | 95.852.023 | 14.437.640 | 15,06 % |
| 3 | 73 | 5.063.984 | 2.278.793 | 45,00 % |
| **Total** | **6.000** | **544.210.671** | **30.226.158** | **5,55 %** |

**Importante:** estas cifras son ilustrativas y no deben utilizarse para realizar provisiones reales. La LGD es un valor convencional y el CCF se estima con una muestra pequeña. Las interfaces del proyecto advierten de estas limitaciones.

### 2.5. Modelo de recobro

El modelo de recobro utiliza un scorecard con la misma metodología general, entrenado con los titulares en mora de abril a junio y evaluado fuera de tiempo con los de julio. La validación cruzada se agrupa por titular.

Las alertas se generan sobre la mora de agosto, último mes con codificación consistente.

| Estrategia | Titulares | PD de no recuperación | Saldo en riesgo (NT$) |
|---|---:|---:|---:|
| Recordatorio | 1.604 | Menor de 0,664 | 40.678.810 |
| Plan de pagos | 682 | 0,664–0,793 | 11.849.547 |
| Gestión especializada | 1.669 | Mayor de 0,793 | 94.877.248 |

Los cortes entre estrategias se fijan mediante los terciles del entrenamiento. El saldo en riesgo se calcula como la PD de no recuperación multiplicada por el saldo.

El modelo distingue entre quienes probablemente se recuperen por sí solos y quienes requieren una gestión más intensiva. No anticipa de forma fiable el paso a 90 días de mora, para lo cual sería necesario entrenar un segundo modelo específico.

## 3. Equidad algorítmica

El modelo desplegado excluye la variable `SEX`. Su Information Value es 0,0088, inferior al umbral de 0,02 utilizado para la selección de variables.

| Configuración | AUC test | Disparate Impact | ΔTPR | ΔFPR |
|---|---:|---:|---:|---:|
| Scorecard sin sexo (desplegado) | 0,7594 | 0,983 | 0,002 | 0,003 |
| Scorecard con sexo (contrafactual) | 0,7605 | 0,924 | 0,039 | 0,038 |
| Rango en 5 particiones (desplegado) | 0,759–0,780 | 0,942–0,983 | 0,002–0,048 | 0,001–0,026 |

El modelo auxiliar que intenta reconstruir el sexo a partir de las variables del modelo obtiene un AUC de 0,563. Este resultado apunta a un riesgo indirecto leve, pero no nulo, que justifica mantener el seguimiento de las métricas de equidad.

## 4. Arquitectura

El sistema se organiza en los siguientes componentes:

```text
06_data/
    UCI_Credit_Card.csv
    config_riesgo.py
          |
          +--------------------------+
          |                          |
          v                          v
01_notebook/                     04_sql/
Modelo de comportamiento         Base SQLite normalizada
EDA, WOE, scorecard,              3 tablas + vista v_scoring
validación, XGBoost, SHAP                 |
MLflow e IFRS 9                           v
                                  02_chatbot/
08_recobro/                       Base de trabajo del agente
Modelo y alertas                  + cartera_scoring
alertas_recobro.xlsx              + alertas_recobro
          |                               |
          +-------------------------------+
                                          |
                +-------------------------+----------------------+
                |                         |                      |
                v                         v                      v
          Agente titular            Agente gestor            API REST
          02_chatbot/               02_chatbot/               05_api/
          Streamlit                 Streamlit                 FastAPI
                |                         |
                +-------------------------+
                                          |
                                          v
                                   07_tableau/
                                   Datos exportados
                                   y enlace publicado
```

### Base de datos

La base de datos principal se encuentra en `04_sql/riesgo_crediticio_taiwan.db`. Utiliza SQLite y organiza la información mediante tres tablas:

- `cliente`: edad, educación y estado civil.
- `tarjeta_credito`: límite de crédito y variable objetivo.
- `historial_pagos`: estados de pago, facturas y pagos de los seis meses observados.

La vista `v_scoring` reconstruye la tabla plana que consume el modelo. La base de datos que utiliza el agente del gestor incorpora además `cartera_scoring`, con los clientes puntuados. El proceso de precálculo puede cargar también la tabla `alertas_recobro` a partir del Excel de alertas.

### Configuración común

`config_riesgo.py` centraliza la lógica de variables derivadas, los criterios de clasificación IFRS 9, los costes del punto de corte y los cálculos de pérdida esperada.

Existen copias del módulo en `02_chatbot/`, `05_api/ms/` y `06_data/`. Si se modifica, las copias deben mantenerse sincronizadas para evitar diferencias entre componentes.

## 5. Decisiones de diseño

**Scorecard interpretable.** Se utiliza un modelo que permite explicar las puntuaciones mediante los tramos de cada variable, en lugar de depender exclusivamente de un modelo de caja negra.

**Sin restricción de monotonía.** Los códigos de estado de pago no están ordenados por riesgo de forma estrictamente creciente. Por ejemplo, pagar la totalidad, con código -1, no presenta necesariamente menos impago que pagar el mínimo, con código 0. Imponer monotonía podría distorsionar la tramificación.

**Selección de variables por criterio de negocio.** Se descartan variables redundantes por construcción. El modelo conserva `PAY_1` y las variables que resumen el comportamiento acumulado en el semestre. La pérdida de AUC asociada a esta selección es de 0,0009, frente a una dispersión entre particiones de 0,0121.

**Prevención de fugas de información.** La winsorización se ajusta exclusivamente con train; la búsqueda de hiperparámetros no utiliza test, y la validación cruzada repite el procedimiento completo en cada partición. El punto de corte se obtiene mediante una fórmula de coste y no mediante una búsqueda sobre test.

**IFRS 9 mediante estados de mora.** La PD a 12 meses y a toda la vida se estima mediante una cadena de Markov sobre tres estados de morosidad y se asigna de forma colectiva por estado. Se reconoce como limitación que este enfoque no recoge toda la persistencia individual del impago.

**Recobro por estrategias.** El objetivo operativo consiste en identificar a los titulares que probablemente se recuperen solos y evitar dedicarles una gestión costosa. Los mensajes que se muestran al titular se generan mediante reglas de código a partir de la estrategia asignada.

**El modelo de lenguaje no calcula el riesgo.** La PD, el score, el Stage, las explicaciones y las cantidades financieras se calculan en código mediante las herramientas de los agentes. El modelo de lenguaje conduce la conversación y comunica los resultados recibidos. Las consultas a la base de datos son parametrizadas y no se permite ejecutar SQL libre generado por el modelo de lenguaje.

## 6. Limitaciones

- El conjunto original no proporciona fechas individuales de observación. La validación temporal se aproxima desplazando la ventana dentro de los seis meses de historial, con un evento aproximado y los mismos titulares.
- La columna `PAY_1` se codifica de forma distinta a los meses anteriores, por lo que las estimaciones que recorren el historial utilizan abril a agosto.
- La LGD del 45 % es un valor convencional, no una estimación propia. La LGD real de tarjetas sin garantía puede ser superior.
- La cadena de Markov no incorpora memoria individual. La persistencia del impago observada por el scorecard sugiere que la PD de vida podría infraestimar el riesgo de los morosos crónicos.
- Los datos proceden de una entidad taiwanesa en 2005. Sus patrones no tienen por qué trasladarse directamente a otros mercados o periodos.
- Los resultados de ECL son ilustrativos y no están preparados para su utilización directa en procesos contables de provisión.

## 7. Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `01_notebook/` | Notebook de modelización y versión HTML: EDA, variables derivadas, WOE, scorecard, validación, XGBoost, SHAP, MLflow e IFRS 9. |
| `02_chatbot/` | Agentes del titular y del gestor, herramientas, estilos, configuración de riesgo y precálculo de cartera. |
| `03_fairness/` | Notebook de equidad: Disparate Impact, análisis contrafactual, estabilidad y modelo proxy. |
| `04_sql/` | Esquema SQL, base SQLite y script de carga de los 30.000 registros. |
| `05_api/` | Servicio FastAPI, configuración, funciones, modelos serializados y Dockerfile. |
| `06_data/` | Dataset original y configuración compartida del riesgo. |
| `07_tableau/` | Datos exportados y archivo `Visualizacion Tableau Public.txt`, con el enlace al dashboard publicado. El ZIP no incluye el libro editable `.twbx`. |
| `08_recobro/` | Notebook `recobro_clientes.ipynb`, resultados del modelo y dependencias. |

La distribución incluye modelos serializados y bases SQLite para facilitar las pruebas sin necesidad de volver a entrenar todos los modelos.

---

## 8. Guía de ejecución

### 8.1. Requisitos

- **Python 3.12**, recomendado para la ejecución local y para mantener la compatibilidad con el modelo serializado.
- Google Colab para reproducir los notebooks.
- Una clave de Groq para utilizar los agentes conversacionales, disponible en [console.groq.com](https://console.groq.com).
- Opcionalmente, Docker para contenerizar la API y Tableau Public Desktop para trabajar con un libro de Tableau editable.

`scikit-learn` está fijado en la versión 1.6.1 para mantener la compatibilidad con el modelo serializado. No se recomienda utilizar Python 3.14 para este proyecto.

**Si solo quieres probar los agentes con los artefactos incluidos en la distribución, puedes saltarte el entrenamiento en Colab y pasar directamente a la preparación de la base de datos y los agentes.** Los apartados de entrenamiento sirven para reproducir el proceso completo o regenerar los modelos.

### 8.2. Preparar el entorno local

Descomprime el proyecto y abre en VS Code la carpeta raíz `Anexos_TFM_PabloBlas_Scoring`, que contiene `README.md` y `requirements.txt`.

En una terminal de Windows PowerShell, desde esa carpeta, ejecuta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Este archivo de dependencias instala los paquetes del proyecto completo, incluida la dependencia `openpyxl` necesaria para leer el Excel de alertas.

Si solo vas a ejecutar los agentes conversacionales, puedes instalar las dependencias específicas de ese módulo en un entorno ya activado:

```powershell
python -m pip install -r .\02_chatbot\requirements.txt
```

Para macOS o Linux, crea y activa el entorno desde la raíz del proyecto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No es necesario incorporar la carpeta `.venv` al repositorio o al ZIP de entrega: puede volver a crearse siguiendo estos pasos.

### 8.3. Entrenar el scorecard principal en Colab

1. Abre `01_notebook/TFM_notebook_credit_taiwan.ipynb` en Google Colab.
2. Carga en la sesión de Colab los archivos `UCI_Credit_Card.csv` y `config_riesgo.py`, de modo que queden accesibles para el notebook.
3. Selecciona **Entorno de ejecución → Ejecutar todas**. La primera celda instala las dependencias y fija `scikit-learn` en la versión 1.6.1.
4. Descarga el archivo generado `artefactos_tfm_taiwan.zip`.

Este archivo contiene:

- `tarjeta_taiwan.pkl`: modelo de scorecard.
- `tarjeta_taiwan_config.pkl`: punto de corte y parámetros de riesgo utilizados por la aplicación.
- `datos_tableau_taiwan.xlsx`: datos exportados para el cuadro de mando.

### 8.4. Copiar los artefactos del scorecard

Si has vuelto a entrenar el modelo, copia los artefactos generados a estas ubicaciones:

| Artefacto | Destino |
|---|---|
| `tarjeta_taiwan.pkl` | `02_chatbot/` y `05_api/ms/model/` |
| `tarjeta_taiwan_config.pkl` | `02_chatbot/` y `05_api/ms/model/` |
| `datos_tableau_taiwan.xlsx` | `07_tableau/` |

Si el config no contiene los parámetros IFRS 9 que necesita la aplicación, vuelve a ejecutar el notebook y copia los artefactos más recientes.

### 8.5. Entrenar el modelo de recobro

1. Abre `08_recobro/recobro_clientes.ipynb` en Google Colab.
2. Carga `UCI_Credit_Card.csv` en la sesión.
3. Ejecuta el notebook completo y descarga `artefactos_recobro.zip`.

Copia los archivos resultantes en las carpetas correspondientes:

- `recobro_modelo.pkl` y `recobro_config.pkl` en `08_recobro/`.
- `alertas_recobro.xlsx` en `08_recobro/` y `02_chatbot/`.

El agente del gestor necesita el Excel para cargar las alertas de recobro.

### 8.6. Análisis de equidad (opcional)

Abre `03_fairness/TFM_fairness_analisis.ipynb` en Google Colab, carga `UCI_Credit_Card.csv` y el mismo `config_riesgo.py`, y ejecuta el notebook.

El análisis reproduce el preprocesado y el modelo desplegado para evaluar las métricas de equidad, el contrafactual con la variable sexo y el posible riesgo indirecto de discriminación.

### 8.7. Crear o regenerar la base de datos

La distribución ya incluye una base de datos en `04_sql/` y otra copia de trabajo en `02_chatbot/`. Si necesitas regenerar la base de datos normalizada a partir del CSV, activa el entorno virtual y ejecuta desde la raíz:

```powershell
Set-Location .\04_sql
python cargar_datos.py
```

El script crea `riesgo_crediticio_taiwan.db` con 30.000 registros en cada una de las tablas principales y la vista `v_scoring`.

Para copiar la base regenerada a la aplicación:

```powershell
Copy-Item .\riesgo_crediticio_taiwan.db ..\02_chatbot\riesgo_crediticio_taiwan.db
Set-Location ..
```

### 8.8. Puntuar la cartera y cargar las alertas

El agente del gestor consulta la cartera ya puntuada. Desde la raíz del proyecto, ejecuta:

```powershell
Set-Location .\02_chatbot
python precalcular_cartera.py
```

El script calcula la cartera y genera o actualiza la tabla `cartera_scoring`. Si `alertas_recobro.xlsx` está en la carpeta y las dependencias están instaladas, carga también las alertas en la tabla `alertas_recobro`.

Repite este paso cuando cambies el modelo de comportamiento, el modelo de recobro o los datos de entrada. Si acabas de regenerar la base SQLite, ejecuta el precálculo después de copiar la base nueva.

### 8.9. Arrancar los agentes conversacionales

Abre **dos terminales distintas** en VS Code. En ambas, activa el entorno virtual desde la raíz del proyecto y cambia a la carpeta del chatbot.

**Terminal 1 - Agente del titular**

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location .\02_chatbot
python -m streamlit run chatbot_titular.py
```

**Terminal 2 - Agente del gestor**

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location .\02_chatbot
python -m streamlit run agente_gestor.py
```

Cada comando inicia una aplicación independiente. Mantén abiertas ambas terminales mientras utilices los agentes.

Cuando la aplicación lo solicite, introduce la clave de Groq en la barra lateral. También puedes definirla como variable de entorno en PowerShell antes de arrancar:

```powershell
$env:GROQ_API_KEY = "TU_CLAVE_DE_GROQ"
```

El agente del titular permite realizar consultas de evaluación orientativa, mientras que el agente del gestor consulta la cartera puntuada y las alertas de recobro.

### 8.10. Arrancar la API REST

Desde la raíz del proyecto, activa el entorno y sitúate en `05_api/`:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location .\05_api
python -m uvicorn main:app --reload
```

La documentación interactiva estará disponible en:

`http://localhost:8000/docs`

Los ejemplos de llamada se encuentran en `05_api/test_api.txt`.

La API permite autenticación mediante una clave de entorno. Para activarla en PowerShell, antes de arrancar el servicio:

```powershell
$env:API_KEY = "una_clave"
python -m uvicorn main:app --reload
```

Cuando `API_KEY` está definida, los endpoints `/info` y `/predict` requieren la cabecera `X-API-Key`. El endpoint `/health` permanece abierto. Si no defines la clave, la API queda sin autenticación y debe utilizarse únicamente para pruebas locales.

Ejemplo de petición desde otra terminal PowerShell, con la API en ejecución:

```powershell
$body = @{
    LIMIT_BAL = 120000
    AGE = 34
    PAY_1 = 0
    PAY_2 = 0
    PAY_3 = 0
    PAY_4 = 0
    PAY_5 = 0
    PAY_6 = 0
    BILL_AMT1 = 3200
    BILL_AMT2 = 2800
    PAY_AMT1 = 3000
    EDUCATION = 2
    MARRIAGE = 1
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:8000/predict" `
    -Headers @{ "X-API-Key" = "una_clave" } `
    -ContentType "application/json" `
    -Body $body
```

### 8.11. Contenedor Docker (opcional)

Desde la raíz del proyecto, activa Docker y ejecuta:

```powershell
Set-Location .\05_api
docker build -t scoring-taiwan .
docker run -e API_KEY=una_clave -p 8000:8000 scoring-taiwan
```

La API quedará accesible en `http://localhost:8000`, y su documentación en `http://localhost:8000/docs`.

### 8.12. Cuadro de mando de Tableau

El directorio `07_tableau/` incluye el Excel exportado y el archivo `Visualizacion Tableau Public.txt`, que contiene el enlace a la versión publicada.

Puedes consultar el dashboard desde este enlace:

[**Abrir dashboard de Tableau Public**](https://public.tableau.com/views/RiskCreditTaiwan/Dashboardbanco?:language=es-ES&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

**Nota:** el ZIP de anexos no incluye el libro editable `Risk Credit Taiwan.twbx`. Por tanto, estas instrucciones permiten consultar la versión publicada, pero no editar localmente el libro a partir de esta distribución. Para actualizar las extracciones o republicar el dashboard es necesario disponer del libro editable.

---

## 9. Resolución de problemas

| Problema | Causa y solución |
|---|---|
| `optbinning` no se instala en local | Comprueba que estás usando Python 3.12 y que el entorno virtual está activado. |
| `InconsistentVersionWarning` al cargar el modelo | La versión de `scikit-learn` no coincide con la del modelo serializado. Utiliza `scikit-learn==1.6.1` en el entorno y evita mezclar entornos Python. |
| Error con `numpy._core` al cargar un `.pkl` | Puede existir una incompatibilidad de versión de NumPy. Instala las versiones requeridas por `requirements.txt` y vuelve a probar. |
| `ModuleNotFoundError: No module named 'openpyxl'` | Activa el entorno correspondiente e instala las dependencias con `python -m pip install -r requirements.txt` desde la raíz. |
| El config no contiene los parámetros IFRS 9 | El config o el modelo serializado pertenece a una versión anterior. Ejecuta el notebook y copia los artefactos nuevos. |
| El agente del gestor no encuentra `cartera_scoring` | Ejecuta `precalcular_cartera.py` desde `02_chatbot/`. |
| El agente no encuentra alertas de recobro | Copia `alertas_recobro.xlsx` en `02_chatbot/`, comprueba que `openpyxl` está instalado y vuelve a ejecutar el precálculo. |
| Resultados distintos entre componentes | Comprueba que las copias de `config_riesgo.py` son idénticas y que todos los componentes utilizan los artefactos de la misma versión. |
| La API no acepta una petición autenticada | Comprueba que `API_KEY` está definida en la terminal que arranca Uvicorn y que la petición envía la cabecera `X-API-Key` con la misma clave. |
| Tableau no encuentra los datos originales | El libro editable no está incluido en el ZIP. La versión publicada puede consultarse en el enlace indicado en el apartado 8.12. |

---

## 10. Tecnologías utilizadas

Python · pandas · scikit-learn · optbinning · XGBoost · SHAP · MLflow · FastAPI · Pydantic · Docker · Streamlit · LangChain · Groq · SQLite · Tableau.

## 11. Datos y licencia

El proyecto utiliza el dataset público *Default of Credit Card Clients*, publicado en UCI Machine Learning Repository.

Yeh, I. C. (2016). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Licencia CC BY 4.0.

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
