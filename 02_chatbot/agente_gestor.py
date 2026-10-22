"""
agente_gestor.py - Asistente de cartera para el area de riesgos

Segundo canal conversacional sobre la misma scorecard. A diferencia del
simulador dirigido al titular (Anexo 2), el destinatario es el gestor: no
introduce variables, pregunta sobre una cartera cuyos datos ya estan en la base.

Lo que aporta frente al cuadro de mando: el dashboard responde a las preguntas
que su autor anticipo (distribucion por stage, consulta por identificador). El
agente cubre la cola larga de preguntas que no merecen una vista propia.

Arquitectura: identica en su principio al simulador. El modelo de lenguaje
elige la tool y redacta; la PD, el score, el stage y el ECL se resuelven en las
tools sobre el scorecard serializado. El LLM no calcula ni escribe SQL.

Ejecutar:
    python -m streamlit run agente_gestor.py

Autor: Pablo Blas Hernandez
"""

import os
import sqlite3
from pathlib import Path

import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from tools_gestor import TOOLS, PROB_CORTE, LGD, CCF, CRITERIO_CORTE, COSTE_FP
from config_riesgo import AVISO_ECL, UMBRAL_SICR_PD, VIDA_ESPERADA

st.set_page_config(page_title="Riesgos · Cartera de tarjetas",
                   page_icon="▦", layout="wide")

DB_PATH = Path(__file__).parent / "riesgo_crediticio_taiwan.db"


def cargar_css(nombre: str) -> None:
    """Inyecta la hoja de estilo. Si falta el fichero, la app sigue
    funcionando con el tema por defecto de Streamlit."""
    ruta = Path(__file__).parent / nombre
    if ruta.exists():
        st.markdown(f"<style>{ruta.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


# El CSS se inyecta antes de pintar nada para que afecte tambien al titulo.
cargar_css("estilo_gestor.css")


@st.cache_data(show_spinner=False)
def contar_cartera() -> int:
    """Titulares puntuados. Si falta la tabla, lo dice en lugar de reventar:
    el paso previo es ejecutar precalcular_cartera.py."""
    try:
        with sqlite3.connect(str(DB_PATH)) as con:
            return con.execute("SELECT COUNT(*) FROM cartera_scoring").fetchone()[0]
    except sqlite3.Error:
        return 0


n_clientes = contar_cartera()
if n_clientes == 0:
    st.error("No encuentro la tabla cartera_scoring. "
             "Ejecuta antes:  python precalcular_cartera.py")
    st.stop()


# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### Configuración")
    api_key = st.text_input("API Key de Groq", type="password",
                            value=os.environ.get("GROQ_API_KEY", ""))
    modelo_llm = st.selectbox("Modelo LLM", ["openai/gpt-oss-120b"])

    if st.button("Reiniciar conversación"):
        st.session_state.historial = []
        st.rerun()

    st.markdown("### Cartera")
    st.caption(
        f"Titulares evaluados: **{n_clientes:,}**".replace(",", ".") + "  \n"
        "Producto: **tarjeta de crédito**  \n"
        "Observación: **abr–sep 2005**  \n"
        "Divisa: **NT$**"
    )

    ROTULO_CORTE = "sensible al coste" if CRITERIO_CORTE == "coste" else "tasa de impago"
    st.markdown("### Parámetros vigentes")
    st.caption(
        f"Punto de corte: **{PROB_CORTE:.3f}** ({ROTULO_CORTE})  \n"
        f"LGD: **{LGD:.0%}** (valor convencional)  \n"
        f"CCF: **{CCF:.0%}** del límite no dispuesto  \n"
        "Stage 3: **90+ días de retraso**  \n"
        f"Stage 2: **>30 días** o **PD > {UMBRAL_SICR_PD}**  \n"
        "Stage 1: **resto**"
    )

    st.markdown("### Modelo")
    st.caption(
        "Scorecard WOE · regresión logística  \n"
        "PD de retraso: **mes siguiente**  \n"
        f"PD de default: **12 meses** y **{VIDA_ESPERADA} meses**  \n"
        "Cadena de Markov por tramo de retraso"
    )

# Sin clave no se arranca el agente: se corta aqui antes de construirlo.
if not api_key:
    st.warning("Introduce tu API Key de Groq en la barra lateral para empezar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key


# ════════════════════════════════════════════════════════════════════
# PROMPT DEL SISTEMA
# Lo que debe ser determinista vive en las tools, no aquí.
# ════════════════════════════════════════════════════════════════════
SYSTEM = f"""Eres un asistente del area de riesgos de un banco. Tu interlocutor
es un gestor de cartera, no un cliente: puedes usar vocabulario tecnico (PD,
score, stage IFRS 9, EAD, ECL) sin traducirlo.

La cartera son titulares de tarjeta YA en vigor. Hay dos probabilidades distintas
y no deben confundirse:
- PD del mes siguiente: la del scorecard, probabilidad de RETRASO en el pago el
  mes que viene. Es la que ordena la cartera y decide el corte.
- PD a 12 meses y a toda la vida (36 meses): probabilidad de DEFAULT, llegar a 90
  dias de impago, obtenida con una cadena de Markov. Es la que usa la provision.
Un retraso es mucho mas frecuente que un default, de modo que la PD a 12 meses
puede ser menor que la del mes siguiente: no es una incoherencia, son sucesos
distintos. La PD de default se asigna por tramo de retraso (evaluacion colectiva):
es comun a todos los titulares del mismo tramo, y la diferenciacion individual la
da la PD del mes siguiente. Si te preguntan por la de un titular, explicalo asi. El modelo no evalua solicitudes nuevas ni otros horizontes: si te
preguntan por ellos, dilo con claridad en lugar de estimarlo.

HERRAMIENTAS
- resumen_cartera: cifras agregadas de un segmento (clientes, EAD, ECL, PD media).
- top_clientes: ranking por ECL o por PD. Por defecto ordena por ECL, que es lo
  util para priorizar: lo que cuesta no es la probabilidad, es la perdida.
- detalle_cliente: ficha de un titular con los factores que elevan su PD.
- simular_pago: contrafactual sobre el estado de pago reciente.
- alertas_recobro: titulares en mora ordenados por saldo en riesgo, con su
  estrategia de recobro, la accion para el gestor y el mensaje para el titular.
  Usala para preguntas de recobro: a quien llamar primero, que ofrecer, cuantos
  se recuperarian con un recordatorio.

REGLAS
1. Nunca des una cifra que no venga de una herramienta. Si no la tienes, llama a
   la herramienta o di que no puedes responder. No estimes, no redondees de
   memoria, no interpoles.
2. Los factores que elevan la PD son los que devuelve detalle_cliente. No
   atribuyas el riesgo a ninguna otra variable: el saldo pendiente en bruto, por
   ejemplo, no entra en el modelo; si lo hacen la utilizacion del limite y el
   esfuerzo de pago, que se calculan con el, y solo puedes citarlos si aparecen
   entre los factores devueltos.
3. El punto de corte es {PROB_CORTE:.3f} y la LGD {LGD:.0%}. No los cambies.
   {('El corte es sensible al coste: penaliza mas dejar pasar un impago, que hace '
     'perder el principal, que marcar a un buen pagador, que cuesta la gestion y el '
     'margen perdido. Por eso marca a mas titulares que la tasa de impago.')
    if CRITERIO_CORTE == 'coste' else 'El corte es la tasa de impago observada.'}
4. El limite recomendado es un criterio de gestion propuesto, no una politica
   calibrada. Presentalo como punto de partida para la decision del gestor.
5. Las cifras de ECL son ilustrativas. Las herramientas devuelven un campo
   `aviso` con sus supuestos: cuando des cifras de ECL o de stage, cierra con una
   frase que los recuerde, sin copiar el aviso entero. Si te piden usarlas para
   provisionar, adviertelo de forma explicita.
6. Responde breve. Cifras primero, una frase de lectura despues. Usa tablas
   cuando devuelvas varios titulares.
7. Los importes llegan ya formateados y en NT$. Reproducelos tal cual, sin
   reformatear, sin cambiar los separadores y sin convertir de moneda. La
   cartera es taiwanesa: nunca uses euros ni simbolo de euro.
8. Las alertas de recobro corresponden a la mora de agosto, el ultimo mes con
   codificacion consistente: dilo al darlas. La estrategia, la accion y el
   mensaje al titular los fija el modelo de recobro: comunicalos tal cual, sin
   reescribirlos ni cambiar la estrategia de ningun titular.
"""


@st.cache_resource(show_spinner=False)
def construir_agente(modelo: str):
    llm = ChatGroq(model=modelo, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agente = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agente, tools=TOOLS,
                         verbose=False, max_iterations=6,
                         handle_parsing_errors=True)


# ════════════════════════════════════════════════════════════════════
# INTERFAZ
# ════════════════════════════════════════════════════════════════════
st.title("Cartera de tarjetas · asistente de riesgos")
st.caption("Scoring de comportamiento · TFM Pablo Blas Hernández")

# Aviso permanente: el alcance del ECL tiene que estar visible en la interfaz,
# no solo cuando alguien lo pregunta.
st.info(AVISO_ECL, icon="ℹ️")

if "historial" not in st.session_state:
    st.session_state.historial = []

if not st.session_state.historial:
    st.markdown("**Preguntas de ejemplo**")
    c1, c2 = st.columns(2)
    ejemplos = [
        "¿Cuántos titulares superan el corte y cuánta provisión suman?",
        "Dame los 15 clientes con mayor pérdida esperada.",
        "¿Cuánta provisión concentra el Stage 3 y qué peso tiene en la cartera?",
        "¿Qué riesgo tiene el titular 4521 y qué factores lo elevan?",
        "¿A qué titulares en mora debo llamar primero?",
        "¿Cuántos titulares en mora se recuperarían con un recordatorio?",
    ]
    for i, texto in enumerate(ejemplos):
        with (c1 if i % 2 == 0 else c2):
            if st.button(texto, key=f"ej{i}", use_container_width=True):
                st.session_state.pendiente = texto
                st.rerun()

for autor, texto in st.session_state.historial:
    with st.chat_message(autor):
        st.markdown(texto)

pregunta = st.chat_input("Pregunta sobre la cartera")
if "pendiente" in st.session_state:
    pregunta = st.session_state.pop("pendiente")

if pregunta:
    st.session_state.historial.append(("user", pregunta))
    with st.chat_message("user"):
        st.markdown(pregunta)

    historial_lc = []
    for autor, texto in st.session_state.historial[:-1]:
        historial_lc.append(HumanMessage(content=texto) if autor == "user"
                            else AIMessage(content=texto))

    with st.chat_message("assistant"):
        with st.spinner("Consultando la cartera"):
            try:
                agente = construir_agente(modelo_llm)
                salida = agente.invoke({"input": pregunta,
                                        "chat_history": historial_lc})
                respuesta = salida["output"]
            except Exception as e:
                respuesta = (f"No he podido completar la consulta: {e}")
        st.markdown(respuesta)

    st.session_state.historial.append(("assistant", respuesta))
