"""
chatbot.py — Pre-evaluacion orientativa de riesgo de credito de tarjetas (Taiwan)

Herramienta ORIENTATIVA de revision de tarjetas YA EN VIGOR (scoring de
comportamiento): estima si el titular entrara en mora el mes siguiente y, a partir
de ahi, si su limite actual deberia mantenerse o revisarse a la baja. NO evalua
solicitudes de credito nuevas, para las que haria falta un modelo de admision.

El titular introduce sus datos sin identificarse (no se pide nombre, DNI ni
identificador). El chatbot no almacena datos identificativos; la consulta solo se
persiste, de forma seudonimizada, si el titular da su consentimiento explicito (paso 4).
El resultado no es vinculante: la revision real la decide el banco.

Flujo de atencion:
  1. Chatbot o agente humano -> si agente, deriva a telefono
  2. Recopilacion de las variables de comportamiento de la tarjeta
  3. Scoring de comportamiento -> nivel de riesgo orientativo (BAJO / ALTO)
  4. Recomendacion sobre el limite (mantener / revisar) e invitacion a la oficina

Tecnologias: Streamlit + LangChain (Agent) + Groq (GPT-OSS) + scikit-learn

Ejecutar:  streamlit run chatbot.py
Autor: Pablo Blas Hernandez
"""

import os
import re
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from tools_titular import TOOLS


# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Revisión de tarjeta",
                   page_icon="💳", layout="centered")


def cargar_css(nombre: str) -> None:
    """Inyecta la hoja de estilo. Si falta el fichero, la app sigue
    funcionando con el tema por defecto de Streamlit."""
    ruta = Path(__file__).parent / nombre
    if ruta.exists():
        st.markdown(f"<style>{ruta.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


# El CSS se inyecta antes de pintar nada para que afecte tambien al titulo.
cargar_css("estilo_titular.css")

st.title("Revisión orientativa de tu tarjeta")
st.caption("Estimación no vinculante del riesgo de pago · "
           "TFM Pablo Blas Hernández")


TELEFONO_FICTICIO = "999999999"

with st.sidebar:
    st.markdown("### Configuración")
    api_key = st.text_input("API Key de Groq", type="password",
                            value=os.environ.get("GROQ_API_KEY", ""))
    modelo_llm = st.selectbox("Modelo LLM", ["openai/gpt-oss-120b"])

    if st.button("Reiniciar conversación"):
        st.session_state.clear()
        st.rerun()

    depurar = st.checkbox("Modo depuración")

    st.markdown("### Tu revisión")
    st.caption(
        "Es una estimación orientativa y no vinculante.  \n"
        "No te pediremos tu nombre ni ningún dato identificativo."
    )

    st.markdown("### Prefieres hablar con alguien")
    st.caption(
        "Atención telefónica: **999 999 999**  \n"
        "De 9:00 a 18:00, de lunes a viernes"
    )

# Sin clave no se arranca el agente: se corta aqui antes de construirlo.
if not api_key:
    st.warning("Introduce tu API Key de Groq en la barra lateral para empezar.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key


# ════════════════════════════════════════════════════════════════════
# CATEGORÍAS Y CODIFICACIÓN DEL DATASET (Taiwan)
# ════════════════════════════════════════════════════════════════════
# Las variables PAY_n son ordinales; se explican para que el agente
# pueda mapear la respuesta del cliente al código numérico correcto.
CODIFICACION = """
- EDUCATION (nivel de estudios): 1=posgrado | 2=universidad | 3=secundaria | 4=otros
- MARRIAGE (estado civil): 1=casado | 2=soltero | 3=otros
- PAY_1..PAY_6 (estado de pago de los últimos 6 meses, siendo PAY_1 el más reciente):
    -2 = sin consumo ese mes
    -1 = pagó la totalidad
     0 = pagó el mínimo / pago revolving
     1..9 = meses de retraso en el pago (1 = un mes de retraso, etc.)
- LIMIT_BAL: límite de crédito YA CONCEDIDO en la tarjeta, en NT$ (numérico libre)
- AGE: edad en años, respuesta numérica libre
- BILL_AMT1..6: importe de la factura de cada mes (NT$), respuesta numérica
- PAY_AMT1..6: importe pagado cada mes (NT$), respuesta numérica
"""


# ════════════════════════════════════════════════════════════════════
# PROMPT DEL AGENTE
# ════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = f"""Eres el asistente virtual de un banco que ofrece una REVISION ORIENTATIVA del
riesgo de una tarjeta de credito YA EN VIGOR (scoring de comportamiento). El titular
introduce sus datos sin identificarse y tu le dices si su comportamiento de pago
situa su perfil en riesgo bajo o alto y que deberia ocurrir con su limite actual.
NO evaluas solicitudes de credito nuevas. NO es una decision vinculante y no se
guarda ningun dato salvo que el titular lo autorice al final: es una orientacion
para que pueda acudir despues al banco con una idea de su situacion.
El cliente es lo primero: se cercano, claro y respetuoso en todo momento.

FLUJO OBLIGATORIO que debes seguir en orden:

PASO 1 — Canal de atención.
Tu primer mensaje (ya enviado) pregunta si desea continuar con el asistente virtual (chatbot)
o hablar con un agente humano.
- Si elige AGENTE HUMANO: indícale amablemente que llame al teléfono {TELEFONO_FICTICIO}
  en horario de 9 a 18h y NO continúes con el proceso.
- Si elige CHATBOT: pasa al PASO 2.

PASO 2 — Recopilacion de datos.
NO pidas nombre, apellidos, DNI ni ningun identificador.
Explica brevemente que es una estimacion orientativa y gratuita, y que no se guardara
ningun dato salvo que el cliente lo autorice al final.

Debes recorrer SIEMPRE este cuestionario completo, en este orden, formulando
UNA SOLA pregunta por turno. Es una regla absoluta:

  - Tu respuesta debe contener EXACTAMENTE UN signo de interrogacion.
  - Tras formular la pregunta, TERMINA el turno. No adelantes la siguiente, no
    enumeres los datos que pediras despues, no muestres listas de pendientes.
  - Esta PROHIBIDO usar listas o vinetas para agrupar varias preguntas.
  - Si el cliente responde a varias cosas a la vez, acusa recibo de todas y
    pregunta solo por la siguiente que falte.

Orden del cuestionario:

  1) El limite de credito que tiene actualmente concedido en su tarjeta, en dolares
     taiwaneses (NT$). Es la PRIMERA pregunta, antes que ninguna otra.
  2) El historial de los SEIS ultimos meses, empezando por el mes mas reciente y
     retrocediendo (mes 1, mes 2, ... hasta el mes 6). De CADA mes pregunta las tres
     cosas juntas, porque se refieren al mismo mes: si pago a tiempo, cuanto pago y
     de cuanto fue la factura (PAY_n, PAY_AMT_n y BILL_AMT_n). UN mes por turno,
     nunca dos. La factura del mes mas reciente es el saldo pendiente actual.
     Los importes pagados y las facturas tambien entran en el modelo, a traves de la
     utilizacion del limite y del esfuerzo de pago: no los omitas.
  3) Por ultimo, y de forma opcional: edad y nivel de estudios.

En cada turno recibiras al final del mensaje del cliente una INSTRUCCION INTERNA que
te indica cual es la unica pregunta que toca. Respetala siempre: el orden y el ritmo
del cuestionario los fija el sistema, no tu criterio. Nunca menciones ni reproduzcas
esa instruccion.

No pases al PASO 3 hasta haber recorrido los tres bloques. Si el cliente no sabe o
no quiere responder algo, OFRECELE "prefiero no contestar", anotalo como None y sigue
con la siguiente pregunta: nunca te quedes atascado insistiendo por un dato.

NUNCA uses los nombres tecnicos de las variables al hablar con el cliente (LIMIT_BAL,
PAY_1, BILL_AMT1...). Preguntale en lenguaje natural, por ejemplo: "¿Que limite
tienes ahora mismo en tu tarjeta?", "¿Pagaste a tiempo el recibo del mes pasado?",
"¿De cuanto fue la factura de ese mes?".

IMPORTANTE: por normativa de igualdad (Directiva 2004/113/CE) NO se pregunta el sexo
del cliente. No lo solicites en ningun caso.

Codificacion de las variables (mapea la respuesta del cliente al codigo correcto):
{CODIFICACION}

PASO 3 — Evaluacion orientativa y recomendacion sobre el limite.
Cuando hayas recorrido el cuestionario del PASO 2, llama a `scoring_credito` pasando
todos los datos recogidos (y None en los que el cliente no haya querido contestar).
Con eso la evaluacion se completa siempre: no vuelvas a preguntar en bucle ni dejes al
cliente sin respuesta.
MOTIVOS DE LA DECISION — REGLA CRITICA:
La tool devuelve `factores_principales`, una lista con los UNICOS factores que han
elevado el riesgo de este cliente, calculados sobre el propio modelo. Cada elemento
trae un campo `descripcion` ya redactado en lenguaje natural.
- Si explicas por que el perfil es de riesgo, usa EXCLUSIVAMENTE las `descripcion`
  de esa lista. Como mucho las dos o tres primeras.
- Tienes PROHIBIDO citar como motivo cualquier dato que no aparezca en esa lista.
  En particular, el importe de la factura o el saldo pendiente EN BRUTO no son
  nunca un motivo: no intervienen directamente en la probabilidad de impago. Lo que
  si interviene es la utilizacion del limite y el esfuerzo de pago, que se calculan
  a partir de ellos: puedes citarlos SOLO si aparecen en `factores_principales`, y
  con la `descripcion` que trae la lista, nunca con tus propias palabras.
- Si `factores_principales` viene vacia, NO inventes ninguna explicacion: di
  simplemente que la evaluacion combina varios factores del historial de pagos y
  que en la oficina pueden detallarselos.
- Si `datos_no_facilitados` no esta vacia, avisa al cliente en una frase de que la
  estimacion se ha hecho con informacion incompleta y podria variar.

La tool te devuelve limite_actual, limite_recomendado, accion_recomendada, decision
y PD. Usalos asi:

- Si decision es SANO (perfil de BAJO riesgo): comunicale que su comportamiento de
  pago situa su perfil en riesgo bajo y que, a priori, su limite de
  {{{{limite_actual}}}} NT$ se mantendria sin cambios. Si te pregunta por una
  ampliacion, invitale a solicitarla en la oficina: eso requiere otra evaluacion.

- Si decision es RIESGO (perfil de ALTO riesgo): comunicaselo con tacto. Explicale que
  con ese comportamiento de pago el banco podria proponerle revisar su limite a la
  baja, hasta unos {{{{limite_recomendado}}}} NT$, como medida de proteccion frente al
  sobreendeudamiento, y que se trata de una orientacion, no de una decision tomada.
  Ademas, llama a `recomendar_credito` (con los datos que ya te dio) para mostrarle
  titulares de bajo riesgo con un perfil similar y que patron de pago los diferencia.

PASO 4 — Consentimiento de guardado (OPCIONAL).
Despues de darle el resultado, pregunta al titular si desea GUARDAR su consulta
(sin nombre ni datos identificativos, con una referencia autogenerada) para
agilizar los tramites si mas adelante acude al banco.
- Si ACEPTA: llama a `guardar_solicitud` y comunicale la referencia asignada.
- Si NO acepta: NO llames a guardar_solicitud; confirma que no se guardara nada.

PASO 5 — Cierre e invitacion al banco.
Termina SIEMPRE recordando que este resultado es ORIENTATIVO y no vinculante, e invita
al titular a acudir a una oficina del banco (o llamar al {TELEFONO_FICTICIO}) con este
resultado, donde un empleado revisara su caso si procede.

REGLAS:
- UNA sola pregunta por turno: un unico signo de interrogacion en toda tu
  respuesta. Nunca un cuestionario largo de golpe, nunca listas de preguntas.
- FORMATO: escribe en texto corriente, en frases normales. NO uses titulos ni
  encabezados (#, ##, ###), NO pongas la pregunta en negrita ni en cursiva, NO
  uses vinetas ni numeracion. Dos o tres frases seguidas, nada mas.
- NUNCA menciones al cliente los nombres tecnicos de las variables (LIMIT_BAL, PAY_1,
  BILL_AMT1, etc.). Traducelos siempre a lenguaje natural: "el limite de tu tarjeta",
  "si pagaste a tiempo el ultimo mes", "el saldo que tienes pendiente".
- SIEMPRE debes terminar comunicando el nivel de riesgo y la recomendacion sobre el
  limite (mantener o revisar a la baja), con su importe. Si una herramienta devuelve un error, no lo muestres al
  cliente: pide con naturalidad el dato que falte o continua con los datos
  que ya tengas, pero nunca acabes el turno sin dar la decision.
- NUNCA inventes ni deduzcas por tu cuenta los motivos de la decision. Los motivos
  son solo los de `factores_principales`. No razones a partir del saldo, del limite
  ni de ningun otro dato que no venga en esa lista.
- NUNCA pidas nombre, DNI ni identificador.
- Solo guardas la consulta si el cliente da su consentimiento explicito (PASO 4).
- Nunca expongas la PD numerica exacta salvo que el cliente la pida.
- Recuerda al cliente que es una PRE-EVALUACION ORIENTATIVA, no una decision vinculante.
- Nunca preguntes ni menciones el sexo del cliente.
"""


def limpiar_formato(texto: str) -> str:
    """Neutraliza el Markdown que a veces emite el modelo y que Streamlit
    renderiza como titulo gigante o cursiva desbordada.

    Tres problemas reales observados:
      - lineas que empiezan por #, ## o ### -> encabezado enorme
      - asteriscos o guiones bajos sin cerrar -> el enfasis se desborda al
        resto del mensaje
      - vinetas para una sola pregunta, que rompen el tono conversacional
    """
    # Bloques de codigo: Streamlit los pinta en monoespaciado pequeno y el
    # resaltador colorea el contenido (de ahi el texto verde). Se conserva el
    # contenido y se quitan las marcas de apertura y cierre.
    texto = re.sub(r"```[a-zA-Z0-9_+-]*\n?", "", texto)
    texto = texto.replace("~~~", "")

    # Sintaxis de color propia de Streamlit: :green[texto] -> texto
    texto = re.sub(r":(?:red|orange|green|blue|violet|gray|grey|rainbow)\[(.*?)\]",
                   r"\1", texto, flags=re.S)

    # Codigo en linea: en una conversacion no aporta y descuadra el tono.
    texto = texto.replace("`", "")

    lineas = []
    for linea in texto.split("\n"):
        # Encabezados Markdown al principio de linea
        linea = re.sub(r"^\s{0,3}#{1,6}\s*", "", linea)
        # Vinetas sueltas (- * +) al principio de linea
        linea = re.sub(r"^\s{0,3}[-*+]\s+", "", linea)
        lineas.append(linea)
    limpio = "\n".join(lineas)

    # Enfasis sin cerrar: si el numero de marcas es impar, se eliminan todas
    # (preferimos texto plano a una cursiva que se come medio mensaje).
    for marca in ("***", "**", "*", "___", "__", "_"):
        if limpio.count(marca) % 2 != 0:
            limpio = limpio.replace(marca, "")

    # CAUSA PRINCIPAL del formato roto: Streamlit interpreta $...$ como LaTeX.
    # Como la moneda del dataset es el dolar taiwanes, cada mensaje contiene
    # varios "NT$": el primero abre modo matematico y el segundo lo cierra, de
    # modo que el texto intermedio se renderiza con tipografia de formula y el
    # simbolo desaparece. Se escapa el dolar para que salga literal.
    limpio = re.sub(r"(?<!\\)\$", r"\\$", limpio)

    return limpio.strip()


# ════════════════════════════════════════════════════════════════════
# GUION DEL CUESTIONARIO
# El orden y el ritmo NO se dejan al criterio del modelo de lenguaje: se
# controlan aqui. En cada turno se inyecta cual es la unica pregunta que debe
# formular. Es el mismo principio que con los factores explicativos: lo que
# debe ser determinista vive en codigo, no en el prompt.
# ════════════════════════════════════════════════════════════════════
GUION = [
    "el limite de credito que tiene actualmente concedido en su tarjeta, en NT$",
    "del ULTIMO mes: si pago a tiempo el recibo, que importe pago y de cuanto fue la factura",
    "lo mismo referido a hace DOS meses: si pago a tiempo, cuanto pago y el importe de la factura",
    "lo mismo referido a hace TRES meses",
    "lo mismo referido a hace CUATRO meses",
    "lo mismo referido a hace CINCO meses",
    "lo mismo referido a hace SEIS meses",
    "su edad y su nivel de estudios, aclarando que ambos son opcionales",
]


def directiva_turno(paso: int) -> str:
    """Instruccion interna que fija la unica pregunta de este turno."""
    if paso < len(GUION):
        return ("\n\n[INSTRUCCION INTERNA, no la menciones ni la reproduzcas] "
                f"En tu respuesta formula UNA SOLA pregunta: {GUION[paso]}. "
                "Texto corriente: sin titulos, sin negrita, sin cursiva, sin "
                "vinetas, sin bloques de codigo ni acentos graves. Un unico signo de "
                "interrogacion. No adelantes ni enumeres las "
                "preguntas siguientes. Si el cliente ya te ha dado ese dato, acusa "
                "recibo y pregunta directamente por el siguiente del guion.")
    return ("\n\n[INSTRUCCION INTERNA, no la menciones ni la reproduzcas] "
            "El cuestionario esta completo: llama ya a scoring_credito con todos los "
            "datos recogidos (None en los que no haya querido contestar) y comunica "
            "el resultado siguiendo el PASO 3.")


@st.cache_resource
def get_agent(model_name: str):
    llm = ChatGroq(model=model_name, temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=False,
                         handle_parsing_errors=True, max_iterations=12,
                         return_intermediate_steps=True)


# ════════════════════════════════════════════════════════════════════
# GRÁFICO DE PERFILES SIMILARES (recomendador KNN)
# ════════════════════════════════════════════════════════════════════
def grafico_vecinos(reco: dict):
    """Grafica el límite de crédito vs saldo de los clientes sanos similares."""
    vecinos = reco.get("vecinos_sanos", [])
    if not vecinos:
        return None

    # Se usan solo las variables de comportamiento: los perfiles de referencia
    # no llevan identificador directo (coherente con el resto de la consulta).
    limites = [v.get("LIMIT_BAL") for v in vecinos]
    saldos  = [v.get("BILL_AMT1") for v in vecinos]
    etiquetas = [f"Perfil {i}" for i in range(1, len(vecinos) + 1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=limites, y=saldos, mode="markers+text",
        name="Clientes sanos similares",
        marker=dict(size=14, color="#2E86AB", opacity=0.7),
        text=etiquetas, textposition="top center"
    ))
    fig.update_layout(
        title="Perfiles de clientes sanos con comportamiento similar",
        xaxis_title="Límite de crédito (NT$)",
        yaxis_title="Saldo pendiente reciente (NT$)",
        height=420, legend=dict(orientation="h", y=-0.25)
    )
    return fig


# ════════════════════════════════════════════════════════════════════
# ESTADO
# ════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "paso_guion" not in st.session_state:
    st.session_state.paso_guion = 0   # posicion actual dentro de GUION

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("grafico"):
            st.plotly_chart(go.Figure(msg["grafico"]), use_container_width=True)

if not st.session_state.messages:
    welcome = (
        "¡Bienvenido a la revisión orientativa del riesgo de tu tarjeta! 💳\n\n"
        "Te daré una **estimación orientativa** de tu perfil de riesgo de "
        "pago y de si tu límite actual debería mantenerse o revisarse. No pediré tu "
        "nombre ni ningún dato identificativo. Al final podrás decidir si quieres "
        "guardar tu consulta o no.\n\n"
        "¿Cómo prefieres continuar?\n\n"
        "1️⃣ **Continuar con el asistente virtual** (rápido, ahora mismo)\n"
        "2️⃣ **Hablar con un agente humano**\n\n"
        "Escribe **1** o **2**."
    )
    with st.chat_message("assistant"):
        st.markdown(welcome)
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.session_state.chat_history.append(AIMessage(content=welcome))


# ════════════════════════════════════════════════════════════════════
# INPUT
# ════════════════════════════════════════════════════════════════════
user_input = st.chat_input("Escribe tu respuesta...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    agent = get_agent(modelo_llm)
    with st.chat_message("assistant"):
        with st.spinner("Procesando..."):
            grafico = None
            # 1) Ejecucion del agente: es lo imprescindible (el resultado del scoring).
            try:
                # La directiva viaja pegada al mensaje del turno, pero NO se guarda
                # en chat_history (mas abajo se almacena el user_input limpio), de
                # modo que no contamina los turnos siguientes.
                entrada = user_input + directiva_turno(st.session_state.paso_guion)
                response = agent.invoke({
                    "input": entrada,
                    "chat_history": st.session_state.chat_history,
                })
                bruto = response["output"]
                respuesta = limpiar_formato(bruto)
                if depurar and bruto != respuesta:
                    st.caption("Respuesta en crudo del modelo:")
                    st.code(repr(bruto), language=None)
                if st.session_state.paso_guion < len(GUION):
                    st.session_state.paso_guion += 1
            except Exception as e:
                respuesta = ("Lo siento, ha habido un problema al procesar tu consulta. "
                             "¿Puedes repetir el ultimo dato, por favor?")
                st.caption(f"Detalle tecnico: {e}")
                response = None

            # 2) Grafico opcional: si falla, se pierde SOLO el grafico, nunca la
            #    respuesta con el resultado del scoring.
            if response is not None:
                try:
                    for action, observation in response.get("intermediate_steps", []):
                        if action.tool == "recomendar_credito":
                            reco = observation if isinstance(observation, dict) else json.loads(observation)
                            if "vecinos_sanos" in reco:
                                grafico = grafico_vecinos(reco)
                except Exception:
                    grafico = None   # el grafico es accesorio: no interrumpe la conversacion

        st.markdown(respuesta)
        if grafico is not None:
            st.plotly_chart(grafico, use_container_width=True)
            st.info("📊 Para un análisis más completo, consulta el **dashboard de Tableau** "
                    "conectado a la base de datos del banco.")

    msg = {"role": "assistant", "content": respuesta}
    if grafico is not None:
        msg["grafico"] = grafico.to_dict()
    st.session_state.messages.append(msg)
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=respuesta))
