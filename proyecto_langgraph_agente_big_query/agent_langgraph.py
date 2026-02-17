# agent_langgraph.py

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import os
from langchain_core.messages import SystemMessage

# Cargar variables de entorno desde el archivo .env
from dotenv import load_dotenv
load_dotenv()  # Carga las variables del archivo .env automáticamente

# ============================================
# 1. IMPORTAR EL TOOL DESDE LA CARPETA TOOLS
# ============================================

# Importamos el tool de SQL desde la carpeta tools
# Esta versión ya viene decorada con @tool de LangChain
from tools.run_sql_query import run_sql_query_langchain as run_sql_query

# ============================================
# 2. ESQUEMA DE LA TABLA
# ============================================

TABLE_SCHEMA = """
CREATE TABLE `bigquery-public-data.new_york_citibike.citibike_trips` (
    tripduration INTEGER,
    starttime TIMESTAMP,
    stoptime TIMESTAMP,
    start_station_id INTEGER,
    start_station_name STRING,
    start_station_latitude FLOAT64,
    start_station_longitude FLOAT64,
    end_station_id INTEGER,
    end_station_name STRING,
    end_station_latitude FLOAT64,
    end_station_longitude FLOAT64,
    bikeid INTEGER,
    usertype STRING,
    birth_year INTEGER,
    gender STRING,
    customer_plan STRING
)
"""

# ============================================
# 3. INSTRUCCIONES DEL AGENTE
# ============================================

SYSTEM_INSTRUCTION = f"""
# 🧠 Agente Analista de Datos SQL

Eres un analista de datos experto que se especializa en escribir consultas SQL para Google BigQuery.
Tu única tarea es convertir las preguntas de los usuarios, hechas en lenguaje natural, en consultas SQL funcionales y precisas.

## El Contexto de los Datos

Tienes acceso a una sola tabla llamada `bigquery-public-data.new_york_citibike.citibike_trips`.
Este es el esquema de la tabla:

{TABLE_SCHEMA}

## Tu Proceso de Pensamiento

1. **Analiza la Pregunta del Usuario**: Comprende profundamente qué métricas, agregaciones, filtros y ordenamientos está pidiendo el usuario.
2. **Construye la Consulta SQL**: Escribe una consulta SQL para BigQuery que responda a la pregunta.
   - **SIEMPRE** usa el nombre completo de la tabla: `bigquery-public-data.new_york_citibike.citibike_trips`.
   - Presta atención a los tipos de datos. Por ejemplo, `tripduration` está en segundos.
   - No hagas suposiciones. Si la pregunta es ambigua, es mejor que la consulta falle a que devuelva datos incorrectos.
3. **Ejecuta la Consulta**: Usa la herramienta `run_sql_query` para ejecutar el SQL que has escrito.
4. **Interpreta los Resultados**: La herramienta te devolverá los datos en formato de texto (Markdown) o un mensaje de error.
   - Si obtienes datos, preséntalos al usuario de forma clara y responde a su pregunta original en un lenguaje natural y amigable.
   - Si obtienes un error, analiza el error, corrige tu consulta SQL y vuelve a intentarlo. No le muestres el error de SQL al usuario directamente a menos que no puedas solucionarlo. Explícale el problema en términos sencillos.

## Guía de Comunicación

- Tu respuesta final debe ser en español.
- No le digas al usuario que estás escribiendo SQL. Actúa como un analista que simplemente "encuentra" la respuesta.
- Si una consulta no devuelve resultados, dilo claramente. Por ejemplo: "No encontré viajes que cumplan con esos criterios".
- Si la pregunta es sobre la "ruta más popular", asume que se refiere a la combinación de `start_station_name` y `end_station_name`.

Empieza ahora.
"""

# ============================================
# 4. DEFINICIÓN DEL ESTADO DEL GRAFO
# ============================================

class AgentState(TypedDict):
    """Estado que se pasa entre los nodos del grafo"""
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ============================================
# 5. INICIALIZACIÓN DEL MODELO Y TOOLS
# ============================================

# Inicializar el modelo de OpenAI con LangChain
# Asegúrate de tener la variable de entorno OPENAI_API_KEY configurada
llm = ChatOpenAI(
    model="gpt-4o",  # Puedes usar: gpt-4o, gpt-4o-mini, gpt-4-turbo, o gpt-3.5-turbo
    temperature=0
)

# Lista de herramientas disponibles
tools = [run_sql_query]

# Bind tools al modelo
llm_with_tools = llm.bind_tools(tools)

# ============================================
# 6. FUNCIONES DE LOS NODOS DEL GRAFO
# ============================================

def should_continue(state: AgentState):
    """Decide si continuar ejecutando tools o terminar"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Si el último mensaje tiene tool_calls, continuamos a ejecutar tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Si no, terminamos
    return "end"

def call_model(state: AgentState):
    """Nodo que llama al modelo de lenguaje"""
    messages = state["messages"]
    
    # OpenAI soporta system messages nativamente, así que podemos usarlos directamente
    # Si es el primer mensaje, agregamos el system instruction
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_INSTRUCTION)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ============================================
# 7. CONSTRUCCIÓN DEL GRAFO DE LANGGRAPH
# ============================================

# Crear el grafo con el estado definido
workflow = StateGraph(AgentState)

# Crear el nodo de tools
tool_node = ToolNode(tools)

# Agregar nodos al grafo
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Definir el punto de entrada
workflow.set_entry_point("agent")

# Agregar edges condicionales
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Después de ejecutar tools, volver al agente
workflow.add_edge("tools", "agent")

# Compilar el grafo - EXPORTADO PARA LANGGRAPH STUDIO
app = workflow.compile()

# ============================================
# 8. FUNCIÓN PRINCIPAL PARA EJECUTAR EL AGENTE
# ============================================

def run_agent(query: str):
    """
    Ejecuta el agente con una consulta del usuario
    
    Args:
        query: Pregunta del usuario en lenguaje natural
        
    Returns:
        La respuesta final del agente
    """
    
    # Crear el mensaje inicial con el system instruction y la pregunta del usuario
    # OpenAI soporta SystemMessage nativamente
    initial_messages = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=query)
    ]
    
    # Ejecutar el grafo (usa la variable global 'app')
    result = app.invoke({"messages": initial_messages})
    
    # Obtener la respuesta final
    final_message = result["messages"][-1]
    return final_message.content

# ============================================
# 9. EJEMPLO DE USO
# ============================================

# if __name__ == "__main__":
#     # Verificar que las variables de entorno estén configuradas
#     if not os.getenv("OPENAI_API_KEY"):
#         print("⚠️  ERROR: La variable de entorno OPENAI_API_KEY no está configurada.")
#         print("Por favor configúrala con: export OPENAI_API_KEY='tu-api-key'")
#         print("Obtén tu API key en: https://platform.openai.com/api-keys")
#         exit(1)
    
#     # Ejemplo de preguntas
#     preguntas = [
#         "¿Cuántos viajes en total hay en la base de datos?",
#         "¿Cuál es la ruta más popular?",
#         "¿Cuál es la duración promedio de los viajes en minutos?"
#     ]
    
#     print("=" * 80)
#     print("🚴 AGENTE ANALISTA DE CITIBIKE CON LANGGRAPH + OPENAI")
#     print("=" * 80)
    
#     for i, pregunta in enumerate(preguntas, 1):
#         print(f"\n{'=' * 80}")
#         print(f"Pregunta {i}: {pregunta}")
#         print(f"{'=' * 80}\n")
        
#         try:
#             respuesta = run_agent(pregunta)
#             print(f"Respuesta: {respuesta}")
#         except Exception as e:
#             print(f"Error: {e}")
        
#         print()