import os
import time
import streamlit as st
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langsmith import traceable

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Nexus Digital Store", page_icon="🎮", layout="wide")

# --- 2. INICIALIZACIÓN DEL BACKEND (CACHÉ) ---
# Usamos @st.cache_resource para no recargar ChromaDB ni los modelos en cada clic del usuario
@st.cache_resource
def init_backend():
    load_dotenv()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Nexus-MultiTenant-UI"
    
    # Modelos
    llm = ChatGroq(model=os.getenv("GROQ_MODEL", "llama3-70b-8192"), temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # VectorStore (Catálogo Multi-Tenant Simulado)
    documentos = [
        {"text": "Cyberpunk 2077. RPG de acción en mundo abierto ambientado en Night City.", "metadata": {"tenant_id": "nexus_chile", "title": "Cyberpunk 2077", "price_clp": 35000, "stock": 15, "genres": "RPG, Acción"}},
        {"text": "Elden Ring. Juego de rol de acción y fantasía oscura.", "metadata": {"tenant_id": "nexus_chile", "title": "Elden Ring", "price_clp": 45000, "stock": 5, "genres": "RPG, Fantasía"}},
        {"text": "Stardew Valley. Simulador de granja y vida rural pacífica.", "metadata": {"tenant_id": "nexus_global_usd", "title": "Stardew Valley", "price_usd": 15, "stock": 100, "genres": "Simulación, Casual"}}
    ]
    texts = [doc["text"] for doc in documentos]
    metadatas = [doc["metadata"] for doc in documentos]
    vectorstore = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings, collection_name="nexus_store_ui")
    
    # Herramienta Externa
    wikipedia = WikipediaAPIWrapper(lang="es", top_k_results=1, doc_content_chars_max=400)
    
    # Grafo de Estado
    class AssistantState(TypedDict):
        query: str
        tenant_id: str
        context: str
        external_context: str
        response: str
        messages: Annotated[list[BaseMessage], add_messages]

    SYSTEM_PROMPT = """Eres NexusBot, el Asistente Experto en Ventas de PC Gaming.
    Tu objetivo es vender basándote en el catálogo interno, y enriquecer la charla usando el contexto externo.
    
    REGLAS ESTRICTAS:
    1. Si el stock es 0, advierte claramente que no hay unidades.
    2. Muestra los precios en la moneda del catálogo local.
    3. Utiliza la información de Wikipedia (Dato curioso) para entusiasmar al cliente.
    4. Piensa paso a paso (Chain-of-Thought) internamente antes de dar la respuesta final.

    CATÁLOGO DISPONIBLE (Interno):
    {context}

    DATO CURIOSO DEL JUEGO (Externo - Wikipedia):
    {external_context}
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

    @traceable(name="nexus_retrieval_ui")
    def retrieve(state: AssistantState) -> dict:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3, "filter": {"tenant_id": state["tenant_id"]}})
        docs = retriever.invoke(state["query"])
        
        context_parts = []
        juegos_encontrados = []
        for i, doc in enumerate(docs, 1):
            md = doc.metadata
            precio = f"${md.get('price_clp', 'N/A')} CLP" if "price_clp" in md else f"${md.get('price_usd', 'N/A')} USD"
            context_parts.append(f"[{i}] {md.get('title')} - {precio} | Stock: {md.get('stock')}")
            juegos_encontrados.append(md.get('title'))
            
        ctx_interno = "\n".join(context_parts) if context_parts else "Sin resultados en el catálogo."
        ctx_externo = "Sin datos externos."
        if juegos_encontrados:
            try:
                ctx_externo = wikipedia.run(juegos_encontrados[0] + " videojuego")
            except: pass
            
        return {"context": ctx_interno, "external_context": ctx_externo}

    @traceable(name="nexus_generation_ui")
    def generate(state: AssistantState) -> dict:
        cadena = prompt_template | llm | StrOutputParser()
        res = cadena.invoke({
            "context": state["context"],
            "external_context": state.get("external_context", ""),
            "messages": state["messages"]
        })
        return {"response": res, "messages": [AIMessage(content=res)]}

    workflow = StateGraph(AssistantState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Compilar con memoria
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

app_ecommerce = init_backend()

# --- 3. INTERFAZ VISUAL STREAMLIT ---
st.title("🎮 Nexus Digital Store - Asistente RAG")

# Control de Sesión (Memoria)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"sesion_{int(time.time())}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Barra lateral: Selector Multi-Tenant
with st.sidebar:
    st.header("⚙️ Entorno Comercial")
    tenant_seleccionado = st.radio(
        "Selecciona la región (Tenant):",
        ("nexus_chile", "nexus_global_usd"),
        format_func=lambda x: "🇨🇱 Tienda Chile (CLP)" if x == "nexus_chile" else "🌎 Tienda Global (USD)"
    )
    st.divider()
    if st.button("🗑️ Reiniciar Conversación"):
        st.session_state.messages = []
        st.session_state.thread_id = f"sesion_{int(time.time())}"
        st.rerun()

# Historial de Chat Visual
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Captura de input del cliente
if prompt := st.chat_input("¿Qué videojuego buscas hoy?"):
    # Renderizar pregunta del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invocar Grafo y Renderizar Respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Parámetros para LangGraph
        inputs = {
            "query": prompt,
            "tenant_id": tenant_seleccionado,
            "messages": [HumanMessage(content=prompt)]
        }
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        
        # Ejecución con spinner visual
        with st.spinner("Consultando stock y bases de datos externas..."):
            resultado = app_ecommerce.invoke(inputs, config=config)
            respuesta_final = resultado["response"]
            
        # Simulación de Streaming visual (Typewriter effect)
        texto_dinamico = ""
        for chunk in respuesta_final.split(" "):
            texto_dinamico += chunk + " "
            message_placeholder.markdown(texto_dinamico + "▌")
            time.sleep(0.02)
        message_placeholder.markdown(texto_dinamico)
        
    # Guardar respuesta en el estado visual
    st.session_state.messages.append({"role": "assistant", "content": texto_dinamico})