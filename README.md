# 🚀 Nexus Digital Store - AI Sales Assistant & Multi-Tenant RAG

> Sistema agéntico de recuperación aumentada de información (RAG) con arquitectura Multi-Tenant, orquestado mediante LangGraph, LCEL y ChromaDB, optimizado para e-commerce global de videojuegos de PC.

---

## 🛠️ Stack Tecnológico y MLOps
* **Orquestación de Agentes:** LangGraph (Grafos de Estado y Memoria de Hilo).
* **Procesamiento Declarativo:** LCEL (LangChain Expression Language).
* **Recuperación Vectorial & Aislamientó:** ChromaDB con partición lógica por `tenant_id`.
* **Inferencia de Alta Velocidad:** SDK de Groq (Llama 3 / Modelos optimizados).
* **Telemetría y Observabilidad:** LangSmith (`@traceable` para tracking de tokens y latencia).
* **Gestión de Entorno:** Astral `uv` y `pyproject.toml`.

---

## 🏗️ Arquitectura de la Solución (Diagrama)

```mermaid
graph TD
    User((Usuario E-commerce)) -->|Query + tenant_id| StateGraph[LangGraph: AssistantState]
    
    subgraph Core RAG Multi-Tenant
        StateGraph --> NodeRetrieve[Nodo: retrieve]
        NodeRetrieve -->|Filtro estricto por metadatos| VectorDB[(ChromaDB)]
        VectorDB -->|Contexto Regional & Precios| NodeRetrieve
        NodeRetrieve --> NodeGenerate[Nodo: generate]
        NodeGenerate -->|Pipeline LCEL| LLM[Groq Inference Engine]
        LLM -->|Respuesta Tipada / Guardrails| NodeGenerate
    end
    
    NodeGenerate -->|Asistencia Comercial en Tiempo Real| User
    
    subgraph Capa de Auditoría MLOps
        NodeRetrieve -.->|@traceable| LangSmith[(LangSmith Telemetry)]
        NodeGenerate -.->|@traceable| LangSmith
    end