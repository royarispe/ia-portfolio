---
title: "Agentes con LangGraph: RAG, Tools y Memoria Conversacional"
date:
---

# Agentes con LangGraph: RAG, Tools y Memoria Conversacional

---

## 📝 Contexto

En este práctico trabajé con **LangGraph**, una librería orientada a la construcción de **agentes conversacionales estructurados** mediante grafos de estado.  
El foco estuvo en ir más allá del uso directo de un LLM, construyendo un **agente multi-turn** capaz de:

- Mantener estado entre turnos.
- Decidir cuándo responder directamente y cuándo llamar herramientas (*tools*).
- Integrar **RAG** como una tool reutilizable.
- Incorporar una **memoria ligera** para resumir la conversación.
- Orquestar todo mediante un **grafo explícito** de ejecución.

El notebook del Colab contiene el pipeline completo.  
Las llamadas al modelo no se ejecutan actualmente debido a que las **credenciales provistas originalmente para la cursada ya expiraron**, pero el código queda listo para funcionar sin cambios al reponer las API keys.

---

## 🎯 Objetivos

Los objetivos principales de este práctico fueron:

- Comprender el modelo mental de **LangGraph** (estado + nodos + transiciones).
- Diseñar un `AgentState` explícito para conversaciones multi-turn.
- Construir un agente que combine:
  - razonamiento con LLM,
  - recuperación de conocimiento (RAG),
  - tools auxiliares.
- Entender cómo los **tool calls** afectan el flujo de ejecución.
- Implementar memoria conversacional ligera mediante resúmenes.
- Explorar patrones reales usados en agentes de soporte y asistentes inteligentes.

---

## 🚀 Desarrollo

### 🤖 Parte 0 – Setup y primer agente mínimo

Comencé con un **agente LangGraph mínimo**, definiendo:

- Un estado básico (`messages`) que acumula el historial.
- Un único nodo `assistant` que llama al modelo con todo el historial.
- Un flujo lineal `START → assistant → END`.

Este primer paso permitió entender la diferencia clave entre:

- `llm.invoke("prompt")`
- y un **estado que viaja por un grafo**, siendo modificado en cada nodo.

✔ El agente ya es *stateful*, incluso en su versión más simple.

---

### 🧱 Parte 1 – Estado del agente con memoria ligera

Luego extendí el estado del agente agregando:

- `messages`: historial completo.
- `summary`: un resumen corto de la conversación.

Esta memoria ligera está pensada para:

- Reducir el contexto enviado al modelo.
- Mantener información clave sin reenviar todo el historial.
- Preparar el agente para conversaciones largas.

Aunque el resumen no es obligatorio en cada ejecución, dejar el estado preparado permite escalar el diseño sin refactorizaciones posteriores.

---

### 📚 Parte 2 – Construcción de un RAG mini como tool

En esta etapa armé un **RAG minimalista** con textos locales:

- Un pequeño corpus manual.
- Split en chunks con solapamiento.
- Embeddings de OpenAI.
- Vector store FAISS.

El objetivo no fue maximizar performance, sino **entender el patrón RAG desde cero** y convertirlo en una **tool reutilizable** (`rag_search`) que el agente pueda invocar cuando lo necesite.

Este enfoque refleja cómo se integran bases de conocimiento internas en agentes reales.

---

### 🛠️ Parte 3 – Tool adicional no-RAG

Además del RAG, agregué tools auxiliares simples, por ejemplo:

- Consulta de estado de pedidos ficticios.
- Obtención de la hora actual.

Estas tools simulan **servicios externos** típicos de un agente de soporte, y sirven para observar cómo el LLM decide cuándo delegar una respuesta a una herramienta.

---

### 🧠 Parte 4 – Tool calling y ToolNode

Aquí se dio el salto conceptual más importante del práctico:

- El LLM se *bindea* con una lista de tools.
- El agente puede responder directamente o emitir `tool_calls`.
- Un `ToolNode` ejecuta las tools solicitadas.
- El flujo vuelve al nodo `assistant`.

El grafo queda con un bucle explícito:

assistant ↔ tools

Esto hace visible algo que normalmente queda oculto en frameworks más “automáticos”:  
👉 **el razonamiento del agente y sus decisiones de control de flujo**.

---

### 💬 Parte 5 – Conversación multi-turn

Probé el agente en múltiples turnos:

- Primer mensaje: pregunta conceptual.
- Segundo mensaje: consulta que requiere usar RAG.
- Observación de cómo el estado evoluciona entre ejecuciones.

El mismo grafo se reutiliza, pero el **estado ya no es el inicial**, sino el resultado del turno anterior.

Este patrón es la base de cualquier asistente conversacional real.

---

### 🧪 Parte 6 – Memoria conversacional con summary (opcional)

De forma opcional, agregué un nodo `memory_node` que:

- Lee el historial reciente.
- Genera un resumen en pocos bullets.
- Actualiza `state["summary"]`.

Este diseño permite:

- Controlar cuándo se actualiza la memoria.
- Evitar enviar información sensible o irrelevante.
- Reducir costos y latencia en conversaciones largas.

---

### ⚡ Parte 7 – Interfaz con Gradio

Finalmente, se implementó una **UI simple con Gradio** para:

- Probar el agente sin editar código.
- Visualizar el historial de mensajes.
- Ver qué tools se activan en cada respuesta.
- Mantener el estado entre interacciones.

Esta interfaz acelera la experimentación y facilita la detección de errores en el comportamiento del agente.

---

### 📸 Evidencia

Notebook del desarrollo completo (incluyendo el RAG minimalista y el chatbot):

[📘 Enlace al Notebook de Google Colab](https://colab.research.google.com/drive/1rxQc42roHYtwAHZ41DyDDdx2wxK4mQ30?usp=sharing)

> Nota: las credenciales de OpenAI utilizadas durante la cursada ya no están activas, por lo que el código no ejecuta inferencias actualmente. El pipeline queda listo para funcionar al reponer las API keys correspondientes.

---

## 🧠 Reflexión Final

Este práctico marcó un punto de inflexión respecto al uso tradicional de LLMs:

- Pasé de **prompts aislados** a **agentes con estado explícito**.
- El uso de LangGraph hace visible el flujo de razonamiento y control.
- RAG como tool muestra por qué este patrón es esencial para respuestas confiables.
- La memoria ligera introduce preocupaciones reales de escalabilidad y privacidad.
- La separación entre reasoning, tools y memoria refleja arquitecturas usadas en producción.

En conjunto, este ejercicio consolida los fundamentos necesarios para avanzar hacia **agentes más complejos**, con planificación, herramientas externas y comportamiento consistente en el tiempo.

## 📚 Referencias

![LangGraph Documentation – State Graphs for LLM Agents](https://langgraph.langchain.com/)
![LangChain OpenAI Integration – ChatOpenAI](https://python.langchain.com/docs/integrations/chat/openai/)
![LangChain Tools – Definición e invocación de herramientas](https://python.langchain.com/docs/concepts/tools/)
![Retrieval-Augmented Generation (RAG) – Conceptos y patrones](https://python.langchain.com/docs/concepts/rag/)
![FAISS Vector Store – Búsqueda vectorial local](https://github.com/facebookresearch/faiss)
![LangChain Text Splitters – Chunking de documentos](https://python.langchain.com/docs/concepts/text_splitters/)
![LangGraph ToolNode – Ejecución de tools en grafos](https://langgraph.langchain.com/docs/concepts/tool_node/)
![LangChain Memory – Manejo de estado y contexto](https://python.langchain.com/docs/concepts/memory/)
![Gradio – Interfaces rápidas para ML y LLMs](https://www.gradio.app/)
![OpenAI Platform – Modelos y parámetros de generación](https://platform.openai.com/docs)
