---
title: "LLMs con LangChain: Prompting, Plantillas y Salida Estructurada"
date:
---

# LLMs con LangChain: Prompting, Plantillas y Salida Estructurada

---

## 🤨 Objetivos de Aprendizaje

En este práctico trabajé con **LLMs integrados en LangChain**, explorando cómo estructurar prompts, controlar parámetros y obtener salidas robustas para aplicaciones reales. Al finalizar, pude:

- Instanciar modelos de OpenAI mediante `ChatOpenAI` y realizar llamadas básicas.
- Ajustar parámetros de decodificación: `temperature`, `max_tokens`, `top_p`.
- Diseñar **prompts reutilizables** con `ChatPromptTemplate` y componerlos con LCEL (`|`).
- Obtener **salidas estructuradas** usando `with_structured_output` (JSON/Pydantic).
- Enviar trazas y métricas a LangSmith para medir tokens, latencia y ejecución.
- Comparar enfoques zero-shot vs few-shot y cómo afectan la consistencia del modelo.
- Implementar pequeñas cadenas para traducción, resumen, Q&A y extracción de información.

---

## 📋 Contexto

Este práctico se centra en construir la base para un **pipeline profesional de LLMs**:

- Prompts claros y modulares  
- Plantillas reutilizables  
- Salidas predecibles  
- Observabilidad (tokens, latencia, logs)  
- Mini-aplicaciones sin dependencias externas  
- Primer paso hacia RAG y sistemas conversacionales

Antes de pasar a Retrieval, este práctico permite dominar los conceptos fundamentales del ecosistema LangChain + OpenAI.

---

## 🚀 Desarrollo

### 🔧 Parte 0 — Setup y “Hello LLM”

Para comenzar el práctico, instalé las dependencias necesarias de **LangChain**, **LangChain-OpenAI**, **LangSmith**, y utilidades opcionales. Luego configuré las API keys mediante variables de entorno.

El primer paso fue inicializar un modelo con:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
resp = llm.invoke("Definí 'Transformer' en una sola oración.")
print(resp.content)
```

✔ Esto confirmó que la conexión con el modelo funcionaba correctamente.  
✔ Además, `temperature=0` garantiza comportamientos deterministas, ideal para ejercicios evaluables.

---

#### 🧠 Setup con *Fill-in-the-blanks*

Luego completé los parámetros básicos necesarios:

```python
MODEL = "gpt-5-mini"
TEMP = 0.0

llm = ChatOpenAI(model=MODEL, temperature=TEMP)
print(llm.invoke("Hola! Decime tu versión en una línea.").content)
```
Probé variaciones cambiando `temperature` para observar diferencias en estilo, creatividad y estabilidad.

#### 📝 Observaciones iniciales

- Con **temperature = 0**, las respuestas son más **sobrias**, **directas** y **técnicas**.  
- Con **temperature = 0.7+**, aparecen más **adjetivos**, **metáforas**, cambios de tono y mayor variabilidad entre ejecuciones.  
- Estos experimentos permitieron ver cómo los parámetros de decodificación afectan incluso a prompts simples.

### 🧩 Parte 1 – Parámetros de decodificación (*temperature*, *max_tokens*, *top_p*)

En esta parte del práctico experimenté con los parámetros que controlan el comportamiento generativo del modelo, especialmente `temperature`, y observé cómo afectaban claridad, creatividad y estabilidad de las respuestas.

Primero probé con una pequeña batería de prompts, por ejemplo:

- `Escribí un tuit (<=20 palabras) celebrando un paper de IA.`
- `Dame 3 bullets concisos sobre ventajas de los Transformers.`

Para cada uno, ejecuté el modelo varias veces cambiando `temperature` (0.0, 0.5 y 0.9) y comparé los resultados.

También completé el bloque de *fill-in-the-blanks* configurando:

- `MODEL = "gpt-5-mini"`
- `TEMP` en distintos valores (`0.0`, `0.5`, `0.9`), pidiendo cosas como: *"Escribí un haiku sobre evaluación de modelos."*

Esto me permitió observar cómo responde el mismo modelo bajo distintos niveles de aleatoriedad.

#### 📝 Observaciones de esta parte

- Con **temperature = 0.0** las respuestas son:
  - Más sobrias, directas y técnicas.
  - Muy estables entre ejecuciones (casi idénticas si repito el mismo prompt).
- Con **temperature ≈ 0.5**:
  - Se mantiene la coherencia, pero aparecen variaciones ligeras en formulaciones y ejemplos.
  - Buen balance entre precisión y creatividad.
- Con **temperature ≥ 0.7**:
  - Surgen más adjetivos, metáforas y cambios de tono.
  - Mayor variabilidad entre ejecuciones: útil para creatividad, menos para tareas evaluables.

Además:

- `max_tokens` determina cuánto puede explayarse el modelo: si es muy bajo tiende a cortar las respuestas.
- `top_p` controla el *nucleus sampling*: al subirlo junto con `temperature` aumenta la diversidad, pero también el riesgo de respuestas menos controladas.
- En general, para tareas “cerradas” y evaluables, la combinación recomendada es **temperature baja** (cerca de 0) y `top_p` moderado, priorizando claridad y reproducibilidad.

### 🧩 Parte 2 – De texto suelto a plantillas con ChatPromptTemplate + LCEL

En esta sección trabajé con **ChatPromptTemplate**, una herramienta clave para separar instrucciones del contenido y construir prompts reutilizables.

La idea principal es armar una plantilla con estructura clara:

- Mensaje del sistema → define el estilo, tono y rol del asistente.
- Mensaje del usuario → contiene la variable dinámica que se completará en cada invocación.
- Encadenamiento con `|` (LCEL) → permite combinar prompt → modelo en una sola unidad ejecutable.

Ejemplo conceptual utilizado:

- Instrucción del sistema: *"Sos un asistente conciso, exacto y profesional."*
- Instrucción del usuario: *"Explicá {tema} en ≤ 3 oraciones, con un ejemplo real."*

Luego probé agregar **few-shot**, es decir, algún ejemplo previo dentro del prompt para guiar el estilo del modelo.  
Esto mejoró consistencia y redujo variabilidad en explicaciones más complejas.

#### 📝 Conclusiones de esta parte

- ChatPromptTemplate ayuda a evitar prompts largos y repetitivos.
- Few-shot mejora claridad cuando el dominio es específico.
- El operador `|` permite componer cadenas limpias, listísimas para producción.
- Separar contenido de instrucciones mejora trazabilidad y debugging en LangSmith.

---

### 🧱 Parte 3 – Salida estructurada (JSON confiable con Pydantic)

Esta sección fue clave: aprendí a generar **JSON válido y estructurado**, sin depender de “prompts frágiles” que piden *"devolvé JSON por favor"*.

LangChain permite usar:

- `with_structured_output`
- Un modelo Pydantic que define los campos obligatorios
- Validación automática del output del LLM

Trabajé con un esquema simple:

- `title`: string  
- `bullets`: lista de puntos

El modelo garantiza:

- Que los campos existan  
- Que el JSON sea válido  
- Que no falten claves  
- Que el formato sea consistente incluso en múltiples ejecuciones  

Esto evita todo el post-procesamiento tradicional y hace la integración MUCHO más robusta.

#### 📝 Conclusiones de esta parte

- La salida estructurada es esencial para pipelines automáticos.
- El LLM ya no devuelve “texto mezclado con JSON”, sino un objeto validado.
- Reduce errores y elimina parsing manual.
- Es ideal para resúmenes, extracción de información, reports y aplicaciones empresariales.

### 📏 Parte 4 – Métricas, Tokens y Observabilidad con LangSmith

En esta sección exploré cómo LangChain envía trazas automáticas a **LangSmith**, permitiendo observar:

- Uso de tokens  
- Tiempos de ejecución (latencia)  
- Estructura interna de cada “runnable”  
- Entrada y salida de cada componente (prompt, llm, parseo, etc.)

Después de ejecutar una cadena como `prompt | llm`, LangSmith registró automáticamente:

- Tokens de entrada  
- Tokens de salida  
- Costos estimados  
- Timeline del pipeline

Esto es fundamental para analizar rendimiento y evitar sorpresas en producción.

#### 📝 Reflexiones sobre observabilidad

- Algunos prompts consumen muchos más tokens de lo esperado (especialmente los que tienen few-shot).  
- Reducir el tamaño del contexto o simplificar instrucciones reduce tokens sin pérdida de calidad.  
- LangSmith hace muy fácil comparar ejecuciones y detectar prompts problemáticos.  
- La trazabilidad es clave cuando las cadenas crecen y se vuelven más complejas (especialmente al integrar RAG).

Esta parte del práctico permite tener control real sobre los costos y el comportamiento del modelo, algo imprescindible en sistemas basados en LLMs.

### 🧪 Parte 5 – Mini-tareas guiadas (Traducción, Resumen, Q&A y Extracción)

En esta sección apliqué lo aprendido para construir pequeñas funcionalidades útiles usando LLMs con LangChain.  
El objetivo fue practicar *prompting estructurado*, *plantillas*, *salida controlada* y *restricciones de formato*.

---

#### 🔤 1. Traductor determinista con salida estructurada

Implementé un traductor usando `with_structured_output`, garantizando que la salida respetara un esquema JSON fijo.  
Esto elimina la fragilidad de “pedir JSON por prompt” y depender del formato que el modelo quiera producir.

**Características:**
- `temperature=0` para máxima estabilidad.  
- Esquema Pydantic con campos `"text"` y `"lang"`.  
- Prompt simple y confiable.

**Resultado:**  
Obtengo siempre un objeto válido con texto traducido y el idioma destino, ideal para integrarlo en un pipeline.

---

#### 📝 2. Resumen ejecutivo con secciones obligatorias

Diseñé un prompt capaz de producir un **resumen estructurado**, con tres secciones:

- Introducción  
- Hallazgos  
- Recomendación  

Esto muestra cómo LangChain permite **forzar formato** sin necesidad de parseadores externos ni regex.  
La consistencia de estilo entre ejecuciones es muy superior a la de un prompt libre.

---

#### ❓ 3. Q&A con contexto “en crudo”

Creé un mini-sistema de pregunta–respuesta donde:

- Se provee un bloque de **contexto textual**.  
- El modelo **solo puede responder usando ese contexto**.  
- Si no alcanza, debe devolver *"No suficiente contexto"*.

Este pequeño ejercicio refleja los límites del prompting sin RAG:  
si el contexto no contiene la información, el modelo “adivina”.  
Imponer esta regla obliga al modelo a declarar insuficiencia de evidencia.

---

#### 🗂️ 4. Extracción de información (NER simplificado)

Finalmente, implementé un extractor estructurado que identifica:

- Título  
- Fecha  
- Entidades (ORG / PER / LOC)

Usando un esquema Pydantic, el modelo produce un JSON limpio con campos obligatorios y opcionales.  
Esta técnica es clave cuando se necesita *automatizar pipelines de datos* basados en texto libre.

---

### 📝 Reflexiones de esta parte

- La salida estructurada es uno de los **superpoderes** de LangChain: elimina casi todo el post-processing manual.  
- Los errores normales (como formato inconsistentes) desaparecen cuando se usa un esquema formal.  
- Q&A sin RAG tiene límites claros: si el contexto es pobre, el modelo rellena con supuestos.  
- Los resúmenes con plantillas mejoran la consistencia y facilitan controlar estilo y longitud.  
- Estas mini-tareas son pequeñas piezas que luego se reutilizan para construir agentes, chatbots y pipelines más complejos.

### 🧪 Parte 6 – Zero-shot vs Few-shot

En esta sección comparé dos enfoques fundamentales en prompting:

- **Zero-shot:** el modelo recibe solo una instrucción general.
- **Few-shot:** el modelo recibe ejemplos previos que guían su comportamiento.

El objetivo fue evaluar cómo cambia la consistencia del modelo ante tareas de **clasificación de sentimiento**.

---

#### 🧪 Zero-shot

En el enfoque zero-shot, el modelo recibe únicamente:

- Una instrucción clara.
- El texto a clasificar.
- Sin ejemplos previos de cómo debe verse la salida.

**Resultados observados:**

- Para textos muy positivos o muy negativos, acierta con buena precisión.
- Para textos neutrales, tiende a variar más entre ejecuciones.
- El formato de salida no siempre es estable (puede agregar explicación adicional).

Esto muestra que sin ejemplos, el modelo depende únicamente de su conocimiento previo y de la claridad del prompt.

---

#### 🧪 Few-shot (1–2 ejemplos)

Luego definí una plantilla con **dos ejemplos etiquetados**:

- Un caso claramente positivo.  
- Un caso claramente negativo.

La tercera entrada —la que yo quería clasificar— seguía el mismo formato.

**Resultados observados:**

- Mayor consistencia en el formato (devuelve solo la etiqueta).
- Menos variabilidad entre ejecuciones incluso con temperature > 0.
- Mejor manejo de los casos ambiguos, especialmente los neutrales.
- Menos propensión a extenderse en explicaciones.

El few-shot actúa como un *molde* que el modelo imita, reduciendo ambigüedad.

---

### 📝 Comparación Zero-shot vs Few-shot

| Aspecto | Zero-shot | Few-shot |
|--------|-----------|----------|
| Consistencia en el formato | Baja | Alta |
| Variabilidad entre ejecuciones | Alta | Muy baja |
| Manejo de ambigüedad | Regular | Mejor |
| Estabilidad en tareas evaluables | Baja | Alta |
| Control del estilo | Limitado | Excelente |

---

### 🧠 Reflexiones de esta parte

- El few-shot funciona como una **demostración explícita del comportamiento esperado**, y el modelo lo replica con gran precisión.
- Para tareas de clasificación con etiquetas cerradas (POS/NEG/NEU), few-shot es significativamente más confiable.
- En cambio, zero-shot es útil para rapidez o cuando no se dispone de ejemplos representativos.
- La elección entre ambos depende del costo, la necesidad de consistencia y el tipo de tarea.

Esta sección demuestra por qué los ejemplos son una herramienta tan poderosa en prompting y cómo pueden transformar la estabilidad del sistema sin necesidad de entrenamiento adicional.

### 🧩 Parte 7 – Resúmenes: Single-doc y Map-Reduce

En esta sección exploré cómo LangChain permite construir **pipelines de resumen** tanto para textos individuales como para múltiples documentos usando la estrategia *map-reduce*.

El objetivo fue observar cómo cambia la calidad del resumen cuando:

1. Se resume un texto completo directamente.  
2. Se fragmenta el texto, se resumen los fragmentos (*map*) y luego se combinan (*reduce*).

---

#### 📄 Resumen de un solo documento

Comencé definiendo un texto largo y aplicando un resumen directo.  
Esto permite evaluar:

- Si el modelo mantiene coherencia global.
- Qué tan bien conserva las ideas principales.
- Si respeta límites como “<=120 tokens” o “en 3 bullets”.

**Observaciones:**

- En textos cortos o medianos, el modelo realiza un buen resumen directo.
- En textos largos, tiende a omitir detalles relevantes o mezclar ideas distantes.
- Cuando el texto supera el contexto disponible del modelo, comienza a hallucinar o inventar detalles.

Esto motivó el uso del enfoque *map-reduce*.

---

#### 🗂️ Resumen Map-Reduce (chunking + combinación)

Luego dividí el texto largo en fragmentos manejables mediante un *text splitter*.

Para cada fragmento:

- Se aplicó un prompt que generaba **2–3 bullets claros y factuales**.
- Estos bullets se acumularon como resultados parciales (*map stage*).

Finalmente, en el paso *reduce*:

- Todos los bullets se consolidaron eliminando redundancias.
- Se generó un **resumen final conciso**, con límite de tokens.

---

### 📝 Comparación de ambos enfoques

| Criterio | Resumen directo | Map-Reduce |
|----------|------------------|------------|
| Manejo de textos largos | Regular | Excelente |
| Nivel de detalle | Medio | Alto (luego sintetiza) |
| Riesgo de hallucinaciones | Mayor | Menor (usa partes concretas) |
| Coherencia global | Buena | Muy buena (con reduce bien diseñado) |
| Costo computacional | Menor | Mayor |

---

### 🧠 Reflexiones de esta parte

- *Map-reduce* es claramente superior para textos extensos o documentos múltiples.
- La calidad del resumen depende fuertemente del **splitter** (chunk_size y overlap).
- El paso *reduce* permite controlar el estilo final del resumen:
  - Más ejecutivo  
  - Más técnico  
  - Más narrativo  

- Este patrón es el mismo que utilizan sistemas avanzados de RAG para generar respuestas consistentes basadas en múltiples documentos.

Esta parte fue clave para comprender cómo escalar resúmenes y preparar pipelines para sistemas de QA y RAG más robustos.

### 🧱 Parte 8 – Extracción de información (IE) con Salida Estructurada

En esta parte trabajé con **extracción de información (Information Extraction)** utilizando *salida estructurada* garantizada mediante `with_structured_output`.  
El objetivo fue obtener datos precisos desde texto libre, evitando parsing manual o formatos inconsistentes.

---

#### 🎯 Objetivo

Tomar un texto y extraer:

- Título (si lo hay)
- Fecha (si aparece explícita o implícita)
- Entidades nombradas (PERSONA, ORGANIZACIÓN, LUGAR)

Usando un esquema Pydantic, el modelo queda obligado a devolver **JSON válido**, cumpliendo tipos y estructura.  
Esto mejora drásticamente la confiabilidad comparado con pedir *"respondé en formato JSON"*.

---

### 🏗️ Diseño del esquema

Definí dos modelos:

- **Entidad(tipo, valor)** → para cada entidad detectada  
- **ExtractInfo(titulo, fecha, entidades[])** → estructura principal

Este enfoque garantiza:

- Campos siempre presentes (con opción de ser null/None si no existe el dato).
- Valores tipados y parseables.
- Respuestas consistentes, sin necesidad de regex o limpieza manual.

---

### 🧪 Ejemplo aplicado

Usé un texto como:

> “OpenAI anunció una colaboración con la Universidad Católica del Uruguay en Montevideo el 05/11/2025.”

El modelo devolvió información estructurada similar a:

- **titulo:** inferido a partir del evento
- **fecha:** "05/11/2025"
- **entidades:**  
  - ORG → OpenAI  
  - ORG → Universidad Católica del Uruguay  
  - LOC → Montevideo  

Esto demostró que el LLM es capaz no solo de entender el contenido, sino de categorizarlo según un esquema formal.

---

### 📝 Observaciones de esta parte

- La salida estructurada evita errores comunes como llaves mal cerradas, JSON inválido o formatos ambiguos.
- El LLM realiza una combinación de comprensión semántica y NER (Named Entity Recognition), produciendo resultados consistentes aun sin un modelo entrenado específicamente para NER.
- Si el texto no incluye fecha o título, el modelo rellena con `null`, manteniendo integridad del schema.
- Es un patrón ideal para:
  - Formularios automáticos  
  - Extracción de datos desde emails  
  - Preprocesamiento para pipelines legales o financieros  
  - Limpieza de información para bases de conocimiento  

---

### 🧠 Reflexiones

- En tareas de extracción, el **structured output** es esencial: reduce errores y simplifica todo el pipeline posterior.
- El modelo puede fallar en casos ambiguos (por ejemplo, fechas implícitas), pero el esquema ayuda a detectar fácilmente esos fallos.
- Este patrón es el bloque fundamental para construir:
  - Sistemas de ingestión documental  
  - Motores de búsqueda semántica  
  - Aplicaciones de RAG con metadatos enriquecidos  

### 🔎 Parte 9 – RAG básico con textos locales

En esta sección construí un **pipeline RAG minimalista**, sin fuentes externas, usando únicamente:

- Textos locales (pequeño corpus manual)
- Embeddings de OpenAI
- Un vector store FAISS
- Recuperación + generación mediante LangChain

El objetivo fue entender cómo funciona RAG desde cero, sin atajos ni magia oculta.

---

### 📚 Construcción del mini-corpus local

Creé un conjunto reducido de documentos, por ejemplo:

- “LangChain soporta structured output…”
- “RAG combina recuperación + generación…”
- “OpenAIEmbeddings permite indexar textos…”

Cada documento se encapsuló como `Document(page_content=...)`.

Luego apliqué un **text splitter** para generar chunks de ~300 caracteres con solapamiento, lo que mejora la recuperación en textos cortos.

---

### 🧠 Embeddings + Vector Store

Utilicé:

- **OpenAIEmbeddings** para representar semánticamente cada chunk.  
- **FAISS** como índice vectorial local, rápido y eficiente.

Esto permitió realizar búsquedas semánticas sin depender de servicios externos.

El `retriever` se configuró con `k=4` para recuperar los 4 fragmentos más relevantes al hacer una pregunta.

---

### 🧩 Cadena RAG

El pipeline se construyó así:

1. **Retriever** → obtiene los fragmentos más relevantes.  
2. **Prompt de combinación** → un template que dice:  
   “Respondé SOLO usando el contexto. Si no alcanza, decí ‘No suficiente contexto’.”  
3. **LLM (gpt-5-mini)** → genera la respuesta final basada exclusivamente en el contexto recuperado.

Este enfoque fuerza un comportamiento grounded, evitando alucinaciones.

---

### 🧪 Ejemplo de pregunta

Dada la consulta:

> “¿Qué ventaja clave aporta RAG?”

El sistema recuperó los fragmentos relevantes y produjo una respuesta del estilo:

- “RAG aporta grounding, combinando recuperación + generación para mejorar precisión y reducir alucinaciones.”

La respuesta se basó únicamente en el contenido del mini-corpus, tal como exigía el prompt.

---

### 📝 Observaciones de esta parte

- Con un corpus tan pequeño, los resultados son muy precisos, ya que las posibilidades son limitadas.  
- El comportamiento “No suficiente contexto” es crucial para distinguir cuándo el sistema puede responder y cuándo no.  
- Ajustar `k` cambia significativamente la calidad: valores altos pueden introducir ruido; valores bajos pueden dejar fuera información relevante.  
- Esta estructura es la base exacta de sistemas RAG más avanzados que integran PDFs, bases de conocimiento o web search.

---

### 🧠 Reflexiones

- Incluso un RAG minimalista muestra por qué este patrón es superior al prompting directo cuando hay conocimiento específico.  
- El control explícito del contexto evita que el modelo “invente”.  
- Este ejercicio es un paso esencial antes de construir chatbots, asistentes corporativos o sistemas de soporte con bases de conocimiento reales.

### 🤖 Desafío Integrador — Chatbot de Soporte “FAQ + WebSearch”

Para cerrar el práctico, implementé un **mini-chatbot de soporte** combinando tres componentes fundamentales:

1. **Un corpus local** con información del producto o dominio.  
2. **Un vector store FAISS** para recuperación semántica basada en embeddings.  
3. **Un LLM estructurado** encargado de generar respuestas finales con fuentes y nivel de confianza.  

Este desafío aplica todo lo aprendido: prompting, structured output, RAG básico y plantillas con LangChain.

---

### 📚 1. Corpus local (FAQs)

Para este ejercicio preparé un pequeño set de documentos representando información interna, por ejemplo:

- Cómo funciona cierto módulo o servicio  
- Preguntas frecuentes del usuario  
- Problemas comunes y soluciones  
- Definiciones o conceptos clave

Estos documentos fueron convertidos en objetos `Document` y luego divididos en chunks mediante `RecursiveCharacterTextSplitter`.

---

### 🧠 2. Indexación con Embeddings + FAISS

Utilicé:

- **OpenAIEmbeddings** para transformar cada chunk en un vector semántico.  
- **FAISS** como base vectorial local para consultas rápidas y eficientes.  

El retriever se configuró con `k=4` para obtener los fragmentos más relevantes ante cada pregunta del usuario.

Este paso establece la base RAG del sistema, donde la recuperación es responsable del grounding del modelo.

---

### 🧩 3. Template para la respuesta final

Luego diseñé un `ChatPromptTemplate` que combina:

- Pregunta del usuario  
- Fragmentos recuperados  
- (Opcional) Resultados de búsqueda web  
- Reglas para evitar alucinaciones  
- Formato requerido de salida  

La plantilla exige que el asistente responda **solo** en base al contexto proporcionado y declare explícitamente si la información no alcanza.

---

### 🧱 4. Salida estructurada con Pydantic

Para garantizar una respuesta confiable, definí un esquema:

```json
{
  "answer": "...",
  "sources": [
    {"title": "...", "url": "..."}
  ],
  "confidence": "low|medium|high"
}
```

Luego usé with_structured_output(...) para que el LLM produzca exactamente ese formato.

Esto permite integrarlo fácilmente en aplicaciones reales de soporte, dashboards o APIs.

### 🧪 5. Funcionamiento del chatbot

El flujo final quedó así:

   - Usuario hace una pregunta.
   - El sistema recupera los chunks más relevantes desde el vector store.
   - (Opcional) Realiza web search si el corpus no contiene suficiente información.
   - El LLM genera la respuesta estructurada:
   - Texto final para el usuario
   - Fuentes citadas
   - Nivel de confianza
   - Este patrón es el mismo utilizado en asistentes modernos empresariales.

### 📝 Reflexiones finales

El sistema muestra por qué RAG + structured output es el estándar actual en chatbots fiables.

La recuperación local evita alucinaciones y reduce costos.

El agregado de WebSearch permite resolver casos donde el corpus es insuficiente.

La estructura JSON permite integrar el bot sin fricciones en otros servicios.

Este desafío conecta perfectamente con lo que viene después: RAG completo, agentes, y tool use.

### 📸 Evidencia

Notebook del desarrollo completo (incluyendo el RAG minimalista y el chatbot):

[📘 Enlace al Notebook de Google Colab](https://colab.research.google.com/drive/1UfOP7aYD4-RUWG0lLadMHoB_zUtAA5V0?usp=sharing)