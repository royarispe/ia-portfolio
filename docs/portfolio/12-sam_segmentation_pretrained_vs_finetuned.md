---
title: "Segment Anything Model (SAM): Pretrained vs Fine-tuned en Segmentación de Inundaciones"
date:
---

# Segment Anything Model (SAM): Pretrained vs Fine-tuned en Segmentación de Inundaciones

---

## 📝 Contexto

En este práctico trabajé con el modelo **Segment Anything Model (SAM)**, uno de los modelos fundacionales más conocidos para **segmentación de imágenes**. La idea central fue comparar el comportamiento de:

- **SAM pre-entrenado (zero-shot)** sobre un nuevo dominio (inundaciones),
- vs. **SAM fine-tuneado** específicamente sobre un dataset de segmentación de áreas inundadas.

El caso de uso está vinculado a un escenario real de **respuesta a desastres**:

- Identificar áreas inundadas en imágenes satelitales o aéreas.
- Medir extensión de agua para apoyo a organismos de emergencia.
- Tener mapas de inundación más precisos para planificación y mitigación.

A diferencia de otros prácticos, acá no se trata de clasificar una etiqueta, sino de predecir **máscaras de segmentación píxel a píxel** sobre imágenes complejas, donde el agua puede confundirse con sombras, reflejos, nubes u otros elementos visualmente similares.

Todo el pipeline (descarga de dataset, exploración, inferencia con SAM pre-entrenado, fine-tuning y evaluación) se implementó en Google Colab.

## 🎯 Objetivos del Práctico

En este práctico trabajé con **SAM (Segment Anything Model)** aplicándolo al caso de negocio de **segmentación de áreas inundadas** a partir de imágenes reales.  
Los objetivos específicos fueron:

- Comprender cómo funciona SAM en modo **zero-shot** (pre-entrenado).
- Explorar distintos tipos de **prompts**: point prompts y box prompts.
- Evaluar su desempeño en un dominio completamente nuevo.
- Preparar un pipeline para **fine-tuning** del modelo en un dataset específico.
- Comparar métricas antes y después del fine-tuning (IoU, Dice, Precision, Recall).
- Realizar un análisis cualitativo y de errores para identificar mejoras y limitaciones.

---

## ⚙️ Setup e Instalación

Para este práctico se instalaron las dependencias necesarias para correr SAM, PyTorch, OpenCV, Albumentations y herramientas de visualización.  
Se configuraron:

- GPU si estaba disponible.
- Seed global para reproducibilidad.
- Librerías SAM (`segment-anything`) y predictores (`SamPredictor`).

Este setup permitió ejecutar tanto la inferencia zero-shot como el entrenamiento del modelo.

---

## 🌊 Dataset Utilizado: Flood Area Segmentation

El dataset seleccionado proviene de **Kaggle**, con imágenes reales de zonas inundadas:

- ~290 imágenes RGB.
- Máscaras binarias asociadas (agua vs no agua).
- Ideal para segmentación supervisada.
- Representa un caso real de monitoreo de desastres.

La estructura del dataset una vez descomprimido es:

flood_dataset/
├── Image/ # Imágenes originales (.jpg)
├── Mask/ # Máscaras binarias (.png)
└── metadata.csv

Se realizó:

- Descarga mediante API de Kaggle.
- Exploración de estructura de carpetas.
- Carga de imágenes y máscaras.
- Visualización inicial para validar integridad y variedad del dataset.
- Cálculo de estadísticas (tamaños, proporción de píxeles de agua, etc.).

Todo esto constituyó la base para los experimentos posteriores con SAM.

## 🚀 Desarrollo

### 🧠 Parte 1 — Inferencia con SAM Pre-Entrenado

En esta sección trabajé con el modelo **SAM (Segment Anything Model)** en modo *zero-shot*, es decir, sin ningún tipo de entrenamiento adicional previo.  
El objetivo fue evaluar qué tan bien SAM podía segmentar correctamente áreas inundadas **sin estar adaptado** a este dominio específico.

---

### 🔧 Carga del Modelo y Predictor

Se utilizó el checkpoint oficial **`sam_vit_b`**, cargado desde el repositorio original.  
Luego, se creó un `SamPredictor`, encargado de ejecutar inferencia interactiva.

SAM se probó usando dos tipos de indicaciones (*prompts*):

- **Point prompts:** un punto dentro de la región de agua.
- **Box prompts:** una caja que delimita aproximadamente el área inundada.

Esto permitió evaluar cuánta información necesita el modelo para funcionar correctamente.

---

### 📌 Experimentos con Point Prompts

El flujo fue:

1. Seleccionar una imagen real del dataset.
2. Ubicar un punto dentro del área con agua.
3. Pasarlo a SAM como indicación.
4. Obtener una máscara segmentada y la puntuación de confianza.

**Hallazgos:**

- SAM detecta bien zonas de agua marcadas con puntos centrales.
- Tiende a extender la máscara más de lo necesario.
- Tiene dificultades en bordes finos y zonas con reflejos.

Se visualizaron:

- Imagen con punto marcado.
- Ground truth de máscara real.
- Predicción de SAM.
- Overlay de la máscara para interpretar aciertos y fallos.

---

### 📦 Experimentos con Box Prompts

Para este caso:

1. Se generó una bounding box a partir de la máscara real.
2. El prompt resultó más informativo y preciso que el punto.
3. SAM respondió con máscaras más consistentes y mejor definidas.

**Observaciones:**

- Box prompts reducen significativamente falsos positivos.
- La segmentación es más estable que con point prompts.
- Aun así, los bordes pueden ser imprecisos.

Se incluyó un análisis visual comparando ground truth vs predicción, además de una vista de diferencias (FP/FN).

---

### 📊 Métricas Iniciales (Zero-Shot)

Se calcularon las métricas más comunes para segmentación:

- **IoU**
- **Dice**
- **Precision**
- **Recall**

Estas evaluaciones iniciales sirvieron de baseline para luego comparar con el modelo fine-tuned.

### 🧠 Parte 2 — Evaluación Completa del Modelo Pre-Entrenado (Zero-Shot Benchmark)

Luego de las pruebas iniciales con prompts individuales, se realizó una evaluación cuantitativa completa sobre el *test set* para medir el rendimiento real de SAM en modo zero-shot sobre el dominio de **flood segmentation**.

---

### 📊 Métricas Evaluadas

Para cada imagen se midieron:

- **IoU (Intersection over Union):** mide superposición entre predicción y máscara real.
- **Dice Coefficient:** más sensible para segmentaciones con áreas pequeñas.
- **Precision:** proporción de píxeles predichos como agua que realmente lo son.
- **Recall:** capacidad del modelo para capturar todas las zonas de agua.

Estas métricas permiten entender diferentes comportamientos:  
- *Precision alta + Recall bajo* → el modelo es conservador, no detecta todo.  
- *Recall alto + Precision baja* → el modelo se “pasa”, detecta agua donde no la hay.  

---

### 🧪 Evaluación con Point Prompts

Para cada imagen:

1. Se localizó un punto dentro del área de agua (ground truth).
2. Se generó la máscara desde SAM usando ese punto.
3. Se calcularon métricas per-image.

**Resultados observados:**

- IoU promedio moderado, con gran variabilidad entre imágenes.
- Buen desempeño en áreas amplias de agua.
- Fallos más frecuentes en zonas delgadas, ríos estrechos o fragmentados.
- Sensibilidad a reflejos y sombras, lo que generó falsos positivos.

Se graficaron histogramas de distribución para entender la dispersión de resultados.

---

### 📦 Evaluación con Box Prompts

El mismo proceso anterior, pero usando bounding boxes derivadas automáticamente del ground truth.

**Hallazgos clave:**

- IoU y Dice significativamente mejores que con point prompts.
- Mucha menor varianza → comportamiento más estable.
- Mejor representación de bordes y contornos.
- Reduce falsos negativos, ya que la box delimita mejor la región de interés.

---

### 📊 Comparación Global Point vs Box

Se observó una tendencia clara:

| Prompt | IoU Promedio | Dice Promedio | Observaciones |
|-------|--------------|----------------|----------------|
| **Point** | Menor | Menor | Altamente dependiente del punto elegido, más ruido |
| **Box** | Mayor | Mayor | Más estable y más cercano a la máscara real |

Además, se visualizaron histogramas comparados para IoU, Dice, Precision y Recall, evidenciando que:

- **Los box prompts desplazan la distribución completa hacia mejores valores.**
- **El pretrained SAM no está adaptado a patrones visuales de inundación**, lo cual limita su desempeño sin fine-tuning.

---

### 📝 Conclusiones de la Evaluación Zero-Shot

- SAM es muy poderoso para segmentación general, pero **no está optimizado para fenómenos como inundaciones**, donde:
  - el agua puede tener múltiples colores,
  - hay reflejos intensos,
  - existen bordes irregulares,
  - hay mucha variabilidad entre escenas.

- **Zero-shot funciona**, pero **no es suficiente para aplicaciones críticas**, especialmente en contextos de disaster response.

Esta evaluación sirvió como *baseline* para comparar con el modelo entrenado específicamente en el dataset (fine-tuned SAM).

### 🧠 Parte 3 — Preparación del Dataset y Fine-Tuning de SAM

Tras evaluar el desempeño del modelo pre-entrenado, avanzamos hacia el objetivo principal del práctico:  
**adaptar SAM al dominio de inundaciones mediante fine-tuning supervisado**.

Esta sección detalla la construcción del dataset, normalización, generación de prompts automáticos, creación de DataLoaders y setup del entrenamiento.

---

### 📁 3.1 — Construcción del Dataset Personalizado

SAM no está diseñado originalmente para entrenarse fácilmente; su arquitectura requiere un *workflow* especial:

- Imágenes deben ser redimensionadas a **1024×1024** (tamaño nativo del encoder de SAM).
- Albumentations se utiliza para aplicar augmentations consistentes entre imagen y máscara.
- Se generan prompts automáticos:
  - **Point prompt**: se selecciona un punto aleatorio dentro del agua.
  - **Box prompt**: se deriva del bounding box del ground truth.

El dataset implementa:

- Redimensionamiento fijo  
- Augmentations (flip, rotate, brightness/contrast)  
- Conversión a tensores PyTorch  
- Generación del prompt correspondiente por muestra  
- Retorno de la máscara original para métricas posteriores  

Esto permite entrenar SAM con batches pequeños (1–4 imágenes) sin inconsistencias de tamaño.

---

### 📦 3.2 — Creación de DataLoaders

Los DataLoaders deben manejar prompts variables, por lo que se implementó un `collate_fn` especial.

Puntos clave:

- Todas las imágenes ya vienen en 1024×1024 → se pueden apilar sin problemas.
- Los prompts se manejan como listas para preservarlos por individuo.
- `batch_size` bajo (2–4) debido al alto consumo de memoria del encoder de SAM.
- `shuffle=True` en entrenamiento, `shuffle=False` en validación.

Esto garantiza:

- Entrenamiento estable  
- Manejo correcto de prompts  
- Máxima utilización de GPU sin OOM  

---

### 🧮 3.3 — Funciones de Pérdida (Loss Functions)

Al ser una tarea de segmentación binaria, utilizamos:

- **Binary Cross Entropy (BCE):** buena para clasificación pixel a pixel  
- **Dice Loss:** ideal para máscaras con clases desbalanceadas (agua vs fondo)

Se define la pérdida combinada:

\[
\text{Loss} = 0.5 \cdot BCE + 0.5 \cdot Dice
\]

Esto ayuda al modelo a aprender tanto la localización como la forma completa de la región inundada.

---

### 🔧 3.4 — Configuración del Fine-Tuning

Una decisión crítica:

#### 🔒 Congelamos el *image encoder*
Porque:
- Es costoso de entrenar (≈300M parámetros).
- Ya es muy bueno extrayendo características generalistas.
- Evitamos sobreajuste con dataset pequeño.
- Ahorra recursos y acelera el entrenamiento x5–x10.

#### 🔥 Entrenamos solo:
- **mask_decoder** → responsable de generar máscaras finales  
- **parte del prompt encoder** (opcional según implementación)

Además:

- **Learning rate** bajo (1e-4) para evitar desestabilizar el decoder.
- **Optimizer:** Adam
- **Scheduler:** StepLR con decay cada 5 epochs

Este setup es estándar para adaptar SAM a dominios especializados.

---

### 🎛️ 3.5 — Training Loop

El training loop implementa:

- Forward por imagen individual (SAM no soporta prompts en batch).
- Cálculo de embeddings congelados.
- Procesamiento de point/box prompts.
- Forward del decoder para predecir la máscara.
- Redimensionamiento a 256×256 (resolución interna del decoder).
- Backpropagation solo sobre parámetros entrenables.
- Cálculo de IoU por muestra para monitoreo.

Cada época registra:

- Training loss  
- Validation loss  
- Training IoU  
- Validation IoU  

Se guarda automáticamente el **best model** según IoU de validación.

---

### 📈 3.6 — Resultados del Entrenamiento

El proceso completa:

- Visualización de curvas de pérdida
- Visualización de evolución de IoU
- Selección de mejor checkpoint
- Preparación del modelo fine-tuned para la fase de evaluación

Estas gráficas permiten validar:

- Si hay overfitting  
- Si el decoder realmente aprende mejores máscaras  
- Cuánto mejora respecto al modelo pre-entrenado  

SAM fine-tuned tiende a mejorar especialmente en:

- Bordes del agua  
- Regiones delgadas  
- Eliminación de falsos positivos por reflejos  
- Detección más completa de zonas inundadas  

---

Con esto, el modelo queda listo para pasar a la evaluación formal y comparativa.

### 🧪 Parte 4 — Evaluación del Modelo Fine-Tuned y Comparación

Tras semanas de hype sobre el poder de SAM, en esta sección comprobamos realmente qué tan bien funciona *antes* y *después* del fine-tuning en el dominio de inundaciones.

Esta etapa incluye:

- Evaluación completa en el conjunto de validación  
- Cálculo de métricas clave (IoU, Dice, Precision, Recall)  
- Comparación estadística Pretrained vs Fine-tuned  
- Visualizaciones cualitativas de mejora  
- Análisis de errores (failure cases)

---

### 📥 4.1 — Cargar el Mejor Modelo Fine-Tuned

Finalizado el entrenamiento, se carga automáticamente:

- El modelo SAM original  
- Los pesos del mejor checkpoint del decoder  
- Un predictor propio para el modelo fine-tuned

Esto permite comparar *lado a lado* el desempeño de ambos modelos sin reconstruir la arquitectura manualmente.

---

### 📊 4.2 — Comparación Pretrained vs Fine-Tuned (Métricas Globales)

Se evalúan ambas versiones de SAM sobre todas las imágenes de validación usando **point prompts automáticos**.

Métricas:

- **IoU (Intersection over Union):** qué tan bien coincide la predicción con el ground truth  
- **Dice Coefficient:** similar a IoU, más sensible en clases desbalanceadas  
- **Precision:** falsos positivos  
- **Recall:** falsos negativos  

Finalmente se comparan las distribuciones:

- Histogramas Pretrained vs Fine-Tuned  
- Gráfico de barras con métricas promedio y mejora porcentual  

Este análisis permite entender no solo si el modelo mejora, sino **cuánto** y **dónde**.

---

### 🖼️ 4.3 — Visualización Cualitativa de Diferencias

Más allá de los números, la parte visual es clave en segmentación.

Se muestran para varios ejemplos:

1. Imagen original + punto de prompt  
2. Predicción del modelo pre-entrenado  
3. Predicción del modelo fine-tuned  
4. Overlay de ambos modelos sobre la imagen  
5. Métricas por imagen (IoU y Dice)

Suelen observarse mejoras claras en:

- Bordes del agua  
- Región inundada completa  
- Reducción de falsos positivos por reflejos o montones de nubes  
- Menor ruido cerca de límites con tierra firme  

En muchos casos, el fine-tuned recupera áreas que el SAM base *ni siquiera detectaba*.

---

### 🧯 4.4 — Análisis de Errores (Failure Cases)

Incluso con fine-tuning, hay desafíos particulares del dominio:

- Reflejos del cielo en agua  
- Sombras profundas  
- Aguas turbias que parecen tierra  
- Zonas inundadas extremadamente finas o mezcladas con vegetación  

El análisis detecta:

- Casos donde IoU < 0.3  
- Ancho promedio de la región inundada  
- Relación agua/fondo  
- Visualización de predicción vs ground truth  

Finalmente se cuantifica:

- Cuántos *failure cases* tenía el modelo pretrained  
- Cuántos tiene el modelo fine-tuned  
- Cuántos se redujeron (porcentaje)

Esto permite entender si el modelo está realmente listo para aplicaciones críticas (spoiler: casi nunca lo está sin un dataset más grande).

---

Con esto queda finalizada toda la etapa de evaluación del práctico, dejando una base sólida para construir la sección de cierre y reflexiones.

## 🧠 Reflexión Final

Para cerrar este práctico, se presentan una serie de preguntas clave destinadas a analizar críticamente el desempeño del modelo, entender sus limitaciones y conectar el trabajo práctico con aplicaciones reales en monitoreo de inundaciones.

Este apartado es fundamental en un contexto académico y profesional, ya que transforma un ejercicio técnico en un proceso de razonamiento y toma de decisiones informadas.

---

### 📝 Preguntas de Reflexión

#### **1. ¿Por qué el pretrained SAM puede fallar en detectar agua en imágenes de inundaciones?**

SAM fue entrenado sobre un dataset gigantesco pero *genérico* (SA-1B).  
La variabilidad, reflejos, turbidez, sombras y mezcla con vegetación en imágenes de inundaciones no forman parte sustancial de ese dataset.  
Por lo tanto, el modelo:

- Puede interpretar reflejos de cielo como objetos separados  
- Puede ignorar agua oscura o cubierta por vegetación  
- Puede fallar en regiones delgadas o irregulares  
- No entiende el *contexto semántico* del dominio (qué es inundación)

Esto explica por qué el zero-shot performance es razonable, pero no óptimo.

---

#### **2. ¿Qué componentes de SAM decidiste fine-tunear y por qué?**

Se decidió:

- **Congelar el image encoder**  
  Porque ya captura buenas representaciones visuales generales y entrenarlo demandaría enormes recursos.

- **Entrenar únicamente el mask decoder**  
  Esto adapta la parte que realmente toma decisiones de segmentación.

Congelar reduce riesgos de overfitting y permite adaptar el modelo a un dataset pequeño (~300 imágenes).

---

#### **3. ¿Cómo se comparan point prompts vs box prompts para este caso?**

- **Point prompts**: funcionan bien si el punto cae en el agua, pero pueden segmentar demasiado poco o demasiado.  
- **Box prompts**: tienden a ser más estables y generan máscaras más completas porque delimitan la región objetivo.

En general:

- **Point prompts = sensibilidad alta a la ubicación del punto**  
- **Box prompts = resultados más consistentes**, especialmente cuando el agua tiene bordes complejos.

---

#### **4. ¿Qué mejoras específicas observaste después del fine-tuning?**

El fine-tuning produjo mejoras notables:

- Mayor cobertura del área inundada  
- Mejora significativa en bordes  
- Reducción de falsos positivos por reflejos  
- Mejor discriminación entre agua y tierra oscura  
- IoU y Dice aumentaron en prácticamente todos los casos  
- Mucha reducción de failures (<40% IoU)

El modelo aprende “qué es agua” en este dominio específico.

---

#### **5. ¿Está listo para deployment en un sistema de respuesta a desastres?**

A pesar de la mejora:

**No totalmente.**

Faltan:

- Dataset más amplio y diverso (varios países, estaciones del año, resoluciones satelitales, variabilidad extrema)  
- Integración con post-procesamiento geoespacial  
- Validación operativa  
- Robustez ante imágenes ruidosas, nubes, lluvia, humo  
- Inferencia más rápida (SAM no es ideal para producción)

El modelo es un buen *prototipo*, pero no un sistema productivo final.

---

#### **6. ¿Cómo cambiaría tu approach con 10× más datos? ¿Y con 10× menos?**

**Con 10× más datos (~3000 imágenes):**

- Descongelar parcialmente el image encoder  
- Aumentar epochs  
- Hacer stratified sampling  
- Entrenar con prompts variados (point + box + mask)  
- Posible uso de EfficientSAM para velocidad

**Con 10× menos datos (~30 imágenes):**

- Usar fuertísimo data augmentation  
- Solo fine-tuning del decoder  
- Usar few-shot prompting  
- Congelar completamente todos los encoders  
- Considerar modelos más livianos como MobileSAM

---

#### **7. ¿Qué desafíos presenta la segmentación de inundaciones?**

La inundación es un problema difícil por factores visuales y ambientales:

- Reflejos del cielo → confunden los modelos  
- Sombras de edificios o árboles  
- Aguas turbias que parecen tierra  
- Vegetación flotante  
- Límites difusos y bordes irregulares  
- Iluminación inconsistente  
- Resoluciones variables de cámaras y satélites  
- Regiones muy delgadas o parcialmente ocultas

Esto hace que la segmentación de inundaciones sea un caso ideal para aplicar fine-tuning.

---

Con estas reflexiones se cierra el análisis conceptual del práctico, integrando tanto los resultados técnicos como la comprensión del contexto de aplicación.

## 📸 Evidencias

Debido a la complejidad del práctico (entrenamiento pesado, múltiples visualizaciones, curvas, comparaciones y análisis extensos), se incluye directamente el enlace al notebook completo donde se ejecuta todo el pipeline:

[📘 Ver Notebook en Google Colab](https://colab.research.google.com/drive/15vU8h89sjUe4WEDGC_wM3nTPlEiQThHi?usp=sharing)

Este notebook contiene:

- Descarga y preparación del dataset  
- Inferencia zero-shot con SAM (punto y caja)  
- Métricas completas (IoU, Dice, Precision, Recall)  
- Fine-tuning del mask decoder  
- Curvas de entrenamiento  
- Comparación cuantitativa y cualitativa  
- Análisis de failures  
- Visualizaciones detalladas del modelo antes y después del fine-tuning  

---

## 📚 Referencias

![SAM Paper](https://arxiv.org/abs/2304.02643)  
![Segment Anything - GitHub](https://github.com/facebookresearch/segment-anything)  
![Flood Area Segmentation Dataset](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)  
![Albumentations Documentation](https://albumentations.ai/docs/)  
![PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
![scikit-image Documentation](https://scikit-image.org/docs/stable/)  
![OpenCV Documentation](https://docs.opencv.org/)  
![SAM HQ](https://github.com/SysCV/sam-hq)  
![FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)  

---

## 🏁 Cierre del Práctico

Este práctico permitió explorar:

- **SAM pretrained vs fine-tuned** en un caso real de segmentación de inundaciones  
- **Prompt engineering** aplicado a segmentación (punto, caja)  
- **Métricas robustas** de segmentación  
- **Curvas de entrenamiento y validación**  
- **Análisis profundo de errores**  
- **Impacto del fine-tuning** en performance y robustez  

El resultado es un pipeline profesional y completamente reproducible, aplicable a casos de monitoreo ambiental, emergencias y sistemas de visión geoespacial.

