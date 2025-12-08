---
title: "Fine-tuning de Transformers para Clasificación Ofensiva"
date:
---

# Fine-tuning de Transformers para Clasificación Ofensiva (Tweets financieros)

---

## 📝 Contexto

En este práctico exploré el salto desde los modelos clásicos basados en **TF-IDF + regresión logística** hacia modelos modernos basados en **Transformers**, aplicados al análisis de sentimiento ofensivo en textos cortos.

El dataset elegido proviene de *Hugging Face Datasets* y contiene tweets financieros en inglés etiquetados en tres clases:

- **0 = Bearish** (sentimiento negativo)
- **1 = Bullish** (sentimiento positivo)
- **2 = Neutral**

La tarea consiste en construir un pipeline completo: desde el EDA inicial, pasando por un baseline clásico, hasta el fine-tuning de un Transformer especializado en lenguaje financiero, evaluando mejoras y comparando enfoques.

---

## 🎯 Objetivos

En este práctico busqué:

- Cargar, normalizar y explorar datasets textuales con *datasets*.
- Visualizar patrones mediante n-grams, WordClouds y proyecciones con PCA/UMAP.
- Construir un **baseline clásico** con TF-IDF + Logistic Regression.
- Entrenar un **Transformer** (FinBERT u otros) mediante fine-tuning.
- Comparar métricas entre enfoques tradicionales y modelos modernos.
- Evaluar desbalance de clases y su impacto en la métrica macro-F1.
- Analizar errores y observar comportamientos de los modelos durante el entrenamiento.

## 🚀 Desarrollo

### 🧩 Parte 1 — Carga y Exploración del Dataset

Para esta primera etapa trabajé con el dataset *Twitter Financial News Sentiment*, proveniente de Hugging Face.  
El objetivo fue preparar un dataset homogéneo con columnas **text** y **label**, para facilitar el pipeline posterior.

### 🔍 Carga del dataset

Utilicé `load_dataset()` indicando correctamente la ruta del repositorio:

```python
raw, source_name = load_financial_news()
```

### 📊 EDA Inicial

Luego normalicé las columnas necesarias, ya que distintos datasets pueden llamar al texto de forma distinta (`"text"`, `"tweet"`, `"content"`, etc.).  
Esto permitió dejar un dataframe consistente:

- **text**: contenido del tweet  
- **label**: 0 = Bearish, 1 = Bullish, 2 = Neutral  

Una vez normalizado, realicé:

- Distribución de clases  
- Distribución de longitudes (tokens por tweet)  
- Revisión de posibles outliers  
- Verificación de balance/imbalance  

Los gráficos mostraron que la clase **Neutral** es la más frecuente, lo que implica:

- Utilizar **macro-F1** como métrica principal  
- Tener precaución con modelos sesgados hacia la clase mayoritaria  

El histograma de longitudes mostró que la mayoría de los tweets tienen entre **10 y 25 tokens**, lo que facilita el trabajo del tokenizer sin truncamientos agresivos.

---

### 🧩 N-grams y WordClouds

Luego generé:

- Top **n-grams (1,2)** por clase usando *CountVectorizer*  
- **WordClouds** por clase para visualizar patrones semánticos  

Esto permitió observar términos clave por sentimiento financiero, como:

- **Bearish**: "drop", "loss", "bearish"  
- **Bullish**: "up", "gain", "bullish"  
- **Neutral**: "market", "report", "fed"  

Estas señales resultan útiles para comparar luego el rendimiento del modelo clásico vs. Transformers.

### 🧪 Baseline Clásico: TF-IDF + Logistic Regression

Antes de avanzar a modelos Transformer, construí un **baseline tradicional** para tener un punto de comparación.  
Este enfoque utiliza:

- **TF-IDF** como representación del texto
- **Logistic Regression** como clasificador lineal multiclase

El pipeline completo fue:

1. **Split estratificado** en train/test  
2. **TF-IDF** con n-grams (1,2) y un máximo de vocabulario configurado  
3. Entrenamiento del modelo  
4. Reporte de métricas y matriz de confusión  

Los resultados mostraron que:

- El modelo captura bien patrones de palabras frecuentes   
- Tiende a confundir **Neutral** con **Bullish/Bearish**, especialmente cuando el tweet es ambiguo  
- La métrica **macro-F1** refleja mejor el desempeño real dado el desbalance de clases  

Este baseline sirve como **referencia mínima** para evaluar si el Transformer realmente aporta mejoras significativas.

### 🤖 Fine-tuning con Transformers (Hugging Face)

Luego del baseline, avancé al enfoque moderno: **fine-tuning de un modelo Transformer preentrenado**, específicamente modelos orientados al dominio financiero como *FinBERT*, y alternativas genéricas como *RoBERTa* o *BERT base*.

#### 🔧 Preparación del dataset

El dataset se dividió utilizando `train_test_split` con **estratificación**, garantizando que las proporciones de cada clase se mantuvieran iguales en train y test.  
Posteriormente convertí los splits a formato **HuggingFace Dataset**, renombrando la columna `label` a `labels` (requerida por Transformers).

Además establecí:

- `num_labels = 3` (Bearish, Bullish, Neutral)
- Casting explícito a `ClassLabel` para evitar problemas durante el entrenamiento

#### 🧰 Tokenización

Se utilizó el tokenizer del checkpoint elegido.  
Claves importantes:

- `truncation=True` para limitar la longitud del input  
- `padding=True` para permitir batching eficiente  
- Soporte a BPE, lo que maneja palabras raras, símbolos financieros, hashtags y emojis  

También inspeccioné manualmente cómo tokeniza frases ofensivas o ambiguas, para confirmar que el modelo interpreta adecuadamente términos clave.

#### 🏋️ Entrenamiento

El entrenamiento se realizó mediante la clase **Trainer**, definiendo:

- `learning_rate`: típicamente 2e-5  
- `batch_size`: entre 8 y 16 dependiendo de VRAM  
- `num_train_epochs`: 3-5  
- `metric_for_best_model="f1"` para priorizar macro-F1  
- `load_best_model_at_end=True`

La función `compute_metrics` devuelve:

- Accuracy
- Macro-F1 (crítico debido al desbalance)

#### 📈 Resultados iniciales

Comparado con el baseline TF-IDF:

- El Transformer mejoró significativamente la **macro-F1**, especialmente en clases minoritarias.  
- Capturó dependencias semánticas y contextuales imposibles para el modelo lineal.  
- Mostró sobreajuste moderado, pero aceptable dada la calidad del dataset.

Estos resultados justifican el uso de Transformers para clasificación ofensiva o de sentimiento en dominios especializados como el financiero.

### 📊 Visualización de Métricas y Comparación Final

Con el modelo Transformer ya entrenado, generé las curvas de validación por época usando los logs del `Trainer`.  
Estas visualizaciones permiten entender:

- Cómo evoluciona la **accuracy**
- Cómo mejora (o empeora) la **macro-F1**
- Si existe overfitting después de cierto número de épocas

Las curvas mostraron un comportamiento estable:  
incremento progresivo hasta estabilizarse alrededor de la época 3–4, lo cual confirma que el número de épocas elegido es adecuado.

#### 🥊 Comparación: Baseline TF-IDF + LR vs Transformer

Evalué ambos modelos sobre el mismo test set:

- **Baseline (TF-IDF + Logistic Regression):** buen rendimiento en la clase mayoritaria, pero pobre en clases minoritarias.
- **Transformer:**  
  - Mejoró sustancialmente la macro-F1  
  - Redujo errores sistemáticos (especialmente confundir Bearish ↔ Neutral)  
  - Capturó matices contextuales como sarcasmos, expresiones idiomáticas y lenguaje financiero

Ejemplo de métricas finales:

- **Baseline:**  
  - Accuracy ≈ más alta por bias hacia neutro  
  - Macro-F1 ≈ más baja por mal rendimiento en clases minoritarias  

- **Transformer:**  
  - Accuracy mejor o similar  
  - Macro-F1 **muy superior**, demostrando real capacidad de clasificación equilibrada

#### 🧠 Interpretación

La diferencia principal radica en que:

- El modelo clásico solo mira frecuencia de palabras (BoW).
- El Transformer entiende **semántica contextual**:
  - relaciones entre tokens  
  - sentimiento implícito  
  - patrones estilísticos  
  - interacciones entre términos financieros  

Esto vuelve al Transformer la opción natural para producción siempre que se disponga de GPU y se necesite robustez frente a ruido del lenguaje real.

### 🧾 Conclusiones Finales

Este práctico permitió comparar dos enfoques muy distintos para la clasificación de sentimiento/ofensividad en texto:

1. **Modelos clásicos (TF-IDF + LR)**
   - Rápidos, interpretables y eficientes.
   - Limitados para capturar contexto.
   - Tienden a favorecer clases mayoritarias.
   - Útiles como baseline para entender el dataset.

2. **Transformers (FinBERT / RoBERTa / BERT)**
   - Comprenden semántica contextual y relaciones entre tokens.
   - Mejoran especialmente la **macro-F1**, clave en datasets desbalanceados.
   - Requieren mayor poder de cómputo, pero ofrecen una mejora clara.

En este caso, el Transformer superó ampliamente al baseline, especialmente en las clases *Bearish* y *Bullish*, donde aparecían más errores del modelo clásico.

El análisis exploratorio previo (EDA, n-grams, WordClouds, PCA/UMAP, etc.) también ayudó a revelar la estructura del dataset, informando decisiones como el tamaño de secuencia, la métrica principal y el tipo de modelo a utilizar.

---

### 📸 Evidencias

[📘 Enlace al Notebook de Google Colab](https://colab.research.google.com/drive/1f-TcN0g_moCMas3dhlh1mWzFM31N5tYN?usp=sharing)

Incluye:
- Carga y normalización del dataset  
- EDA completo  
- Baseline TF-IDF + LR  
- Fine-tuning del Transformer  
- Curvas de entrenamiento y comparación de métricas  

---

### 🤔 Reflexiones

**1. ¿Qué desafíos presenta este dataset?**  
Tweets cortos, lenguaje ruidoso, jerga financiera, emojis y sarcasmos. Esto afecta el rendimiento del baseline, pero es manejado mejor por Transformers.

**2. ¿Por qué usar macro-F1 en vez de accuracy?**  
Porque la clase *Neutral* domina el dataset, y accuracy puede ocultar fallos graves en clases minoritarias.

**3. ¿Qué modelo elegiría para producción?**  
El Transformer, siempre que haya GPU disponible. Ofrece mejores predicciones equilibradas y maneja mejor lenguaje ambiguo.

**4. ¿Qué mejoraría como siguiente paso?**  
- Más limpieza y normalización del texto  
- Técnicas de data augmentation para texto  
- Incluir embeddings financieros específicos (Word2Vec entrenado en corpus financiero)  
- Probar modelos multilingües para tweets mezclados EN/ES  

**5. ¿Qué aprendí?**  
Que el análisis exploratorio es clave antes de entrenar cualquier modelo, y que los Transformers son claramente superiores cuando el contexto semántico importa.

---
