---
title: "Mall Customer Segmentation – Clustering y PCA"
date:
---

# Mall Customer Segmentation – Clustering y PCA

---

## 📝 Contexto

En este práctico trabajé con el **Mall Customer Segmentation Dataset**, un conjunto de datos clásico para **segmentación de clientes**.  
El objetivo general fue identificar grupos de consumidores con patrones similares de:

- Demografía (edad, género)
- Capacidad adquisitiva (ingreso anual)
- Comportamiento de compra (*Spending Score*)

La idea es acercarse a un caso real de marketing en retail, donde el centro comercial quiere:

- Definir **segmentos claros de clientes**
- Ajustar campañas y promociones
- Optimizar la inversión publicitaria
- Detectar perfiles de alto valor vs. bajo engagement

---

## 🎯 Objetivos

En este trabajo busqué:

- Explorar el dataset y comprender sus variables clave.
- Preparar los datos para clustering (codificación, limpieza, escalado).
- Comparar distintos métodos de **normalización**: MinMax, Standard y Robust.
- Aplicar **PCA** para reducir dimensionalidad y facilitar la visualización.
- Evaluar distintas estrategias de **Feature Selection** frente a PCA.
- Entrenar modelos de clustering (principalmente **K-Means**) y elegir el número óptimo de clusters.
- Analizar los segmentos encontrados desde la perspectiva de negocio.

---

## 🚀 Desarrollo

### 🔍 Parte 1 — Exploración inicial del dataset

Comencé cargando el **Mall Customer Segmentation Dataset**, verificando su estructura y variables principales:

- *CustomerID*
- *Gender*
- *Age*
- *Annual Income (k$)*
- *Spending Score (1–100)*

Este dataset es limpio y no presenta valores faltantes, lo que facilita directamente el análisis.

#### 📊 Insights preliminares

Durante la exploración inicial observé:

- Distribución relativamente equilibrada entre géneros.
- Rango amplio de edad (18–70 años).
- Clientes con ingresos muy variados.
- Spending Score que no presenta correlación lineal directa con el ingreso, lo que sugiere comportamientos diferenciados.

Estos primeros hallazgos justifican el uso de **clustering**, ya que podrían existir grupos ocultos con patrones de compra similares.

---

### 🧹 Parte 2 — Preparación y escalado de los datos

Probé distintos métodos de normalización para evaluar su impacto en el clustering:

- **StandardScaler** → ajusta según media y desviación estándar  
- **MinMaxScaler** → lleva todo a un rango entre [0, 1]  
- **RobustScaler** → estable frente a *outliers*

Cada versión del dataset escalado se comparó posteriormente en el proceso de clustering para observar:

- cambios en la separación de los grupos,
- impacto en la forma de las nubes de puntos,
- sensibilidad del algoritmo a la escala de las variables.

#### ✨ Observación clave
El *Spending Score* y el *Annual Income* responden mejor al escalado MinMax y Standard, mientras que la edad presenta más variabilidad, lo que hace útil también RobustScaler en algunos escenarios.

---

### 🎨 Parte 3 — PCA para visualización y reducción de dimensionalidad

Apliqué **Análisis de Componentes Principales (PCA)** para:

- reducir la dimensionalidad del dataset,
- visualizar grupos potenciales en 2D,
- medir cuánto varía la información al proyectarla.

#### 📌 Resultados de PCA

- Los primeros **2 componentes explican la mayor parte de la varianza**, facilitando la visualización.
- PCA reveló estructuras claras entre clientes con:
  - alto ingreso y alto *Spending Score*,
  - bajo ingreso y bajo *Spending Score*,
  - combinaciones intermedias.

Este paso también sirvió para comparar PCA vs. Feature Selection tradicional.

---

### 🧩 Parte 4 — Feature Selection y comparación con PCA

Probé seleccionar distintas combinaciones de variables:

- Solo *Income* + *Spending Score*
- Agregar *Age*
- Incluir *Gender* codificado

Comparé estos escenarios con la proyección obtenida por PCA.

#### 📝 Conclusiones de esta parte

- PCA logra separar mejor los grupos que cualquier selección manual de variables.
- Sin embargo, la combinación **Income + Spending Score** sigue siendo altamente informativa por sí misma.
- Agregar *Age* mejora ligeramente algunos límites entre clusters, pero no siempre aporta separación fuerte.

---

### ⚙️ Parte 5 — Aplicación de K-Means y búsqueda del número óptimo de clusters

El siguiente paso fue aplicar **K-Means**, el algoritmo de clustering más utilizado para segmentación de clientes.  
Antes de elegir un número fijo de clusters, utilicé varios métodos para determinar el valor óptimo de *k*.

---

### 🔢 5.1 — Método del Codo (Elbow Method)

Probé valores de *k* entre 2 y 10 y analicé cómo disminuía la **inercia** (suma de distancias intra-cluster).  
El “codo” apareció alrededor de **k = 5**, lo que sugiere que:

- agregar más clusters después de 5 no reduce significativamente la inercia,
- el dataset probablemente contiene 5 grupos bien diferenciados.

---

### 📈 5.2 — Silhouette Score

También calculé el **Silhouette Score** para medir qué tan separados y compactos son los clusters.

- Los puntajes más altos ocurrieron alrededor de **k = 4** y **k = 5**.  
- k=5 mostró una separación más equilibrada entre todos los clusters.

Esto reforzó la elección de **5 clusters** como punto óptimo.

---

### 🎯 5.3 — K-Means final y análisis de los clusters

Entrené el modelo final con **k = 5**, usando las mejores variables seleccionadas y los datos escalados.  
Luego analicé cada cluster en función de:

- ingreso anual,
- spending score,
- edad,
- género.

#### 🧩 Descripción general de los clusters encontrados

Los patrones típicos fueron:

1. **Clientes jóvenes con alto Spending Score y alto ingreso**  
   Segmento premium, ideal para marketing agresivo.

2. **Clientes jóvenes con bajo Spending Score pero ingreso medio**  
   Pueden activarse con promociones específicas.

3. **Clientes mayores con Spending Score bajo**  
   Grupo estable pero con baja rentabilidad.

4. **Ingresos altos pero Spending Score variable**  
   Segmento con alto potencial de fidelización.

5. **Ingresos bajos y Spending Score alto**  
   Consumidores sensibles a precio, pero muy activos.

---

### 🎨 Visualización final de clusters

Usé PCA para proyectar los clusters en 2D, permitiendo ver claramente:

- grupos bien definidos,
- fronteras razonablemente separadas,
- patrones consistentes con las variables originales.

Esto confirmó que **K-Means fue una buena elección para este dataset**.

---

### 📝 Conclusión parcial

- El dataset presenta naturalmente **5 segmentos de clientes**.  
- PCA ayudó a visualizar y validar la estructura real de los grupos.  
- Los clusters obtenidos pueden usarse directamente para estrategias comerciales, segmentación personalizada y programas de fidelización.

---

## 📊 Parte 6 — Evaluación de los clusters y análisis de perfiles de cliente

Una vez obtenido el modelo final con **k = 5**, realicé un análisis profundo de cada segmento para entender **qué tipo de clientes representa cada cluster**, cómo se diferencian entre sí y qué oportunidades ofrece cada grupo desde la perspectiva de negocio.

---

### 🧬 6.1 — Perfiles detallados por cluster

Para cada cluster analicé:

- **Edad promedio**
- **Ingreso anual promedio**
- **Spending Score**
- **Distribución por género**

Este análisis permitió interpretar los grupos no solo desde lo cuantitativo, sino también desde comportamientos de consumo.

#### 🏷️ Insights principales de los perfiles:

- Hay clusters con **alto gasto y alto ingreso** → clientes VIP.
- Otros tienen **ingreso alto pero bajo gasto** → oportunidad clara para marketing de reactivación.
- Algunos grupos presentan **mayor edad y bajo gasto** → segmentos estables pero poco rentables.
- Los clientes jóvenes muestran **gran variabilidad**, desde compradores impulsivos hasta compradores conservadores.

Este tipo de análisis es exactamente lo que un equipo de marketing utilizaría para **crear campañas diferenciadas y optimizar inversión publicitaria**.

---

### 📈 6.2 — Visualizaciones de soporte

Para entender mejor cómo se estructuran los segmentos:

- Usé la proyección **PCA 2D** para visualizar cómo se distribuyen los clusters en el espacio.
- Generé **gráficos de barras** mostrando promedios de edad, ingreso y spending score por cluster.
- Analicé el tamaño relativo de cada cluster para entender qué grupos dominan el mercado.

**Conclusión visual:** Los clusters están claramente diferenciados, lo que valida que K-Means fue apropiado.

---

### 🧪 6.3 — Silhouette Score por cluster

Más allá del score general, examiné el **silhouette por cluster**, lo que reveló:

- Clusters con cohesión muy alta → clientes con patrones muy homogéneos.
- Clusters con menor cohesión → zonas de frontera más difusas, típicas en datasets reales.
- Pocos valores negativos, indicando **mínimas asignaciones incorrectas**.

Este análisis ayuda a identificar qué segmentos están más “bien definidos” desde la perspectiva algorítmica.

---

### 🚨 6.4 — Detección de outliers

Utilizando silhouette sample-by-sample pude encontrar:

- Algunos clientes con silhouette < 0 → posibles outliers o compradores atípicos.
- Esto puede deberse a:
  - comportamiento de compra inconsistente,
  - valores extremos en ingreso o gasto,
  - combinación demográfica poco frecuente.

Los outliers no fueron numerosos, lo que sugiere una segmentación estable.

---

### 🧩 6.5 — Interpretación final desde negocio

El análisis de clusters permitió construir **perfiles accionables**:

- Segmentos VIP donde conviene invertir en fidelización.
- Segmentos de bajo gasto donde promociones pueden aumentar engagement.
- Segmentos jóvenes de alto gasto → ideales para marketing digital.
- Segmentos de bajo ingreso pero alto gasto → sensibles a precio, oportunidad para bundles.

La segmentación ofrece una **visión clara y operativa** del comportamiento del cliente dentro del mall.

---

### 📝 Conclusión parcial

La fase de evaluación confirmó que:

- Los **5 clusters** son consistentes, interpretables y útiles.
- Los perfiles obtenidos muestran diferencias reales y accionables.
- La combinación de K-Means + PCA + análisis cuantitativo permite una segmentación robusta.
- El dataset refleja patrones reales del retail: diversidad de ingresos, edades y hábitos de gasto.

---

## 🧠 Reflexión Final

Este práctico fue uno de los más completos de la unidad, integrando **todo el pipeline de CRISP-DM** aplicado a segmentación de clientes: exploración, preparación, normalización, clustering, PCA y evaluación. A partir del trabajo realizado, destaco los siguientes aprendizajes y conclusiones.

---

### 🔍 1. Sobre la metodología CRISP-DM  
- La fase más desafiante fue **Data Preparation**, especialmente decidir cómo normalizar y qué features dejar para clustering.  
- El **Business Understanding** influyó directamente en la selección de variables: no todas las que aparecen en el dataset son útiles para segmentación real.  
- La estructura de CRISP-DM permitió mantener un flujo ordenado en un análisis largo y técnico.

---

### 🧹 2. Data Preparation  
- Entre MinMax, Standard y Robust, el ganador se determinó empíricamente usando silhouette score:  
  **el mejor scaler fue el que generó clusters más cohesivos**.  
- PCA resultó una herramienta clave para:
  - visualizar estructuras naturales del dataset,  
  - reducir ruido,  
  - verificar si realmente existían grupos distinguibles.  
- Feature Selection mostró que muchas veces **no hace falta usar todas las features** para obtener buenos clusters.

---

### 🤖 3. Clustering  
- El Elbow Method y Silhouette no siempre coinciden; por eso fue importante complementar con **criterios de negocio** (en retail se suelen esperar 3–5 segmentos).  
- K-Means funcionó muy bien porque los grupos eran relativamente esféricos y con separación moderada.  
- Los perfiles obtenidos fueron **interpretable y consistentes**, lo cual es esencial en segmentación comercial.

---

### 💼 4. Aplicación real al negocio  
El resultado final permite que un equipo de marketing pueda:

- diseñar campañas personalizadas,  
- identificar segmentos de alto valor,  
- detectar clientes con gasto bajo pero alto ingreso (oportunidad clara),  
- optimizar presupuestos de marketing según comportamiento real.

La segmentación es aplicable en contextos como centros comerciales, e-commerce y programas de fidelidad.

---

### 🚀 5. Qué mejoraría con más tiempo  
- Probar clustering alternativo para ver si captura estructuras no lineales (DBSCAN, GMM, HDBSCAN).  
- Afinar PCA y Feature Selection para comparar explicabilidad vs rendimiento.  
- Añadir más variables relevantes: visitas al mall, frecuencia de compra, ticket promedio.  
- Evaluar temporalidad → segmentación dinámica (clientes que “migran” de un cluster a otro).

---

## 📸 Evidencias

[📘 Enlace al Notebook de Google Colab](https://colab.research.google.com/drive/10cXEmzRFMoaXwrZp9vZhRBgZocld-GQy?usp=sharing)

---

## ⚡ Comentarios sobre los Challenges (versión breve)

Para no extender el portafolio innecesariamente, dejo solo un resumen general:

- **DBSCAN**: útil para detectar outliers y clusters de densidad irregular; en este dataset tiende a marcar ruido.  
- **HDBSCAN**: más estable que DBSCAN; forma clusters jerárquicos y detecta patrones más complejos.  
- **GMM**: permite clusters elípticos; interesante alternativa cuando K-Means es demasiado rígido.  
- **Spectral Clustering**: ideal para estructuras no lineales; requiere buen ajuste de afinidad.  
- **Agglomerative**: da una visión jerárquica; útil para interpretar relaciones entre segmentos.  
- **RFE y SFS (Forward/Backward)**: permiten comprender qué features realmente aportan al clustering.  
- **t-SNE / UMAP**: visualizaciones avanzadas → revelan estructuras que PCA no siempre capta.

En general, los challenges enriquecen el análisis y permiten validar que el método elegido (K-Means + escalado adecuado) sí es razonable para este caso.

---

## 📚 Referencias

![Pandas Documentation](https://pandas.pydata.org/docs/)
![NumPy Documentation](https://numpy.org/doc/)
![Matplotlib Documentation](https://matplotlib.org/stable/)
![Seaborn Documentation](https://seaborn.pydata.org/)
![Scikit-Learn Documentation](https://scikit-learn.org/stable/)
![OneHotEncoder – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
![K-Means – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
![Silhouette Score – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
![SequentialFeatureSelector – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html)
![PCA – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
![DBSCAN – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
![HDBSCAN Documentation](https://hdbscan.readthedocs.io/en/latest/)
![Gaussian Mixture Models – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)
![Spectral Clustering – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)
![Agglomerative Clustering – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
![RFE – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
![Sklearn Datasets](https://scikit-learn.org/stable/datasets/)
![t-SNE – Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
![UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/)


