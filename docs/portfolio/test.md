---
title: "Práctica 1: EDA del Titanic en Google Colab"
date: 2025-08-12
---

# Práctica 1: EDA del Titanic en Google Colab

---

## 📝 Contexto

> **Nota:**  
    En esta ocasión como primer acercamiento al trabajo con ML, rama de la IA vamos a trabajar con el dataset del [Titanic](https://www.kaggle.com/competitions/titanic/data), de esta forma a través de la práctica comenzamos a ponernos manos a la obra para explorar este mundo en auge dentro de la informática.

---

## 🎯 Objetivos

Explorar, preparar y utilizar herramientas clave para el aprendizaje durante el curso:

- [Google Colab](https://colab.google/)
- [Kaggle](https://www.kaggle.com/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Numpy](https://numpy.org/doc/stable/)
- [Matplotlib](https://matplotlib.org/stable/users/index)
- [Seaborn](https://seaborn.pydata.org/tutorial.html)

---

## ⏱️ Actividades y Tiempos Estimados

| Actividad                      | Tiempo      |
|--------------------------------|:----------:|
| **Tarea 1:** Setup en Colab    | 5 min      |
| **Tarea 2:** Cargar dataset    | 5-10 min   |
| **Tarea 3:** Conocer dataset   | 10 min     |
| **Tarea 4:** EDA visual        | 15 min     |
| **Tarea 5:** Preguntas finales | —          |

---

## 🚀 Desarrollo

### Tarea 1 ✅ - Setup en Colab

> Leí la [documentación](https://colab.research.google.com/#scrollTo=vwnNlNIEwoZ8) para entender Google Colab.  
> Es un notebook digital que se ejecuta en la nube, aprovechando el hardware de Google (GPU/TPU) y facilitando la colaboración y el acceso desde cualquier lugar.

---

### Tarea 2 ✅ - Cargar el dataset de Kaggle

> Kaggle es una plataforma y comunidad para aprender y practicar Data Science.  
> Usamos datasets públicos para entrenar modelos, analizar datos y buscar patrones.

---

### Tarea 3 ✅ - Conocer el dataset

> Kaggle detalla los datos del [dataset](https://www.kaggle.com/competitions/titanic/data).  
> Probé funciones de Pandas para explorar el DataFrame y aprendí sobre sus utilidades:

```python
train.shape  # Tupla con filas y columnas
train.columns  # Nombres de columnas
train.head(3)  # Primeras 3 filas
train.info()  # Resumen del dataset
train.describe(include='all').T  # Estadísticas descriptivas
train.isna().sum().sort_values(ascending=False)  # Valores nulos ordenados
train['Survived'].value_counts(normalize=True)  # Proporción de supervivientes
```

> Pandas permite manipular, filtrar y analizar datos.  
> Matplotlib y Seaborn ayudan a visualizar los resultados.

---

### Tarea 4 ✅ - EDA visual con Seaborn/Matplotlib

> **¿Qué es EDA?**  
> El análisis exploratorio de datos (EDA) usa estadísticas y visualizaciones para descubrir patrones y preparar los datos para análisis más profundos.

> Seaborn (sobre Matplotlib) facilita la creación de gráficos atractivos y el análisis visual.

---

## 📸 Evidencias

- Capturas de gráficos y outputs
- Enlace al notebook: `p1_eda_titanic.ipynb`
- Resultados obtenidos en la exploración

---

## 💡 Reflexión

> **Reflexión personal:**  
> - Qué aprendiste  
> - Qué mejorarías  
> - Próximos pasos  
>
> *(Completa tu reflexión aquí)*

---

## 📚 Referencias

- [Google Colab Docs](https://colab.research.google.com/#scrollTo=vwnNlNIEwoZ8)
- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/frame.html)
- [Matplotlib Docs](https://matplotlib.org/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)