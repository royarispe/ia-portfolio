---
title: "Práctica 1: EDA del Titanic en Google Colab"
date: 2025-08-12
---

## Contexto

!!! note "Contexto"
    En esta ocasión como primer acercamiento al trabajo con ML, rama de la IA vamos a trabajar con el dataset del [Titanic](https://www.kaggle.com/competitions/titanic/data).  
    A través de la práctica comenzamos a ponernos manos a la obra para explorar este mundo en auge dentro de la informática.

---

## Objetivos

Los objetivos para este primer práctico consisten en todo lo que es explorar, preparar y utilizar herramientas que nos ayudarán en el aprendizaje a lo largo del curso:

- [GoogleColab](https://colab.google/)
- [Kaggle](https://www.kaggle.com/)
- [Pandas](https://pandas.pydata.org/docs/)
- [Numpy](https://numpy.org/doc/stable/)
- [Matplotlib](https://matplotlib.org/stable/users/index)
- [Seaborn](https://seaborn.pydata.org/tutorial.html)

---

## Actividades (con tiempos estimados)

| Actividad                           | Tiempo   |
|------------------------------------|:--------:|
| Tarea 1 - Setup en Colab           | 5 min    |
| Tarea 2 - Cargar el dataset        | 5-10 min |
| Tarea 3 - Conocer el dataset       | 10 min   |
| Tarea 4 - EDA visual               | 15 min   |
| Tarea 5 - Preguntas finales        | —        |

---

## Desarrollo

???+ info "Ver desarrollo paso a paso"
    - **Tarea 1 [x] - Setup en Colab**  
      *(explicación completa que ya escribiste)*

    - **Tarea 2 [x] - Cargar el dataset de Kaggle**  
      *(tu texto sobre qué es un dataset, Kaggle y la plataforma)*

    - **Tarea 3 [x] - Conocer el dataset**  
      *(tu texto sobre atributos, pandas, funciones y ejemplos)*  

      ```python linenums="1"
      train.shape
      train.columns
      train.head(3)
      train.info()
      train.describe(include='all').T
      train.isna().sum().sort_values(ascending=False)
      train['Survived'].value_counts(normalize=True)
      ```

    - **Tarea 4 [x] - EDA visual con seaborn/matplotlib**  
      *(tu explicación sobre qué es EDA y por qué es útil, con seaborn y matplotlib)*

---

## Evidencias

- Capturas de gráficos y outputs
- Enlace al notebook: [p1_eda_titanic.ipynb](../notebooks/p1_eda_titanic.ipynb)
- Resultados obtenidos en la exploración

---

## Reflexión

!!! tip "Reflexión personal"
    - Qué aprendiste  
    - Qué mejorarías  
    - Próximos pasos  

*(Acá copiás tu reflexión cuando la completes)*

---

## Referencias

- [Google Colab Docs](https://colab.research.google.com/#scrollTo=vwnNlNIEwoZ8)  
- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)  
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/frame.html)  
- [Matplotlib Docs](https://matplotlib.org/)  
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
