# Clasificador de spam con Machine Learning

Proyecto final de la asignatura de Aprendizaje Automático (Grado en Ingeniería Matemática, Universidad CEU San Pablo, mayo de 2026). Proyecto en equipo.

## Objetivo

Construir un clasificador de correos spam sobre el dataset **Spambase** (UCI) y elegir el modelo más adecuado para un filtro de correo real. En este problema, un **falso positivo** (un correo legítimo marcado como spam) es más grave que un falso negativo, así que el modelo final se elige priorizando la **precisión**, no solo la accuracy.

## Dataset

- **Spambase**, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/94/spambase
- 4.601 correos y 57 variables: frecuencia de palabras, frecuencia de caracteres especiales y longitud de las secuencias de mayúsculas.
- Clasificación binaria: `1` = spam, `0` = no spam.
- Se incluye una copia en `dataset.csv`.

## Pipeline

1. **Análisis exploratorio:** valores nulos, tipos de datos, distribuciones y correlaciones con la variable objetivo.
2. **Outliers:** se detectan con el método IQR, pero no se eliminan. En datos de frecuencias, los valores extremos contienen información útil.
3. **Transformación logarítmica** `log(1+x)` para corregir el fuerte sesgo de las variables, seguida de **estandarización**.
4. **Dos enfoques de preparación de variables:**
   - Selección de variables por correlación de Pearson con la variable objetivo.
   - Reducción de dimensionalidad con **PCA** (48 componentes que conservan el 95 % de la varianza).
5. **Modelos entrenados en ambos enfoques:** regresión logística (umbral 0,5 y 0,7), árbol de decisión, KNN, SVM (kernel lineal y RBF), red neuronal MLP y Naive Bayes.
6. **Evaluación:** accuracy, precisión, recall, F1-score, matriz de confusión y validación cruzada estratificada (5 folds).

## Resultados principales

Conjunto de test: 1.381 correos.

| Modelo | Enfoque | Accuracy | Precisión | Recall | Falsos positivos |
|---|---|---|---|---|---|
| **Regresión logística (umbral 0,7)** | Correlación | 0,916 | **0,961** | 0,819 | **18** |
| SVM (RBF) | Correlación | 0,934 | 0,936 | 0,893 | 33 |
| Red neuronal MLP (mejorada) | PCA | **0,946** | 0,934 | 0,930 | 36 |

- La **mejor accuracy** (94,6 %) la obtiene la red neuronal MLP con PCA.
- La **mayor precisión** y el menor número de falsos positivos los obtiene la regresión logística con el umbral de decisión subido a 0,7.
- La validación cruzada confirma que la regresión logística con umbral 0,7 es estable: precisión media de 0,951 entre folds.

## Modelo final

**Regresión logística con umbral de decisión 0,7.** Subir el umbral de 0,5 a 0,7 obliga al modelo a exigir más evidencia antes de marcar un correo como spam. Pierde algo de recall, pero reduce los falsos positivos a la mitad respecto a los modelos con mejor accuracy. Además, es un modelo sencillo, interpretable y barato de poner en producción.

PCA mantiene resultados muy competitivos y mejora la accuracy de la red neuronal, pero no mejora la precisión de la regresión logística, que es la métrica prioritaria en este problema.

## Cómo ejecutarlo

```bash
git clone https://github.com/fernandoviolero/clasificador-spam-ml.git
cd clasificador-spam-ml
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo jupyter
jupyter notebook ProyectoFinal_definitivo.ipynb
```

## Estructura

```
├── ProyectoFinal_definitivo.ipynb   # Notebook con todo el análisis
├── dataset.csv                      # Copia del dataset Spambase
└── README.md
```

## Autores

- Hugo Fernández Cavero
- Mario Gutiérrez Morano
- Víctor Pérez Dalmases
- Diego Ruiz Jover
- Fernando Violero Muñoz
