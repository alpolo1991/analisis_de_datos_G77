# Machine Learning: Regresión Lineal, Regresión Logística y Árboles de Decisión

## Descripción

Este proyecto explora tres algoritmos fundamentales de aprendizaje supervisado: **Regresión Lineal**, **Regresión Logística**, y **Árboles de Decisión**. Estos algoritmos se utilizan para resolver problemas tanto de regresión como de clasificación. A continuación, se describe cada uno de estos algoritmos, su aplicación, y el proceso general para implementarlos usando Python y bibliotecas como `scikit-learn`.

## Contenidos

1. [Regresión Lineal](#regresion-lineal)
2. [Regresión Logística](#regresion-logistica)
3. [Árboles de Decisión](#arboles-de-decision)
4. [Requisitos](#requisitos)
5. [Instalación y Uso](#instalacion-y-uso)
6. [Referencias](#referencias)

---

## Regresión Lineal

### Descripción

La **Regresión Lineal** es un algoritmo de aprendizaje supervisado utilizado para predecir un valor continuo. El modelo establece una relación lineal entre una variable dependiente (Y) y una o más variables independientes (X). Se utiliza para problemas de regresión donde se busca predecir un valor numérico.

### Ejemplo de Aplicación

Predecir el precio de una vivienda basado en características como tamaño, número de habitaciones, y antigüedad.

### Fórmula Matemática

En su forma más simple, la regresión lineal sigue la fórmula:

\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n
\]

Donde:
- \(Y\) es la variable dependiente (resultado o predicción).
- \(X_1, X_2, ..., X_n\) son las variables independientes.
- \(\beta_0, \beta_1, ..., \beta_n\) son los coeficientes de regresión.

### Implementación en Python

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Crear el modelo de regresión lineal
model = LinearRegression()

# Dividir los datos en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)
```

## Regresión Logística

### Descripción

La **Regresión Logística** es un algoritmo utilizado para la clasificación binaria (sí/no, 0/1). Aunque se llama regresión, se usa para predecir una categoría, y no un valor continuo. El modelo estima la probabilidad de que un dato pertenezca a una clase determinada utilizando la función sigmoide.

### Ejemplo de Aplicación

Clasificar si un cliente comprará un producto o no, basado en características como edad, ingresos, y comportamiento de compra.

### Fórmula Matemática

La regresión logística se basa en la función sigmoide:

Donde: 

- \P(Y=1|Y\) es la probabilidad de que 𝑌 sea 1.
- \(X_1, X_2, ..., X_n\) son las variables independientes.
- \(\beta_0, \beta_1, ..., \beta_n\) son los coeficientes de regresión.

### Implementación en Python

```python
from sklearn.linear_model import LogisticRegression

# Crear el modelo de regresión logística
model = LogisticRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)
```

## Árboles de Decisión

### Descripción
Los **Árboles de Decisión** son algoritmos que pueden ser usados para problemas tanto de clasificación como de regresión. Un árbol de decisión divide los datos en subconjuntos basándose en preguntas binarias (verdadero/falso) sobre las características. Se utiliza tanto para predecir categorías como valores numéricos.

### Ejemplo de Aplicación

Clasificar si un correo es spam o no, basado en palabras clave y otras características.

### Funcionamiento

El árbol crea particiones sucesivas de los datos en función de una característica que maximiza la "pureza" de las particiones (en términos de clasificación) o minimiza el error (en términos de regresión).

### Implementación en Python

```python
from sklearn.tree import DecisionTreeClassifier

# Crear el modelo de árbol de decisión
model = DecisionTreeClassifier()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir
y_pred = model.predict(X_test)
```

### Visualización del Árbol

```python
from sklearn import tree
import matplotlib.pyplot as plt

# Visualizar el árbol de decisión
plt.figure(figsize=(20,10))
tree.plot_tree(model, filled=True)
plt.show()
```

## Requisitos

- Python 3.x
- Bibliotecas necesarias:
  - scikit-learn
  - pandas
  - matplotlib
  - numpy
  - seaborn

 ## Instalación de las bibliotecas necesarias

 ```python
pip install scikit-learn pandas matplotlib numpy seaborn
```

## Instalación y Uso

1. Clona este repositorio:
 ```python
git clone [Ánalisis de datos](https://github.com/alpolo1991/analisis_de_datos_G77.git)
```

2. Instala los requisitos:
 ```python
pip install -r requirements.txt
```

3. Ejecuta el script:
```python
python file_script.py
```

## Referencias

1. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
2. [Regresión Lineal en Wikipedia](https://es.wikipedia.org/wiki/Regresi%C3%B3n_lineal)
3. [Regresión Logística en Wikipedia](https://es.wikipedia.org/wiki/Regresi%C3%B3n_log%C3%ADstica)
4. [Árboles de Decisión en Wikipedia ](https://es.wikipedia.org/wiki/%C3%81rbol_de_decisi%C3%B3n)
