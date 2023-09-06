#Sebastian Burgos Alanís A01746459
#Algoritmo KNN con uso de Framworks
#05/09/23

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

"""Carga , lectura y limpieza de los datos"""
#En comparación de la entrega anterior, cambié el dataset ya que usando el anterior, me daba un acurracy muy bajo
#Investigando, Este dataset es mucho mejor para probar KNN. 
dataset = pd.read_csv("KNNAlgorithmDataset.csv", delimiter=",")
#Se elimina id ya que sus valores no son útiles y la tabla Unnamed: 32 ya que no sé qué datos son. 
dataset = dataset.drop(columns=["id", "Unnamed: 32"], axis=1)
#Se elimian todos los NaN
dataset = dataset.dropna()

"""Asignación de Y y X, se elimina la columna de diagnosis en X ya que es la que vamos a predecir. """
#Se cambia M y B por binario para facilitar la lectura y procesamiento de datos. 
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
X = dataset.drop('diagnosis', axis=1)
y = dataset['diagnosis']

"""Dividir los datos en entrenamiento y prueba"""
#Se utiliza random_state 42 ya que en la ciencia de datos, es un valor que ha demostrado ser más útil que otros
#Se utiliza test_size despues de prueba y error. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
#Se estandarizan los datos ya que para KNN, sí unfluye en el resultado. 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#Se llama el clasificador de neighbors y se utiliza el 7 de forma arbitraria. 
knn = KNeighborsClassifier(n_neighbors=7) 
knn.fit(X_train, y_train)
#Se llama la prediccion de los Y. 
y_pred = knn.predict(X_test)

"""Métricas, matrices y acurracy"""
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("\nMatriz de Confusion:\n", confusion)
print("\nClases:\n", classification_report_str)