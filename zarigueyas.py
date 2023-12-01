#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error

#El problema sera que dado un .csv sobre datos de zarigueyas, se creara un arbol
#el cual sea capaz de predecir una edad dado unos datos sobre una zarigueya


# In[81]:


#Uso de pandas para acceder al archivo excel
df = pd.read_csv(r"C:\Users\Usuario\Desktop\Regresion\possum.csv")
df["sex"] = df["sex"].replace({"m":1.0, "f":0.0})
df.head()


# In[82]:


#Resumen del dataFrame,(NroColumnas,Nombre,Cuantos no nulos, tipoDeDato)
df. info()
print("LEER: Observar que en 4 tenemos 102 datos no nulos y en 9 103 datos no nulos")
print("Como no estan completos, depresiaremos esos casos")


# In[83]:


#Eliminacion de filas que contienen valores nulos
df = df.dropna()
#Ejucion de info para demostrar que se eliminaron los nulos
df.info()
print("OBSERVAR que ahora solo tenemos 101 casos")


# In[84]:


#Se eliminara columnas que no son de importancia para el problema
#x tendra una copia de df
#axis = 1 (elimina columnas), axis = 0 (elimina filas)
#             caso    sitio   Pais  Edad
x = df.drop(["case", "site", "Pop", "age"], axis=1)
x.info()
#Sexo, Tamaño de la cabeza, Ancho del craneo, Largo total, Longitud de la cola,
#Largo del pie, Longitud de la oreja, distancia entre extremos de los ojos,
#Pecho y Vientre


# In[85]:


# Guardaremos las edades (df sigue teniendo la tabla original)
y = df["age"]
y.head()


# In[86]:


#Usando la biblioteca sklearn, dividiremos nuestros datos en entrenamiento y prueba
#x = Nuestra tabla de datos
#y = Las respuestas (Años de cada zarigueya)
#test_size = %de datos para hacer pruebas (0.10 = 10%, el 90% restante sera para el entrenamiento)
#random_state = Obtener la misma division de datos en conjuntos de entrenamiento y prueba en distindas ejecuciones
xEntrenamiento, xPruebas, yEntrenamiento, yPruebas = train_test_split(x, y, test_size = 0.10, random_state = 44)


# In[87]:


#Crearemos nuestro arbol de regresion
arbol = DecisionTreeRegressor(random_state = 44, max_depth=5)

#Contruira un arbol de decision que se ajuste a los datos de entrenamiento
#para poder realizar predicciones
arbol.fit(xEntrenamiento, yEntrenamiento)

#Dado que ya contruimos un arbol, haremos las predicciones con nuestro xPruebas
#y los guardaremos en predicciones (Todo lo que tiene xPruebas es nuevo para
# nuestro arbol dado que se entreno con xEntrenanmiento)
ejeX = [1,2,3,4,5,6,7,8,9,10,11]
predicciones = np.round(arbol.predict(xPruebas)).astype(float)
print("Lo que conseguimos")
print(predicciones)
print("Lo que deberias conseguir")
aux = yPruebas.tolist()
print(aux)
plt.plot(ejeX, predicciones,label="Conseguido", color="blue")
plt.plot(ejeX, yPruebas,label="Esperado", color = "red")
plt.xlabel("Iteracion")
plt.xticks(ejeX)
plt.legend()
plt.show()


# In[89]:


#Evaluar el rendimiento del arbol
#Puntuacion de nuestro arbol
#Coeficiente de determinacion
puntuacion = arbol.score(xPruebas, yPruebas)
#Error cuadratico medio
ecm = mean_squared_error(yPruebas, predicciones)

#Grafica del arbol
plt.figure(figsize=(20,10))
plot_tree(arbol, filled = True, feature_names=x.columns.tolist())
plt.show()

#Pruebas 
#aux = np.array([[1.0, 85.1, 51.5, 76.0, 35.5, 70.3, 52.6, 14.4, 23.0, 27.0]])
#aux2 = pd.DataFrame(aux, columns=["sex","hdlngth","skullw","totlngth","taill","footlgth","earconch","eye","chest","belly"])
#imp = np.round(arbol.predict(aux2)).astype(float)
#print(imp)