import numpy as np
import pandas as pd
import matplotlib.pyplot
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from sklearn.linear_model import LinearRegression

datos = {
      "edad": [18,20,22,25,30,35,40,45,50,55,60,65,70],
      "Costo_seguro": [1500, 1400, 1300, 1200, 1100, 1000, 950, 900, 850, 800, 780, 770, 760]
}

df = pd.DataFrame(datos)

x = np.array(datos["edad"]).reshape(-1,1)
y = np.array(datos["Costo_seguro"])

model = LinearRegression()
model.fit(x,y)

def calcularCosto(edad):
 return round(model.predict([[edad]])[0], 2)

def generar_grafico():
  if not os.pathexists("static"):
      os.makedirs("static")

      plt.figure(figsize=(8,5))
      sns.scatterplot(x=datos["edad"], y=datos["Costo_seguro"], label= "Datos", color="red")
      plt.plot(datos["edad"], model.predict(x), color="blue", linewidth=2, label="LineadeRegresion")
      plt.xlabel("Edad del conductor")
      plt.ylabel("Costo del Seguro")
      plt.title("Regresion Lineal del costo del seguro segun la edad del conductor")
      plt.legend()
      plt.grid(True)


      ruta_imagen = os.path.join("static", "regresion_lineal.png")
      plt.savefig(ruta_imagen)
      plt.close()
