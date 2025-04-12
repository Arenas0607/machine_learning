import matplotlib.pyplot as plt
import numpy as np


edades = np.array([18, 25, 30, 35, 40, 45, 50, 60])
costos = np.array([2000, 2200, 2500, 2700, 3000, 3300, 3600, 4000])


coef = np.polyfit(edades, costos, 1)
modelo = np.poly1d(coef)

def calcularCosto(edad):
    
    costo_estimado = modelo(edad)
    return round(costo_estimado, 2)

def generar_grafico():
    
    x = np.linspace(15, 65, 100)
    y = modelo(x)

    plt.figure(figsize=(10, 6))
    plt.scatter(edades, costos, color='blue', label='Datos Reales')
    plt.plot(x, y, color='red', label='Modelo de Regresión')
    plt.title('Regresión Lineal: Edad vs Costo del Seguro')
    plt.xlabel('Edad')
    plt.ylabel('Costo del Seguro')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plt.savefig("static/regresion_lineal.png")
    plt.close()




