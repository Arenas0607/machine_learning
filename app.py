from flask import Flask, render_template, request
from datetime import datetime
import re 
import pandas as pd
import numpy as np
import joblib 
import LinearRegression
import ModeloRegresion
from RegresionLogistica import entrenar_modelo
from ModeloRegresion import accuracy, precision, recall
import modelos_clasificacion

app = Flask(__name__)

modelo_entrenado = joblib.load('modelo_entrenado.pkl')

MENU_HTML_BASE = """
<!DOCTYPE html>
<html>
<head>
    <title>Menú Principal</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f8ff;
            margin: 0;
            padding: 20px;
        }
        .contenedor {
            max-width: 600px;
            margin: 50px auto;
            text-align: center;
        }
        h1 {
            color: #0066cc;
            margin-bottom: 30px;
        }
        .opcion {
            background-color: #add8e6;
            color: #000080;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            text-decoration: none;
            display: block;
            font-weight: bold;
        }
        .submenu {
            background-color: #d1ecf1;
            padding: 10px;
            margin-top: 20px;
            border-radius: 5px;
        }
        .submenu h2 {
            color: #0c5460;
            margin-bottom: 15px;
        }
        .submenu-item {
            background-color: #bee5eb;
            color: #0c5460;
            padding: 10px;
            margin: 5px 0;
            border-radius: 3px;
            text-decoration: none;
            display: block;
        }
    </style>
</head>
<body>
    <div class="contenedor">
        <h1>Menú Principal</h1>
        <a href="/pagina" class="opcion">Página HTML</a>
        <a href="/LinearRegression" class="opcion">Regresión Lineal</a>
        <a href="/RegresionLogistica" class="opcion">Regresión Logística</a>
        <a href="/predecir" class="opcion">Modelo Regresion</a>
        <a href="/modelos" class="opcion">Modelos de Clasificación Supervisados</a>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return MENU_HTML_BASE 

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    match_object = re.match("[a-zA-Z]+", name)

    clean_name = match_object.group(0) if match_object else "Friend"
    return f"Hello there, {clean_name}! Hour: {now}"

@app.route("/pagina")
def pagina():
    return render_template("pagina.html")

@app.route("/LinearRegression", methods=['GET', 'POST'])
def inicio():
    prediccion = None
    if request.method == "POST":
        try:
            edad = float(request.form["edad"])
            prediccion = LinearRegression.calcularCosto(edad)
        except ValueError:
            prediccion = "Datos no válidos"
        
        LinearRegression.generar_grafico()
    
    return render_template("LinearRegressionGrades.html", result=prediccion)

@app.route('/RegresionLogistica')
def regresion_logistica():
    precision = entrenar_modelo()
    return render_template('RegresionLogistica.html', precision=precision)

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
    if request.method == 'POST':
        try:
           
            if not all(k in request.form for k in ["notas", "asistencia", "participacion", "tipo_colegio"]):
                return "Error: Faltan datos en el formulario.", 400
            
            notas = float(request.form['notas'])
            asistencia = float(request.form['asistencia'])
            participacion = int(request.form['participacion'])
            tipo_colegio = 1 if request.form['tipo_colegio'] == 'Privado' else 0

           
            if not (0 <= notas <= 100) or not (0 <= asistencia <= 100) or not (1 <= participacion <= 10):
                return "Error: Valores fuera de rango.", 400

        
            entrada = np.array([[notas, asistencia, participacion, tipo_colegio]])
            resultado = modelo_entrenado.predict(entrada)[0]
            probabilidad = modelo_entrenado.predict_proba(entrada)[0][1]

          
            ModeloRegresion.generar_matriz()

            return render_template('Desercion.html', 
                                   resultado=resultado, 
                                   probabilidad=round(probabilidad, 2), 
                                   accuracy=round(accuracy, 2), 
                                   precision=round(precision, 2), 
                                   recall=round(recall, 2),
                                   confusion_matrix=True)
        
        except ValueError:
            return render_template('Desercion.html', 
                                   resultado="Error: Entrada no válida", 
                                   probabilidad=0, 
                                   accuracy=round(accuracy, 2), 
                                   precision=round(precision, 2), 
                                   recall=round(recall, 2))
        except Exception as e:
            return render_template('Desercion.html', 
                                   resultado=f"Error inesperado: {str(e)}", 
                                   probabilidad=0, 
                                   accuracy=round(accuracy, 2), 
                                   precision=round(precision, 2), 
                                   recall=round(recall, 2))
    
    ModeloRegresion.generar_matriz()

    return render_template('Desercion.html', 
                           resultado=None, 
                           probabilidad=None, 
                           accuracy=round(accuracy, 2), 
                           precision=round(precision, 2), 
                           recall=round(recall, 2),
                           confusion_matrix=True)

@app.route("/modelos")
def modelos():
    # Asegúrate de que la base de datos esté inicializada
    modelos_clasificacion.init_db()
    
    # Obtener todos los modelos de clasificación de la base de datos
    modelos = modelos_clasificacion.get_all_modelos()
    
    # Renderizar el HTML y pasar los modelos como variable
    return render_template("modelos_supervisados.html", modelos=modelos)



if __name__ == "__main__":
    # Asegurarse de que la base de datos está inicializada antes de arrancar la app
    print("Inicializando base de datos...")
    modelos_clasificacion.init_db()
    print("Base de datos inicializada correctamente")
    
    # Iniciar la aplicación Flask
    app.run(debug=True)
