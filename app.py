from flask import Flask, render_template, request
from datetime import datetime
import re 
import os
import LinearRegression

app = Flask(__name__)

MENU_HTML = """
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
        
    </style>
</head>
<body>
    <div class="contenedor">
        <h1>Menú Principal</h1>
        <a href="/pagina" class="opcion">Página HTML</a>
        <a href="/LinearRegression" class="opcion">Regresión Lineal</a>
        <a href="/RegresionLogistica" class="opcion">Regresión Logística</a>
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return MENU_HTML

@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()

    match_object = re.match("[a-zA-Z]+", name)

    if match_object:
        clean_name = match_object.group(0)
    else:
        clean_name = "Friend"

    content = "Hello there, " + clean_name + "! Hour: " + str(now)
    return content

@app.route("/pagina")
def pagina():
    return render_template("pagina.html")

@app.route("/LinearRegression", methods= ["GET", "POST"])
def inicio():
    prediccion = None
    if request.method == "POST":
     try:
        edad = float(request.form["edad"])
        prediccion = LinearRegression.calcularCosto(edad)
     except ValueError:
        prediccion = "Datos no validos"
        
    LinearRegression.generar_grafico()
    return render_template("LinearRegressionGrades.html", result = prediccion)

@app.route('/RegresionLogistica')
def RegresionLogistica():
    return render_template('RegresionLogistica.html')

if __name__ == "__main__":
    app.run(debug=True)
