from flask import Flask, render_template, request, send_file
from datetime import datetime
import re
import os
import pandas as pd
import ModeloRegresion as modelo 
import LinearRegression
from ModeloClasificacion import SentimentAnalyzer
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuración para carga de archivos
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Inicializar el analizador de sentimientos
analyzer = SentimentAnalyzer()

@app.route("/")
def home():
    return "Hello, Flask!"

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

@app.route("/menu")
def menu():
    return render_template("Menu.html") # menu de navegacion

@app.route("/Pagina")
def pagina():
    return render_template("Pagina.html")

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

#Modelo de Regresión Logística

@app.route('/desercion', methods=['POST', 'GET'])
def desercion():
    if request.method == 'POST':

        datos = {
            'notas': float(request.form.get('notas')),
            'asistencia': float(request.form.get('asistencia')),
            'participacion': int(request.form.get('participacion')),
            'tipo_colegio': 0 if request.form.get('tipo_colegio') == 'Publico' else 1
        }
        
        resultado = modelo.predecir_con_matriz(datos)
        
        return render_template('Desercion.html', 
                             datos=datos,
                             probabilidad=resultado['probabilidad'],
                             resultado=resultado['resultado'],
                             accuracy=resultado['accuracy'],
                             precision=resultado['precision'],
                             recall=resultado['recall'])
    else:
        return render_template('Desercion.html', 
                             datos=None, 
                             probabilidad="0%", 
                             resultado="", 
                             accuracy=0.0, 
                             precision=0.0, 
                             recall=0.0)

@app.route('/clasificador', methods=['GET', 'POST'])
def clasificador():
    active_tab = request.args.get('tab', 'single')
    error = None
    single_result = None
    batch_results = None
    download_link = None

    if request.method == 'POST':
        if active_tab == 'single':
            review_text = request.form.get('review_text')
            if review_text:
                single_result = analyzer.predict(review_text)
        else:
            if 'file' not in request.files:
                error = 'No se seleccionó ningún archivo'
            else:
                file = request.files['file']
                if file.filename == '':
                    error = 'No se seleccionó ningún archivo'
                else:
                    try:
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)
                        
                        if filename.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(filepath)
                        else:
                            df = pd.read_csv(filepath)
                        
                        results = []
                        for review in df['review']:
                            result = analyzer.predict(review)
                            results.append({
                                'review': review,
                                'prediction': result['prediction'],
                                'num_pos_words': result['features'][0],
                                'num_neg_words': result['features'][1],
                                'length_review': result['features'][2]
                            })
                        
                        batch_results = results
                        
                        results_df = pd.DataFrame(results)
                        output_filename = f"results_{filename}"
                        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                        results_df.to_excel(output_path, index=False)
                        download_link = output_filename
                        
                    except Exception as e:
                        error = f"Error al procesar el archivo: {str(e)}"

    return render_template('ModeloClasificacion.html',
                         active_tab=active_tab,
                         error=error,
                         single_result=single_result,
                         batch_results=batch_results,
                         download_link=download_link)

@app.route('/uploads/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                    as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
