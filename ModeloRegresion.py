import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Variables globales para caching
_df = None
_modelo = None
_X_train, _X_test, _y_train, _y_test = None, None, None, None

MODEL_PATH = 'models/modelo_desercion.pkl'

def cargar_datos():
    global _df
    if _df is None:
        try:
            _df = pd.read_csv('dataseed_desercion.csv')
            _df['tipo_colegio'] = _df['tipo_colegio'].map({'Publico': 0, 'Privado': 1})
            _df['desercion'] = (_df['notas'] < 50).astype(int)
        except FileNotFoundError:
            print("Error: El archivo 'dataseed_desercion.csv' no se encontró.")
            return None
    return _df

def inicializar_modelo():
    global _modelo, _X_train, _X_test, _y_train, _y_test

    if os.path.exists(MODEL_PATH):
        return cargar_modelo()

    df = cargar_datos()
    if df is None:
        return False

    X = df[['notas', 'asistencia', 'participacion', 'tipo_colegio']]
    y = df['desercion']

    _X_train, _X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    _modelo = LogisticRegression(max_iter=1000)
    _modelo.fit(_X_train, _y_train)

    os.makedirs('models', exist_ok=True)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(_modelo, f)

    return True

def cargar_modelo():
    global _modelo, _X_train, _X_test, _y_train, _y_test
    if _modelo is None:
        try:
            with open(MODEL_PATH, 'rb') as f:
                _modelo = pickle.load(f)
            df = cargar_datos()
            if df is not None:
                X = df[['notas', 'asistencia', 'participacion', 'tipo_colegio']]
                y = df['desercion']
                _X_train, _X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        except FileNotFoundError:
            return inicializar_modelo()
    return True

def predecir_con_matriz(datos_usuario):
    if not cargar_modelo():
        return None

    X_usuario = pd.DataFrame([datos_usuario])
    probabilidad = _modelo.predict_proba(X_usuario)[:, 1][0]
    resultado = 'Riesgo' if probabilidad > 0.5 else 'No Riesgo'

    y_pred = _modelo.predict(_X_test)

    return {
        'resultado': resultado,
        'probabilidad': f"{probabilidad:.2%}",
        'accuracy': accuracy_score(_y_test, y_pred),
        'precision': precision_score(_y_test, y_pred),
        'recall': recall_score(_y_test, y_pred)
    }

def generar_matriz_confusion():
    if not cargar_modelo():
        return False

    y_pred = _modelo.predict(_X_test)
    matriz = confusion_matrix(_y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Riesgo", "Riesgo"], 
                yticklabels=["No Riesgo", "Riesgo"])
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    os.makedirs("static", exist_ok=True)
    plt.savefig("static/matriz_confusion.png")
    plt.close()
    return True

def entrenar_modelo():
    if inicializar_modelo():
        generar_matriz_confusion()
