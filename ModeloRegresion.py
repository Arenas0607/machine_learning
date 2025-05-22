import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


np.random.seed(42)
n_samples = 500

df = pd.DataFrame({
    'notas': np.random.uniform(50, 100, n_samples),
    'asistencia': np.random.uniform(60, 100, n_samples),
    'participacion': np.random.randint(1, 11, n_samples),
    'tipo_colegio': np.random.choice([0, 1], n_samples)
})

df['desercion'] = ((df['notas'] < 65) & (df['asistencia'] < 75) & (df['participacion'] < 4)).astype(int)


X = df[['notas', 'asistencia', 'participacion', 'tipo_colegio']]
y = df['desercion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)


joblib.dump(model, "modelo_entrenado.pkl")


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

def generar_matriz():

 plt.figure(figsize=(5, 4))
 sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Deserta', 'Deserta'], yticklabels=['No Deserta', 'Deserta'])
 plt.xlabel('Predicción')
 plt.ylabel('Real')
 plt.title('Matriz de Confusión')

 ruta_imagen2 = os.path.join("static", "confusion_matrix.png")
 plt.savefig(ruta_imagen2)
 plt.close()
generar_matriz()