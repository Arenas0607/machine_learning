import os 
from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import nltk
from nltk.corpus import opinion_lexicon

nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        # Asegurar que los recursos necesarios estén descargados
        for resource in ['opinion_lexicon', 'punkt']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource)
        nltk.download('punkt_tab')  
        
        self.positive_words = set(opinion_lexicon.positive())
        self.negative_words = set(opinion_lexicon.negative())
        self.model = None

    def load_data(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"El directorio {path} no existe")
        
        data = []
        for label in ['pos', 'neg']:
            dir_path = os.path.join(path, 'txt_sentoken', label)
            for file in os.listdir(dir_path):
                with open(os.path.join(dir_path, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                data.append({'review': text, 'sentiment': 1 if label == 'pos' else 0})
        return pd.DataFrame(data)

    def extract_features(self, text: str) -> List[int]:
        try:
            words = nltk.word_tokenize(text.lower())
            num_pos = len([w for w in words if w in self.positive_words])
            num_neg = len([w for w in words if w in self.negative_words])
            length = len(words)
            return [num_pos, num_neg, length]
        except LookupError as e:
            print(f"Error al procesar el texto: {e}")
            print("Asegúrate de que los recursos de NLTK estén descargados correctamente.")
            return [0, 0, 0]
        except Exception as e:
            print(f"Error inesperado al procesar el texto: {e}")
            return [0, 0, 0]

    def train_model(self, data_path: str):
        try:
            df = self.load_data(data_path)
            features = df['review'].apply(lambda x: pd.Series(self.extract_features(x)))
            features.columns = ['num_pos_words', 'num_neg_words', 'length_review']
            
            X = features
            y = df['sentiment']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            self.model = MultinomialNB()
            self.model.fit(X_train, y_train)
           
            y_pred = self.model.predict(X_test)
            print("Reporte de clasificación:")
            print(classification_report(y_test, y_pred))
            
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, 'movie_review_model.joblib')
            joblib.dump(self.model, model_path)
            print(f"Modelo guardado en: {model_path}")
            
        except Exception as e:
            print(f"Error durante el entrenamiento: {e}")

    def predict(self, text: str) -> Dict:
        if self.model is None:
            try:
                self.model = joblib.load(os.path.join('models', 'movie_review_model.joblib'))
            except FileNotFoundError:
                raise Exception("No se encontró un modelo entrenado. Por favor, entrene el modelo primero.")
        
        features = self.extract_features(text)
        features_df = pd.DataFrame([features], columns=['num_pos_words', 'num_neg_words', 'length_review'])
        prediction = self.model.predict(features_df)[0]
        
        return {
            'prediction': prediction,
            'features': features,
            'text': text
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.train_model('review_polarity')