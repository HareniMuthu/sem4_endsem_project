import joblib
from flask import current_app
import os

def get_kmers(sequence, size=4):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def predict_promoter(sequence):
    model_folder = current_app.config['MODEL_PATH']
    model_path = os.path.join(model_folder, '/Users/muthusupriya/Documents/Hareni/sem4/BIO/promoter_prediction_app/model/lr/logistic_regression_model.joblib')
    vectorizer_path = os.path.join(model_folder, '/Users/muthusupriya/Documents/Hareni/sem4/BIO/promoter_prediction_app/model/lr/vectorizer.joblib')
    
    loaded_model = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)
    
    sequence_kmers = ' '.join(get_kmers(sequence))
    sequence_vectorized = loaded_vectorizer.transform([sequence_kmers])
    prediction = loaded_model.predict(sequence_vectorized.reshape(1, -1))  # Reshape to 2D array
    return prediction[0]
