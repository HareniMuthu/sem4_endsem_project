# eval.py
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Load the dataset
data = pd.read_csv(r'C:\h_dataset\features_with_labels.csv')

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# List of models to evaluate
models = [
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\adaboost.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\bagging.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\catboost_model.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\decision_tree.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\gradient_boosting.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\knn.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\logistic_regression.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\naive_bayes.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\neural_net.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\perceptron.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\random_forest.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\svm.pkl',
    r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\xgboost_model.pkl'
    ]

# Dictionary to hold performance metrics
performance_metrics = {}

# Evaluate each model
for model_filename in models:
    model = joblib.load(model_filename)
    y_pred = model.predict(X_scaled)
    
    # Ensure predictions are encoded
    if isinstance(y_pred[0], str):
        y_pred = label_encoder.transform(y_pred)
    
    accuracy = accuracy_score(y_encoded, y_pred)
    conf_matrix = confusion_matrix(y_encoded, y_pred)
    class_report = classification_report(y_encoded, y_pred, output_dict=True)
    performance_metrics[model_filename] = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
    print("{} Performance Metrics:".format(model_filename.split('\\')[-1]))
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_report(y_encoded, y_pred))
    print("\n")

# Evaluate deep learning model separately
deep_learning_model = load_model(r'D:\B-TECH\SEM_4\h\ml\fnal\code\model\trained_models\deep_learning_model.h5')
y_pred = np.argmax(deep_learning_model.predict(X_scaled), axis=1)
accuracy = accuracy_score(y_encoded, y_pred)
conf_matrix = confusion_matrix(y_encoded, y_pred)
class_report = classification_report(y_encoded, y_pred, output_dict=True)
performance_metrics['deep_learning_model.h5'] = {
    'accuracy': accuracy,
    'confusion_matrix': conf_matrix,
    'classification_report': class_report
}
print("Deep Learning Performance Metrics:")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_encoded, y_pred))
print("\n")

# Create a summary DataFrame
summary_df = pd.DataFrame({
    'Model': [k.split('\\')[-1].split('.')[0] for k in performance_metrics.keys()],
    'Accuracy': [v['accuracy'] for v in performance_metrics.values()]
})

# Print the summary table
print("Summary of Performance Metrics:")
print(summary_df)

# Determine and print the best classifier
best_classifier = summary_df.loc[summary_df['Accuracy'].idxmax()]
print("\nBest Classifier: {} with Accuracy: {}".format(best_classifier['Model'], best_classifier['Accuracy']))
