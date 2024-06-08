import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
import numpy as np

# Load the dataset
data = pd.read_csv(r'C:\h_dataset\features_with_labels.csv')

# Separate features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create Pool for CatBoost
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# Define CatBoost parameters
params = {
    'iterations': 1000,
    'learning_rate': 0.1,
    'depth': 6,
    'eval_metric': 'MultiClass',
    'random_seed': 42,
    'use_best_model': True,
    'task_type': 'GPU',  # Use GPU
    'verbose': 100  # Show progress every 100 iterations
}

# Train the model with progress bar
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=test_pool)

# Save the model to a file
model_filename = 'catboost_model.pkl'
joblib.dump(model, model_filename)

# Predictions
y_pred = model.predict(X_test)

# Performance metrics
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Correlation Matrix
correlation_matrix = pd.DataFrame(X).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
plt.title("Feature Correlation Matrix")
plt.show()

# Plot learning curves
evals_result = model.evals_result_
iterations = np.arange(len(evals_result['learn']['MultiClass']))

# Plot log loss
fig, ax = plt.subplots()
ax.plot(iterations, evals_result['learn']['MultiClass'], label='Train')
ax.plot(iterations, evals_result['validation']['MultiClass'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('CatBoost Log Loss')
plt.show()

# Plot classification accuracy
fig, ax = plt.subplots()
ax.plot(iterations, evals_result['learn']['MultiClass'], label='Train')
ax.plot(iterations, evals_result['validation']['MultiClass'], label='Test')
ax.legend()
plt.ylabel('Classification Accuracy')
plt.title('CatBoost Classification Accuracy')
plt.show()
