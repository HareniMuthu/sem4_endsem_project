import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv('/Users/muthusupriya/Documents/Hareni/sem4/BIO/DATASET/combined_features_with_dna_shapes_imputed.csv')

# Preprocessing
X = data.drop('Class', axis=1)
y = data['Class']

# Convert '+' and '-' in target to numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardizing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to tune
models_params = {
    'Random Forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'SVM': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf']
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs']
        }
    }
}

# Hyperparameter tuning and evaluation results
# Hyperparameter tuning and evaluation results
results = {}
for name, item in models_params.items():
    clf = GridSearchCV(item['model'], item['params'], cv=5, return_train_score=False)
    clf.fit(X_train_scaled, y_train)
    model = clf.best_estimator_
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {
        'best_params': clf.best_params_,
        'accuracy': accuracy,
        'classification_report': report
    }

# Print the results
for model_name, model_results in results.items():
    print(f"Results for {model_name}:")
    print("Best Parameters:", model_results['best_params'])
    print("Accuracy:", model_results['accuracy'])
    print("Classification Report:\n", model_results['classification_report'])
    print("\n")
