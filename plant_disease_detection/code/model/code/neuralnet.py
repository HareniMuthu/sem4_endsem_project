# neural_net.py
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings

# Ignore convergence warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Neural Network Classifier
clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(30, 20, 15, 10), learning_rate='constant', solver='lbfgs', max_iter=1000)
clf.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(clf, 'neural_net.pkl')

# Print performance metrics
y_pred = clf.predict(X_test_scaled)
print("Neural Net Performance Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
