import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/promoters (1).csv'
dataset = pd.read_csv(file_path)

# Encode the Class labels: + as 1 and - as 0
dataset['Class'] = dataset['Class'].map({'+': 1, '-': 0})

# Split the data into training and testing sets
X = dataset['Sequence']
y = dataset['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to extract k-mers
def get_kmers(sequence, size=4):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Apply the k-mer extraction
X_train_kmers = X_train.apply(lambda x: ' '.join(get_kmers(x)))
X_test_kmers = X_test.apply(lambda x: ' '.join(get_kmers(x)))

# Vectorize the sequences using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train_kmers)
X_test_vectorized = vectorizer.transform(X_test_kmers)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vectorized, y_train)

# Predict on the test set with Random Forest
y_pred_rf = rf_model.predict(X_test_vectorized)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vectorized, y_train)

# Predict on the test set with SVM
y_pred_svm = svm_model.predict(X_test_vectorized)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

# Save the models and vectorizer
joblib.dump(rf_model, '/Users/muthusupriya/Documents/Hareni/sem4/BIO/random_forest_model.joblib')
joblib.dump(svm_model, '/Users/muthusupriya/Documents/Hareni/sem4/BIO/svm_model.joblib')
joblib.dump(vectorizer, '/Users/muthusupriya/Documents/Hareni/sem4/BIO/vectorizer.joblib')

# Output the results
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"Random Forest Classification Report:\n{report_rf}")

print(f"SVM Accuracy: {accuracy_svm}")
print(f"SVM Classification Report:\n{report_svm}")
