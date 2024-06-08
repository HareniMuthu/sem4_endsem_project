import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
file_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/DATASET/promoters (1).csv'
dataset = pd.read_csv(file_path)

# Verify the specific sequence
sequence_to_check = "tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt"
sequence_info = dataset[dataset['Sequence'].str.contains(sequence_to_check)]
print(sequence_info)

# Encode the Class labels: + as 1 and - as 0
label_encoder = LabelEncoder()
dataset['Class'] = dataset['Class'].map({'+': 1, '-': 0})

# Check the encoding
print(dataset['Class'].value_counts())

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

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

# Save the model and vectorizer
model_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/CODE/logisticregression.joblib'
vectorizer_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/CODE/vectorizer.joblib'

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print(f"Model saved to {model_path}")
print(f"Vectorizer saved to {vectorizer_path}")
