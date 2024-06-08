import joblib
import tkinter as tk
from tkinter import messagebox

# Define the function to extract k-mers
def get_kmers(sequence, size=4):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Load the model and vectorizer
model_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/CODE/logisticregression.joblib'
vectorizer_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/CODE/vectorizer.joblib'

loaded_model = joblib.load(model_path)
loaded_vectorizer = joblib.load(vectorizer_path)

# Function to predict if a sequence is a promoter region or not
def predict_promoter(sequence):
    sequence_kmers = ' '.join(get_kmers(sequence))
    sequence_vectorized = loaded_vectorizer.transform([sequence_kmers])
    prediction = loaded_model.predict(sequence_vectorized)
    return prediction[0]

# Function to handle prediction and display result
def handle_prediction():
    user_sequence = entry_sequence.get()
    predicted_class = predict_promoter(user_sequence)
    result = "Promoter Region" if predicted_class == 1 else "Non-Promoter Region"
    messagebox.showinfo("Prediction Result", f"The entered sequence is a: {result}")

# Create the main window
root = tk.Tk()
root.title("Promoter Prediction")

# Add a label and entry for the DNA sequence
label_sequence = tk.Label(root, text="Enter DNA Sequence:")
label_sequence.pack(pady=10)

entry_sequence = tk.Entry(root, width=50)
entry_sequence.pack(pady=10)

# Add a button to perform the prediction
button_predict = tk.Button(root, text="Predict", command=handle_prediction)
button_predict.pack(pady=10)

# Run the application
root.mainloop()
