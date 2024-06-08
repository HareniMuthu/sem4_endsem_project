import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load the dataset
file_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/promoters (1).csv'
dataset = pd.read_csv(file_path)

# Encode the Class labels: + as 1 and - as 0
dataset['Class'] = dataset['Class'].map({'+': 1, '-': 0})

# Split the data into training and testing sets
X = dataset['Sequence']
y = dataset['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the sequences
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad the sequences
maxlen = max([len(seq) for seq in X_train_seq])
X_train_padded = pad_sequences(X_train_seq, maxlen=maxlen, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=maxlen, padding='post')

# Convert labels to numpy array
y_train = np.array(y_train)
y_test = np.array(y_test)

# CNN Model
cnn_model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=maxlen),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(5),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_padded, y_train, epochs=10, validation_data=(X_test_padded, y_test))

# Evaluate the CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_padded, y_test)
print(f"CNN Accuracy: {cnn_accuracy}")

# Save the CNN model
cnn_model.save('/Users/muthusupriya/Documents/Hareni/sem4/BIO/cnn_model.h5')
print(f"CNN model saved to /mnt/data/cnn_model.h5")
