import os  # Import the os module
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Bidirectional, Dense, Dropout, Input

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

# Load GloVe embeddings
embeddings_index = {}
glove_dir = '/Users/muthusupriya/Downloads/glove.6B'
with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Prepare embedding matrix
embedding_dim = 100
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Improved LSTM-GRU Model with Pretrained Embeddings and Attention
def attention_mechanism(inputs):
    hidden_size = inputs.shape[2]
    score_first_part = tf.keras.layers.Dense(hidden_size, use_bias=False, name='attention_score_vec')(inputs)
    h_t = tf.keras.layers.Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(inputs)
    score = tf.keras.layers.Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
    attention_weights = tf.keras.layers.Activation('softmax', name='attention_weight')(score)
    context_vector = tf.keras.layers.Dot(axes=[1, 1], name='context_vector')([inputs, attention_weights])
    pre_activation = tf.keras.layers.Concatenate(name='attention_output')([context_vector, h_t])
    attention_vector = tf.keras.layers.Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector

inputs = Input(shape=(maxlen,))
embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False)(inputs)
lstm_out = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
gru_out = Bidirectional(GRU(128, return_sequences=True))(lstm_out)
attention_out = attention_mechanism(gru_out)
dropout = Dropout(0.5)(attention_out)
dense1 = Dense(256, activation='relu')(dropout)
dropout1 = Dropout(0.5)(dense1)
outputs = Dense(1, activation='sigmoid')(dropout1)

improved_model = Model(inputs=[inputs], outputs=[outputs])

improved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
improved_model.fit(X_train_padded, y_train, epochs=100, batch_size=32, validation_data=(X_test_padded, y_test))

# Evaluate the Improved Model
improved_loss, improved_accuracy = improved_model.evaluate(X_test_padded, y_test)
print(f"Improved Model Accuracy with LSTM-GRU and Attention: {improved_accuracy}")

# Save the Improved Model
improved_model.save('/Users/muthusupriya/Documents/Hareni/sem4/BIO/improved_lstm_gru_model_with_attention.h5')
