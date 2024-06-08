import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/DATASET/combined_features_with_dna_shapes_imputed.csv'
final_dataset = pd.read_csv(file_path)

# Separate features and target variable
X = final_dataset.drop('Class', axis=1)
y = final_dataset['Class']

# Handle Missing Values using SimpleImputer (if necessary)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Reshape data for LSTM input
X_lstm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Define an LSTM model with L2 regularization and BatchNormalization
def create_lstm_model(input_shape, num_classes):
    model = Sequential([
        Bidirectional(LSTM(32, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.4),
        
        Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.01))),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0001)  # Use Adam optimizer with a learning rate of 0.0001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (X_lstm.shape[1], 1)
num_classes = y_categorical.shape[1]
lstm_model = create_lstm_model(input_shape, num_classes)

# Add callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

# Train the model for 500 epochs
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_categorical, test_size=0.2, random_state=42)
lstm_model.fit(X_train, y_train, epochs=500, batch_size=64, 
               validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

# Evaluate the model on the test set
lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Test Accuracy: {lstm_accuracy}")

# Predict the classes for the test set
y_pred = lstm_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Print the classification report
print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

# Print the confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Save the LSTM model in the native Keras format
lstm_model.save('/Users/muthusupriya/Documents/Hareni/sem4/BIO/lstm_model_updated.keras')