import tkinter as tk
from tkinter import ttk
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set tight_layout to True to avoid the warning
sns.set(rc={'figure.autolayout': True})

# Load the dataset
file_path = r"C:\Users\Surya Krishna\Downloads\IndianWeatherRepository (1).csv" 
weather_data = pd.read_csv(file_path)

# Selecting features and target variable
features = ['temperature_celsius', 'wind_kph', 'humidity', 'country', 'region', 'location_name']
target = 'condition_text'

# Handling missing values by dropping them
processed_data = weather_data[features + [target]].dropna()

# Splitting the dataset into features (X) and target variable (y)
X = processed_data[features]
y = processed_data[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: OneHotEncoder for categorical features and StandardScaler for numerical features
categorical_features = ['country', 'region', 'location_name']
numerical_features = ['temperature_celsius', 'wind_kph', 'humidity']

# Creating a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create a pipeline that first preprocesses the data and then applies the SVC
svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', SVC(kernel='linear'))])  # Use linear kernel for hyperplane visualization

# Training the SVM model
svm_pipeline.fit(X_train, y_train)

# Classification metrics
y_pred = svm_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

# Function to display performance metrics in the GUI
def display_metrics():
    metrics_label.config(text=f"Classification Metrics:\n"
                              f"Accuracy: {accuracy:.2f}\n"
                              f"Precision: {precision:.2f}\n"
                              f"Recall: {recall:.2f}\n"
                              f"F1 Score : {f1:.2f}")

# Function to get real-time weather data
def get_real_time_weather():
    city = city_entry.get()
    api_key = '91380130e946e8f179d8d7c22b7791ea'  # Replace with your OpenWeatherMap API key
    response = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=imperial&APPID={api_key}")
    
    if response.status_code == 200:
        weather_data = response.json()
        
        if 'weather' in weather_data and len(weather_data['weather']) > 0:
            weather = weather_data['weather'][0]['main']
            temp = weather_data['main']['temp']
            temp_c = (temp - 32) * 5 / 9
            description = weather_data['weather'][0]['description']
            feels_like = weather_data['main']['feels_like']
            feels_like_c = (feels_like - 32) * 5 / 9
            humidity = weather_data['main']['humidity']
            
            weather_info_label.config(text=f"The weather in {city} is: {weather}\n"
                                           f"The temperature in {city} is: {temp}°F/{temp_c:.2f}°C \n"
                                           f"Description: {description}\n"
                                           f"The weather in {city} feels like: {feels_like}°F/{feels_like_c:.2f}°C\n"
                                           f"Humidity: {humidity}\n")
        else:
            weather_info_label.config(text='Weather data not found for this location.')
    else:
        weather_info_label.config(text='City Not Found!')

# Function to create an SVM hyperplane plot
def plot_svm_hyperplane():
    plt.figure(figsize=(8, 6))

    # Plot the data points
    plt.scatter(X_train['temperature_celsius'], X_train['humidity'], c=y_train, cmap='viridis', marker='o', s=50)
    plt.xlabel('Temperature (Celsius)')
    plt.ylabel('Humidity')
    
    # Plot the SVM decision function (hyperplane)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Create grid to evaluate model
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = svm_pipeline.decision_function(xy).reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    plt.title('SVM Hyperplane')
    plt.show()

# Function to predict weather and display model performance metrics in a bar chart
def predict_weather_and_plot():
    # Retrieving values from the GUI
    country = country_entry.get()
    region = region_entry.get()
    location = location_entry.get()
    temperature = float(temperature_entry.get())
    wind_speed = float(wind_speed_entry.get())
    humidity = float(humidity_entry.get())
    
    # Creating a DataFrame for the input data
    input_data = pd.DataFrame([[temperature, wind_speed, humidity, country, region, location]], 
                              columns=features)

    # Predicting the weather condition
    predicted_condition = svm_pipeline.predict(input_data)[0]
    
    # Displaying the prediction
    prediction_label.config(text=f"Predicted Weather Condition: {predicted_condition}")
    
    # Calculate model performance metrics
    y_pred = svm_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    # Create a bar chart for model performance metrics
    plt.figure(figsize=(8, 6))
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]
    bar_positions = range(len(metrics_labels))
    bar_width = 0.5
    plt.bar(bar_positions, metrics_values, bar_width, color='skyblue', alpha=0.7)
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Model Performance Metrics')
    plt.xticks(bar_positions, metrics_labels)
    plt.show()

# Setting up the main window
root = tk.Tk()
root.title("Weather Prediction System")

# Adding real-time weather section to the GUI
ttk.Label(root, text="Enter City for Real-Time Weather:").grid(column=0, row=0, padx=10, pady=10)
city_entry = ttk.Entry(root)
city_entry.grid(column=1, row=0, padx=10, pady=10)

real_time_weather_button = ttk.Button(root, text="Get Real-Time Weather", command=get_real_time_weather)
real_time_weather_button.grid(column=0, row=1, columnspan=2, pady=10)

weather_info_label = ttk.Label(root, text="Real-Time Weather Info: ")
weather_info_label.grid(column=0, row=2, columnspan=2)

# Creating input fields for the SVM prediction features
ttk.Label(root, text="Country:").grid(column=0, row=3, padx=10, pady=10)
country_entry = ttk.Entry(root)
country_entry.grid(column=1, row=3, padx=10, pady=10)

ttk.Label(root, text="Region:").grid(column=0, row=4, padx=10, pady=10)
region_entry = ttk.Entry(root)
region_entry.grid(column=1, row=4, padx=10, pady=10)

ttk.Label(root, text="Location Name:").grid(column=0, row=5, padx=10, pady=10)
location_entry = ttk.Entry(root)
location_entry.grid(column=1, row=5, padx=10, pady=10)

ttk.Label(root, text="Temperature (Celsius):").grid(column=0, row=6, padx=10, pady=10)
temperature_entry = ttk.Entry(root)
temperature_entry.grid(column=1, row=6, padx=10, pady=10)

ttk.Label(root, text="Wind Speed (kph):").grid(column=0, row=7, padx=10, pady=10)
wind_speed_entry = ttk.Entry(root)
wind_speed_entry.grid(column=1, row=7, padx=10, pady=10)

ttk.Label(root, text="Humidity (%):").grid(column=0, row=8, padx=10, pady=10)
humidity_entry = ttk.Entry(root)
humidity_entry.grid(column=1, row=8, padx=10, pady=10)

# Button to trigger SVM model prediction and plot
predict_button = ttk.Button(root, text="Predict Weather", command=predict_weather_and_plot)
predict_button.grid(column=0, row=9, columnspan=2, pady=10)

# Label to display the SVM prediction
prediction_label = ttk.Label(root, text="Predicted Weather Condition: ")
prediction_label.grid(column=0, row=10, columnspan=2)

# Button to display model metrics
metrics_button = ttk.Button(root, text="Display Model Metrics", command=display_metrics)
metrics_button.grid(column=0, row=11, columnspan=2, pady=10)

metrics_label = ttk.Label(root, text="Model Performance Metrics")
metrics_label.grid(column=0, row=12, columnspan=2)

# Button to plot SVM hyperplane
hyperplane_button = ttk.Button(root, text="Plot SVM Hyperplane", command=plot_svm_hyperplane)
hyperplane_button.grid(column=0, row=13, columnspan=2, pady=10)

# Running the GUI
root.mainloop()