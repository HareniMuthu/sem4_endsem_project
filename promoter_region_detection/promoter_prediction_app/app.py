from flask import Flask, render_template, request, current_app
from utils import predict_promoter

app = Flask(__name__)
app.config['MODEL_PATH'] = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/promoter_prediction_app/model/lr'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form['sequence']
    prediction = predict_promoter(sequence)
    result = "Promoter Region" if prediction == 1 else "Non-Promoter Region"
    return render_template('index.html', sequence=sequence, result=result)

if __name__ == '__main__':
    app.run(debug=True)
