from flask import Flask, render_template, request, jsonify
import pandas as pd
from StockPredictor import StockPredictor

app = Flask(__name__)

# Sample stock data (replace this with your real data)
# Assuming a single feature (e.g., time) and a target variable (e.g., stock price)
data = pd.DataFrame({
    'Time': [1, 2, 3, 4, 5],
    'StockPrice': [100, 120, 150, 130, 140]
})

predictor = StockPredictor(data)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        time = float(request.form['time'])
        predicted_price = predictor.predict_stock_price(time)
        return render_template('index.html', time=time, predicted_price=predicted_price)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        time = float(data['time'])
        predicted_price = predictor.predict_stock_price(time)
        return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
