import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def fetch_stock_data(ticker, period='1y'):
    try:
        data = yf.Ticker(ticker).history(period=period)
        if not data.empty:
            return data['Close']
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return pd.Series()

def prepare_data(data, look_back=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def predict_stock_price(ticker):
    data = fetch_stock_data(ticker)
    if data.empty:
        return {"error": f"No data available for {ticker}"}
    
    X, y, scaler = prepare_data(data)
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_30_days = scaler.transform(data[-30:].values.reshape(-1, 1))
    prediction = model.predict(last_30_days.reshape(1, -1))
    predicted_price = scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
    
    return {
        "ticker": ticker,
        "predicted_price": round(predicted_price, 2),
        "current_price": round(data.iloc[-1], 2),
        "prediction_date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    }

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.json.get('ticker')
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400
    
    result = predict_stock_price(ticker)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)