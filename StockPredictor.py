import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta

# This function fetches stock data for all S&P 500 stocks from Yahoo Finance.
def fetch_sp500_stock_data():
    sp500_tickers = yf.Tickers('^GSPC').tickers
    stock_data = {}
    for ticker in sp500_tickers:
        data = ticker.history(period='20y', interval='1d')
        stock_data[ticker.ticker] = data
    return stock_data

# This function creates a simple RNN model.
def create_rnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(1, 1)),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='rmsprop', loss='mse')
    return model

# This function predicts stock prices for tomorrow using a trained RNN model.
def predict_stock_prices(time, model, stock_data):
    predicted_prices = {}
   
    for ticker, data in stock_data.items():
        data = data.drop(['Date', 'Dividends', 'Stock Splits'], axis=1)
        data = data.dropna()
        data = data.values.reshape(-1, 1)

        price = model.predict(np.array([[time]]))[0][0]
        predicted_prices[ticker] = price

    return predicted_prices

# This is the main function of the program. It first fetches stock data, then predicts stock prices for tomorrow, and finally prints the predicted prices.
def main():

    # Disable eager execution for TensorFlow to avoid immediate evaluation of operations.
    tf.compat.v1.disable_eager_execution()

    # Get the current date.
    current_date = datetime.now().date()

    # Predict stock prices for tomorrow.
    tomorrow = current_date + timedelta(days=1)

    # Fetch stock data for all S&P 500 stocks.
    stock_data = fetch_sp500_stock_data()

    # Create a RNN model.
    model = create_rnn_model()

    # Placeholder: Train your RNN model using the stock data

    # Predict stock prices for all stocks in the S&P 500 index.
    predicted_prices = predict_stock_prices(tomorrow, model, stock_data)

    # Print the predicted stock prices for all stocks.
    for ticker, price in predicted_prices.items():
        print(f"Predicted price for {ticker} on {tomorrow}: ${price:.2f}")

if __name__ == '__main__':
    main()
