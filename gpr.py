import yfinance as yf
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

# Get the stock data
msft = yf.Ticker("MSFT")
stock_data = msft.history(period="max")

# Define the training data and target variable
X = np.array(range(len(stock_data)))
y = np.array(stock_data['Close'])

# Define the kernel function
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# Train the GPR model
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
gpr.fit(X.reshape(-1, 1), y)

# Predict the future stock prices
X_pred = np.array(range(len(stock_data), len(stock_data) + 30))
y_pred, sigma = gpr.predict(X_pred.reshape(-1, 1), return_std=True)

# Plot the predicted prices and the confidence interval
plt.plot(stock_data['Close'], label='Historical Data')
plt.plot(X_pred, y_pred, label='Predicted Price')
plt.fill_between(X_pred, y_pred - 2 * sigma, y_pred + 2 * sigma, alpha=0.2)
plt.xlabel('Days since IPO')
plt.ylabel('Price')
plt.title('Microsoft Stock Price Prediction with GPR')
plt.legend()
plt.show()
