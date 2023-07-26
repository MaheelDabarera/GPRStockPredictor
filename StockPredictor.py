import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class StockPredictor:
    def __init__(self, data):
        self.data = data
        self.regressor = self._train_model()

    def _train_model(self):
        X = self.data[['Time']]
        y = self.data['StockPrice']
        kernel = 1.0 * RBF()
        regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        regressor.fit(X, y)
        return regressor

    def predict_stock_price(self, time):
        predicted_price = self.regressor.predict([[time]])[0]
        return predicted_price
