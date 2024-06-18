import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_augmented = np.c_[np.ones(X.shape[0]), X]  
        self.coefficients = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ y

    def predict(self, X):
        X_augmented = np.c_[np.ones(X.shape[0]), X]  
        return X_augmented @ self.coefficients