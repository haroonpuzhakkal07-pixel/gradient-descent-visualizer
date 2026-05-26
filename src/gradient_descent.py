import numpy as np


class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0
        self.b = 0

        self.cost_history = []
        self.m_history = []
        self.b_history = []

    def compute_cost(self, X, y):
        n = len(y)
        y_pred = self.m * X + self.b
        cost = (1 / n) * np.sum((y - y_pred) ** 2)
        return cost

    def fit(self, X, y):
        n = len(y)

        for _ in range(self.epochs):

            y_pred = self.m * X + self.b

            dm = (-2 / n) * np.sum(X * (y - y_pred))
            db = (-2 / n) * np.sum(y - y_pred)

            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

            cost = self.compute_cost(X, y)

            self.cost_history.append(cost)
            self.m_history.append(self.m)
            self.b_history.append(self.b)

    def predict(self, X):
        return self.m * X + self.b