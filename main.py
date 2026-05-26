import numpy as np

from src.gradient_descent import LinearRegressionGD
from src.visualize import plot_regression, plot_cost
from src.animation import animate_gradient_descent


X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([3, 5, 7, 9, 11, 13], dtype=float)

model = LinearRegressionGD(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X, y)

predictions = model.predict(X)

print(f"Slope (m): {model.m}")
print(f"Intercept (b): {model.b}")

plot_regression(X, y, predictions)
plot_cost(model.cost_history)

animate_gradient_descent(X, y, model)