import matplotlib.pyplot as plt


def plot_regression(X, y, predictions):
    plt.scatter(X, y, color="blue", label="Data Points")
    plt.plot(X, predictions, color="red", label="Regression Line")

    plt.title("Linear Regression using Gradient Descent")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

    plt.show()


def plot_cost(cost_history):
    plt.plot(cost_history)

    plt.title("Cost Function during Gradient Descent")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")

    plt.show()