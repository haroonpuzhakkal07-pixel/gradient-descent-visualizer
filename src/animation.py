import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_gradient_descent(X, y, model):

    fig, ax = plt.subplots()

    ax.scatter(X, y, color='blue')

    line, = ax.plot([], [], color='red')

    ax.set_xlim(min(X) - 1, max(X) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)

    def update(frame):

        m = model.m_history[frame]
        b = model.b_history[frame]

        y_pred = m * X + b

        line.set_data(X, y_pred)

        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(model.m_history),
        interval=60,
        blit=True
    )

    plt.title("Gradient Descent Optimization")
    plt.show()