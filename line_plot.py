import matplotlib.pyplot as plt
from pandas import Series


def line_plot(x: Series, y_data: list[Series], color_data: list[Series] = None):
    plt.figure()

    for y in y_data:
        label = y.name
        color = None if color_data is None else next((e for e in color_data if e.name == label), None)[0]
        plt.plot(x, y, color=color, label=label)

    plt.legend()
    plt.show()
