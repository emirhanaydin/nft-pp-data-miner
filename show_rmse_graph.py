import math

from matplotlib import pyplot as plt

from plot_metrics import PlotMetrics


def show_rmse_graph(plot_metrics: list[PlotMetrics]):
    name_list: list[str] = []
    rmse_list: list[float] = []

    for pm in plot_metrics:
        name_list.append(pm.plot_name)
        rmse_list.append(pm.metrics.rmse)

    ylim = [
        math.floor(min(rmse_list)),
        math.ceil(max(rmse_list)),
    ]

    plt.bar(name_list, rmse_list, color='blue')
    plt.ylim(ylim)
    plt.title('Root Mean Squared Error')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.show()
