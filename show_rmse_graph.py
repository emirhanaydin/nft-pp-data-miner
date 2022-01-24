from matplotlib import pyplot as plt

from plot_metrics import PlotMetrics


def show_rmse_graph(plot_metrics: list[PlotMetrics], margin_ratio=0.99):
    name_list: list[str] = []
    rmse_list: list[float] = []

    for pm in plot_metrics:
        name_list.append(pm.plot_name)
        rmse_list.append(pm.metrics.rmse)

    min_val = min(rmse_list)
    max_val = max(rmse_list)

    ylim = [min_val * margin_ratio, max_val * (2 - margin_ratio)]

    plt.bar(name_list, rmse_list, color='#5395c3')
    plt.ylim(ylim)
    plt.title('Root Mean Squared Error')
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.show()
