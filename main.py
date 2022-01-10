import pandas as pd

from cross_validator import CrossValidator
from decision_tree_model import DecisionTreeModel
from linear_model import LinearModel
from mean_filter import mean_filter
from plot_metrics import PlotMetrics
from polynomial_model import PolynomialModel
from show_rmse_graph import show_rmse_graph
from show_test_pred_graph import show_test_pred_graph


def main():
    dataset = pd.read_csv('data.csv').dropna(subset=['Front'])
    x = dataset.iloc[:, 7:8].values
    y = dataset.iloc[:, 11].values
    y = mean_filter(y)

    validator = CrossValidator(x, y)
    results = [
        validator.predict(LinearModel()),
        validator.predict(PolynomialModel()),
        validator.predict(DecisionTreeModel(max_depth=2)),
    ]

    model_titles = [
        "Linear Regression",
        "Polynomial Regression",
        "Decision Tree Regression",
    ]
    iter_range = range(len(model_titles))

    # Show RMSE graph
    plot_metrics = [PlotMetrics(model_titles[i], results[i].metrics) for i in iter_range]
    show_rmse_graph(plot_metrics)

    # Show test - prediction graph
    for i in iter_range:
        show_test_pred_graph(results[i].best_test_pred, model_titles[i])


if __name__ == '__main__':
    main()
