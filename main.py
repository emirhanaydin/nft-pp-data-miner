import pandas as pd

from cross_validator import CrossValidator
from df_filter import (
    cat_to_num_filter,
    column_filter,
    DataFrameFilter,
    empty_filter,
    unique_filter,
    correlation_filter,
)
from line_plot import line_plot
from mean_filter import mean_filter
from metrics import Metrics
from model import (
    DecisionTreeModel,
    LinearModel,
    PolynomialModel,
)
from plot_metrics import PlotMetrics
from show_rmse_graph import show_rmse_graph


def main():
    dataset = DataFrameFilter(pd.read_csv('data.csv')) \
        .pipe(empty_filter) \
        .pipe(unique_filter) \
        .pipe(column_filter) \
        .pipe(cat_to_num_filter) \
        .pipe(correlation_filter) \
        .filter()

    x = dataset.iloc[:, dataset.columns != 'Price'].values
    y = dataset['Price'].values
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
    plot_metrics = [PlotMetrics(model_titles[i], Metrics.from_test_pred(y, results[i].prediction)) for i in iter_range]
    show_rmse_graph(plot_metrics)

    # Show test - prediction graph
    max_rows = 200
    idxes = [i for i, v in enumerate(y) if v < 0.5]
    idxes = idxes[:max_rows]
    x_plot = pd.Series(data=idxes, name='id')
    y_plot = [pd.Series(data=results[i].prediction[idxes], name=model_titles[i]) for i in iter_range]
    y_plot.insert(0, pd.Series(data=y[idxes], name='Price'))
    color_data = ['#56b4e9', '#e69f00', '#cc79a7', '#009e73']
    color_data = [pd.Series(data=[color_data[i]], name=y_plot[i].name) for i in range(len(y_plot))]

    line_plot(x_plot, y_plot, color_data)


if __name__ == '__main__':
    main()
