import matplotlib.pyplot as plt

from cross_validator import TestPredictionResult


def show_test_pred_graph(test_pred_result: TestPredictionResult, title: str):
    x_test = test_pred_result.x_test
    pred = test_pred_result.prediction

    plt.figure()
    plt.scatter(x_test, pred, s=20, edgecolor='black', c='darkorange', label='data')
    plt.plot(x_test, pred, color='yellowgreen', linewidth=2)
    plt.xlabel('data')
    plt.ylabel('prediction')
    plt.title(title)
    plt.legend()
    plt.show()
