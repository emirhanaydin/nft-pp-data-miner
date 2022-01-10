from metrics import Metrics


class PlotMetrics:
    def __init__(self, plot_name: str, metrics: Metrics):
        self._plot_name = plot_name
        self._metrics = metrics

    @property
    def plot_name(self):
        return self._plot_name

    @property
    def metrics(self):
        return self._metrics
