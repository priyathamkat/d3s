from typing import Any, List
from IPython.display import clear_output, display


class ScoreTracker:
    def __init__(self, name: str, low: int = 1, high: int = 5) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.score = 0
        self.n = 1
        self._scores = []

    def update(self, value):
        assert self.low <= value <= self.high
        self._scores.append(value)
        self.score += (value - self.score) / self.n
        self.n += 1

    def __str__(self) -> str:
        percentage = 100 * (self.score - self.low) / (self.high - self.low)
        return f"Average {self.name} ({self.low} to {self.high}) over {self.n - 1} samples: {self.score} ({percentage:.2f})"


def format_attr_name(name):
    return name.lower().replace(" ", "_")


class ExperimentTracker:
    def __init__(self, name: str, metrics: List[Any]) -> None:
        self.name = name
        self.metrics = []
        for metric in metrics:
            if isinstance(metric, tuple):
                metric, low, high = metric
                metric_attr_name = format_attr_name(metric)
                setattr(
                    self, metric_attr_name, ScoreTracker(metric, low=low, high=high)
                )
            else:
                metric_attr_name = format_attr_name(metric)
                setattr(self, metric_attr_name, ScoreTracker(metric))

            self.metrics.append(getattr(self, metric_attr_name))

    def __str__(self) -> str:
        output = f"{self.name} experiment\n"
        for metric in self.metrics:
            output += f"{str(metric)}\n"
        return output


def experiment(name: str, images: List[Any], metrics: List[Any]):
    tracker = ExperimentTracker(name, metrics)
    for prompt, image in images:
        clear_output(wait=True)
        print(f"Class: {prompt}")
        display(image)
        for metric in tracker.metrics:
            try:
                value = int(
                    input(f"Score for {metric.name} ({metric.low} to {metric.high}):")
                )
            except AssertionError:
                print("Try again")
                value = int(
                    input(f"Score for {metric.name} ({metric.low} to {metric.high}):")
                )
            metric.update(value)
    print("-----Results-----")
    print(tracker)