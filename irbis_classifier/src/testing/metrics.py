from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Sequence

import sklearn.metrics as sklearn_metrics


def create_confusion_matrix(
    y_true: Sequence[int],
    y_predicted: Sequence[int],
) -> list[list[int]]:
    assert len(y_true) == len(y_predicted), 'number of predictions must be equal to the number of objects'

    confision_matrix: list[list[int]] = []

    unique_actual_labels = sorted(list(set(y_true)))  # since we have all the classes in the val/test, it's OK

    for _ in unique_actual_labels:
        confision_matrix.append([0] * len(unique_actual_labels))

    for true_target, predicted_target in zip(y_true, y_predicted):
        confision_matrix[true_target][predicted_target] += 1

    return confision_matrix


class MetricFactory:
    def get_metrics_funcion(
        self,
        metric_name: str,
    ) -> Callable[[Sequence[int], Sequence[int]], float]:
        for object_name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isfunction(obj) and object_name == metric_name:
                return obj

        if hasattr(sklearn_metrics, metric_name):
            return getattr(
                sklearn_metrics,
                metric_name,
            )

        raise ValueError(f"metric you've provided '{metric_name}' wasn't found")
