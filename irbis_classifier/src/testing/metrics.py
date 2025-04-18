from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Sequence

import sklearn.metrics as sklearn_metrics


def create_confusion_matrix(
    y_true: Sequence[int],
    y_predicted: Sequence[int],
) -> list[list[int]]:
    """
    Creates a confusion matrix in the format of list of lists

    Args:
        y_true (Sequence[int]): a sequence of ground true labels
        y_predicted (Sequence[int]): a sequence of predicted labels for the same objects

    Raises:
        ValueError: raised if y_true and y_predicted have different sizes

    Returns:
        list[list[int]]: a constructed confusion matrix
    """
    if len(y_true) != len(y_predicted):
        raise ValueError('number of predictions must be equal to the number of objects')

    confision_matrix: list[list[int]] = []

    unique_actual_labels: list[int] = sorted(list(set(y_true)))
    unique_predicted_labels: list[int] = sorted(list(set(y_predicted)))

    for _ in unique_actual_labels:
        confision_matrix.append([0] * (max(unique_predicted_labels) + 1))

    for true_target, predicted_target in zip(y_true, y_predicted):
        confision_matrix[true_target][predicted_target] += 1

    return confision_matrix


class MetricFactory:
    """
    Factory that creates a metric by its string name
    """
    def get_metrics_funcion(
        self,
        metric_name: str,
    ) -> Callable[[Sequence[int], Sequence[int]], float]:
        """
        This method finds and returns a metric as a python function by its string name;
        Firstly it will try to find handwritten implementation if the irbis_classifier/src/testing/metrics.py
        then it will try to find proper impementation in the sklearn library

        Args:
            metric_name (str): name of a metric you want to use

        Raises:
            ValueError: raised if given metric name was not found nor in local and sklearn implementation

        Returns:
            Callable[[Sequence[int], Sequence[int]], float]: a metric as a python function
        """
        for object_name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isfunction(obj) and object_name == metric_name:
                return obj

        if hasattr(sklearn_metrics, metric_name):
            return getattr(
                sklearn_metrics,
                metric_name,
            )

        raise ValueError(f"metric you've provided '{metric_name}' wasn't found")
