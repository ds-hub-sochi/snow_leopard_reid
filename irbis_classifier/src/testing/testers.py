from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt
import scipy.stats as sts
from joblib import delayed, Parallel
from loguru import logger
from tqdm import tqdm


@dataclass
class MetricsEstimations:
    point: float
    lower: float
    upper: float


T = TypeVar('T', bound=Sequence[int])


class ClassificationTesterInterface(ABC):
    @abstractmethod
    def get_estimation_over_class(  # pylint: disable=too-many-positional-arguments
        self,
        metric: Callable[[T, T, Any], float],
        y_true: T,
        y_predicted: T,
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
        metric_kwargs: dict[str, str] | None = None,
    ) -> dict[int, MetricsEstimations]:
        """
        This method independently computes some metric over every label presented in the dataset.
        Confidence intervals will be computed using the bootstrap estimations

        Args:
            metric (Callable[[T, T, Any], float]): metric you want to use
            y_true (T): ground true labels
            y_predicted (T): predicted labels
            bootstrap_size (int, optional): a size of a bootstrapped sample. Defaults to 10000.
            alpha (float, optional): confidence level of a confidence interval. Defaults to 0.95.
            metric_kwargs (dict[str, str] | None, optional): metrci specific kwargs. Defaults to None.

        Returns:
            dict[int, MetricsEstimations]: mapping from labels to metric's confidence intervals 
        """
        pass

    @abstractmethod
    def get_cumulative_estimate(  # pylint: disable=too-many-positional-arguments
        self,
        metric: Callable[[T, T, Any], float],
        y_true: npt.NDArray[np.int_],
        y_predicted: npt.NDArray[np.int_],
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
        metric_kwargs: dict[str, str] | None = None,
    ) -> MetricsEstimations:
        """
        Computes cumulative metric over the dataset

        Args:
            metric (Callable[[T, T, Any], float]): metric you want to use
            y_true (T): ground true labels
            y_predicted (T): predicted labels
            bootstrap_size (int, optional): a size of a bootstrapped sample. Defaults to 10000.
            alpha (float, optional): confidence level of a confidence interval. Defaults to 0.95.
            metric_kwargs (dict[str, str] | None, optional): metrci specific kwargs. Defaults to None.

        Returns:
            MetricsEstimations: confidence interval for the given metric
        """
        pass


class ClassificationTester(ClassificationTesterInterface):
    def get_estimation_over_class(  # pylint: disable=too-many-positional-arguments
        self,
        metric: Callable[[T, T, Any], float],
        y_true: T,
        y_predicted: T,
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
        metric_kwargs: dict[str, str] | None = None,
    ) -> dict[int, MetricsEstimations]:
        parameters: list[str] = list(inspect.signature(metric).parameters.keys())

        assert all(
            [
                'y_true' in parameters,
                'y_pred' in parameters,
            ]
        ), logger.error('your function must have the following args: "y_true", "y_pred"')

        y_true_array: npt.NDArray[np.int_] = np.array(y_true)
        y_predicted_array: npt.NDArray[np.int_] = np.array(y_predicted)

        estimations: dict[int, MetricsEstimations] = {}

        unique_classes: list[int] = sorted(list(set(y_true)))
        for label in tqdm(unique_classes):
            answer_indexes: npt.NDArray[np.int_] = (y_true_array == label).nonzero()[0]
            prediction_indexes: npt.NDArray[np.int_] = (y_predicted_array == label).nonzero()[0]

            union_indexes: npt.NDArray[np.int_] = np.array(
                list(set(answer_indexes.tolist()) | set(prediction_indexes.tolist())),
                dtype=np.int32,
            )

            current_label_y_true: npt.NDArray[np.int_] = y_true_array[union_indexes]
            current_label_y_true[current_label_y_true != label] = label + 1
            current_label_y_true = np.where(current_label_y_true == label, 1, 0)

            current_label_y_predicted: npt.NDArray[np.int_] = y_predicted_array[union_indexes]
            current_label_y_predicted[current_label_y_predicted != label] = label + 1
            current_label_y_predicted = np.where(current_label_y_predicted == label, 1, 0)

            estimations[label] = self.get_cumulative_estimate(
                metric,
                current_label_y_true,
                current_label_y_predicted,
                bootstrap_size,
                alpha,
                metric_kwargs,
            )

        return estimations

    def get_cumulative_estimate(  # pylint: disable=too-many-positional-arguments
        self,
        metric: Callable[[T, T, Any], float],
        y_true: T,
        y_predicted: T,
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
        metric_kwargs: dict[str, str] | None = None,
    ) -> MetricsEstimations:
        if metric_kwargs is None:
            metric_kwargs = {}

        y_true_array = np.array(y_true)
        y_predicted_array = np.array(y_predicted)

        bootstrap_indexes: npt.NDArray[np.int_] = np.random.choice(np.arange(y_true_array.shape[0]), size=(bootstrap_size, y_true_array.shape[0]))
        
        y_true_bootstrapped: npt.NDArray[np.int_] = y_true_array[bootstrap_indexes]
        y_predicted_bootstrapped: npt.NDArray[np.int_] = y_predicted_array[bootstrap_indexes]

        metric_estimations: npt.NDArray[np.float_] = np.array(
            list(
                Parallel(n_jobs=os.cpu_count())(delayed(metric)(temp_true, temp_predicted, **metric_kwargs) for \
                temp_true, temp_predicted in zip(y_true_bootstrapped, y_predicted_bootstrapped))
            )
        )

        std_estimation: np.float32 = np.std(
            metric_estimations,
            ddof=1,
        )

        point_estimation: float = metric(
            y_true=y_true_array,
            y_pred=y_predicted_array,
            **metric_kwargs,
        )

        normal_quantile: np.float32 = sts.norm.ppf((1 + alpha) / 2)

        return MetricsEstimations(
            point_estimation,
            point_estimation - normal_quantile * std_estimation,
            point_estimation + normal_quantile * std_estimation,
        )
