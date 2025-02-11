import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from loguru import logger
import numpy as np
import numpy.typing as npt
import scipy.stats as sts
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
        metric: Callable[[T, T], float],
        y_true: T,
        y_predicted: T,
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
    ) -> dict[int, MetricsEstimations]:
        pass

    @abstractmethod
    def _get_class_estimations(  # pylint: disable=too-many-positional-arguments
        self,
        metric: Callable[[T, T], float],
        y_true: npt.NDArray[np.int_],
        y_predicted: npt.NDArray[np.int_],
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
    ) -> MetricsEstimations:
        pass


class ClassificationTester(ClassificationTesterInterface):
    def get_estimation_over_class(  # pylint: disable=too-many-positional-arguments
        self,
        metric: Callable[[T, T], float],
        y_true: T,
        y_predicted: T,
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
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

            estimations[label] = self._get_class_estimations(
                metric,
                current_label_y_true,
                current_label_y_predicted,
                bootstrap_size,
                alpha,
            )

        return estimations

    def _get_class_estimations(  # pylint: disable=too-many-positional-arguments
        self,
        metric: Callable[[T, T, Any], float],
        y_true: npt.NDArray[np.int_],
        y_predicted: npt.NDArray[np.int_],
        bootstrap_size: int = 10000,
        alpha: float = 0.95,
    ) -> MetricsEstimations:
        bootstrap_indexes = np.random.choice(np.arange(y_true.shape[0]), size=(y_true.shape[0], bootstrap_size))
        
        y_true_bootstrapped = y_true[bootstrap_indexes]
        y_predicted_bootstrapped = y_predicted[bootstrap_indexes]

        metric_estimations = np.array(
            [
                metric(temp_true, temp_predicted) for temp_true, temp_predicted in \
                zip(y_true_bootstrapped, y_predicted_bootstrapped)
            ]
        )
        std_estimation = np.std(
            metric_estimations,
            ddof=1,
        )

        point_estimation: float = metric(
            y_true=y_true,
            y_pred=y_predicted,
        )

        normal_quantile: float = sts.norm.ppf((1 + alpha) / 2)

        return MetricsEstimations(
            point_estimation,
            point_estimation - normal_quantile * std_estimation,
            point_estimation + normal_quantile * std_estimation,
        )
