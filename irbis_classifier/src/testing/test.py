from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from irbis_classifier.src.testing.metrics import create_confusion_matrix, MetricFactory
from irbis_classifier.src.testing.testers import ClassificationTester, MetricsEstimations


@torch.inference_mode()
def test_model(  # pylint: disable=too-many-locals
    test_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    metrics: Sequence[str],
    bootstrap_size: int = 10000,
    alpha: float = 0.95,
) -> tuple[dict[str, tuple[MetricsEstimations, dict[int, MetricsEstimations]]], list[list[int]]]:
    targets_lst: list[int] = []
    predicted_labels_lst: list[int] = []

    device: torch.device = next(model.parameters()).device

    for input_batch, targets in tqdm(test_dataloader):
        input_batch = input_batch.to(device)
        targets = targets.to(device)

        with torch.autocast(
            device_type='cuda',
            dtype=torch.float16,
        ):
            predicted_logits = model(input_batch)

        predicted_labels = torch.argmax(
            predicted_logits,
            dim=1,
        )

        predicted_labels_lst.extend(predicted_labels.tolist())
        targets_lst.extend(targets.tolist())

    confusion_matrix: list[list[int]] = create_confusion_matrix(
        y_true=targets_lst,
        y_predicted=predicted_labels_lst,
    )

    tester: ClassificationTester = ClassificationTester()
    metric_factory: MetricFactory = MetricFactory()

    metric_results: dict[str, tuple[MetricsEstimations, dict[int, MetricsEstimations]]] = {}

    for metric_name in metrics:
        try:
            metric = metric_factory.get_metrics_funcion(metric_name)
        except ValueError as error:
            logger.warning(f'{str(error)}')
        else:
            kwargs: dict[str, Any] = {
                'average': 'macro',
            }

            cumulative_value: MetricsEstimations = tester.get_cumulative_estimate(
                metric,
                np.array(targets_lst),
                np.array(predicted_labels_lst),
                bootstrap_size,
                alpha,
                kwargs,
            )

            kwargs: dict[str, Any] = {
                'average': 'binary',
                'pos_label': 1,
            }

            value_over_classes: dict[int, MetricsEstimations] = tester.get_estimation_over_class(
                metric,
                np.array(targets_lst),
                np.array(predicted_labels_lst),
                bootstrap_size,
                alpha,
                kwargs,
            )

            metric_results[metric_name] = (
                cumulative_value,
                value_over_classes,
            )

    return (
        metric_results,
        confusion_matrix,
    )
