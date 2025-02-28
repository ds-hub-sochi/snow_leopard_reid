from typing import TypeVar

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from irbis_classifier.src.testing.testers import ClassificationTester, MetricsEstimations
from irbis_classifier.src.utils import create_confusion_matrix


T = TypeVar('T', int, float)


@torch.inference_mode()
def test_model(  # pylint: disable=too-many-locals
    test_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    bootstrap_size: int = 10000,
    alpha: float = 0.95,
) -> tuple[MetricsEstimations, dict[int, MetricsEstimations], list[list[T]]]:
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

    tester: ClassificationTester = ClassificationTester()

    kwargs: dict[str, str] = {
        'average': 'macro',
    }

    confusion_matrix: list[list[T]] = create_confusion_matrix(
        y_true=targets_lst,
        y_predicted=predicted_labels_lst,
        normalize=True,
    )

    f1_score_macro: MetricsEstimations = tester._get_class_estimations(
        f1_score,
        np.array(targets_lst),
        np.array(predicted_labels_lst),
        bootstrap_size,
        alpha,
        kwargs,
    )

    kwargs['average'] = 'binary'

    f1_score_over_classes: dict[int, MetricsEstimations] = tester.get_estimation_over_class(
        f1_score,
        targets_lst,
        predicted_labels_lst,
        bootstrap_size,
        alpha,
        kwargs,
    )

    return (
        f1_score_macro,
        f1_score_over_classes,
        confusion_matrix,
    )
