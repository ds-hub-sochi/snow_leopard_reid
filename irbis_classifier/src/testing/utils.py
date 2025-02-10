import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from irbis_classifier.src.testing.testers import ClassificationTester, MetricsEstimations


@torch.inference_mode()
def test_model(
    test_dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    bootstrap_size: int = 10000,
    alpha: float = 0.95
) -> dict[int, MetricsEstimations]:
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

    return tester.get_estimation_over_class(
        f1_score,
        targets_lst,
        predicted_labels_lst,
        bootstrap_size,
        alpha,
    )
