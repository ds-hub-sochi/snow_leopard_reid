from collections import OrderedDict

import torch

from irbis_classifier.src.training import AnimalDataset


def get_classes_counts(
    dataset: AnimalDataset,
    n_classes: int,    
) -> OrderedDict[int, int]:
    counts: OrderedDict[int, int] = {}
    for i in range(n_classes):
        counts[i] = 0

    for _, label in dataset:
        counts[label] += 1

    return counts


def get_classes_weights(
    counts: OrderedDict[int, int],
) -> torch.Tensor:
    total_count: int = sum(list(counts.values()))
    for key in counts:
        counts[key] /= total_count

    weights: list[int] = []
    for key in counts:
        weights.append(1/counts[key])

    weights_tensor: torch.Tensor = torch.Tensor(weights)

    return weights_tensor / weights_tensor.sum()
