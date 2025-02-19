from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pandas as pd
import torch
from loguru import logger


def get_classes_counts(
    dataset_csv: str | Path,
    n_classes: int,    
) -> OrderedDict[int, int]:
    logger.info("processing classes' counts")

    dataset_df: pd.DataFrame = pd.read_csv(dataset_csv)

    counts: OrderedDict[int, int] = {}
    for i in range(n_classes):
        counts[i] = 0

    for label in range(n_classes):
        counts[label] += dataset_df[dataset_df.class_id == label].shape[0]

    logger.success("classes' counts were processed")

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
