from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from random import sample
from typing import TypeVar

import pandas as pd
import torch
from loguru import logger


def filter_non_images(image_paths: list[Path]) -> list[Path]:
    return [
        image_path for image_path in image_paths
        if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}
    ]


def fix_rus_i_naming(filename: str) -> str:
    # Fix inconsistency with 'й' symbol.
    # First one is quite normal in Ubuntu/Mac and presents in .json markup
    # The second one is different and presents in original filenames
    return filename.replace('й', 'й')


def sample_from_dataframe(
    df: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    indices: list[int] = sample(list(df.index), sample_size)

    return df.loc[indices]


T = TypeVar('T', int, float)


def create_confusion_matrix(
    y_true: Sequence[int],
    y_predicted: Sequence[int],
    normalize: bool = False
) -> list[list[T]]:
    assert len(y_true) == len(y_predicted), 'number of predictions must be equal to the number of objects'

    confision_matrix: list[list[T]] = []

    unique_actual_labels = sorted(list(set(y_true)))  # since we have all the classes in the val/test, it's OK

    for _ in unique_actual_labels:
        confision_matrix.append([0] * len(unique_actual_labels))

    for true_target, predicted_target in zip(y_true, y_predicted):
        confision_matrix[true_target][predicted_target] += 1

    if normalize:
        for true_target in range(len(confision_matrix)):
            row_sum: int = sum(confision_matrix[true_target])

            for predicted_target in range(len(confision_matrix[true_target])):
                confision_matrix[true_target][predicted_target] /= row_sum
                confision_matrix[true_target][predicted_target] = round(
                    confision_matrix[true_target][predicted_target],
                    2,
                )

    return confision_matrix


def save_model_as_traced(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    save_path: str | Path,
) -> None:
    traced_model: torch.jit.ScriptModule = torch.jit.trace(model, sample_input)
    traced_model.save(save_path)

    logger.success('model saved is traced model')
