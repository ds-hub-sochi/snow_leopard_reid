from collections.abc import Sequence
from pathlib import Path
from random import sample

import pandas as pd


def filter_non_images(image_paths: list[Path]) -> list[Path]:
    return [
        image_path for image_path in image_paths
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}
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

    sampled_values = df.loc[indices]

    return sampled_values


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
