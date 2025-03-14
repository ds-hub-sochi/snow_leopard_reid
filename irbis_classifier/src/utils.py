from __future__ import annotations

from pathlib import Path
from random import sample

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


def save_model_as_traced(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    save_path: str | Path,
) -> None:
    traced_model: torch.jit.ScriptModule = torch.jit.trace(
        model,
        sample_input,
    )
    traced_model.save(save_path)

    logger.success('model saved is traced model')
