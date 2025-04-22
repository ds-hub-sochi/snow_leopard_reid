from __future__ import annotations

from pathlib import Path
from random import sample
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from PytorchWildlife.models import detection as pw_detection


def filter_non_images(image_paths: list[Path]) -> list[Path]:
    """
    FIlter non-images files from the given list of pathes

    Args:
        image_paths (list[Path]): list of pathes you want to filted

    Returns:
        list[Path]: list of only images pathes
    """
    return [
        image_path for image_path in image_paths
        if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}
    ]


def fix_rus_i_naming(filename: str) -> str:
    """
    Fix inconsistency with 'й' symbol.
    First one is quite normal in Ubuntu/Mac and presents in .json markup
    The second one is different and presents in original filenames
    """
    return filename.replace('й', 'й')


def sample_from_dataframe(
    df: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """
    This function is used to sample given number of rows from the dataframe

    Args:
        df (pd.DataFrame): dataframe whose rows you want to sample
        sample_size (int): number of rows you want to sample

    Returns:
        pd.DataFrame: a dataframe with re-sampled rows
    """
    indices: list[int] = sample(list(df.index), sample_size)

    return df.loc[indices]


def save_model_as_traced(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    save_path: str | Path,
) -> None:
    """
    Simple wrapper that serves to save nn.Module model as a traced model

    Args:
        model (torch.nn.Module): model you want to save as a traced model
        sample_input (torch.Tensor): a tensor that serves as a sample input so everything can be compiled properly
        save_path (str | Path): path to the file where you want to save your model
    """
    traced_model: torch.jit.ScriptModule = torch.jit.trace(
        model,
        sample_input,
    )
    traced_model.save(save_path)

    logger.success('model saved as traced model')


@torch.inference_mode()
def detect_in_image(  # pylint: disable=too-many-locals
    image: np.ndarray,
    detection_model: pw_detection.MegaDetectorV5,
    confidence_threshold: float = 0.5,
) -> list[dict[str, float]]:
    """
    This function is used to infer MegaDetector so we can get an animals markup. 
    Markup is presented in the x_center, y_center, width, height format

    Args:
        image: (np.ndarray): an image where you want to detect animals
        detection_model (pw_detection.MegaDetectorV5): model you want to use for the detection
        confidence_threshold (float, optional): a confidence of a model. Defaults to 0.5.

    Returns:
        list[dict[str, float]]: list of dict with markup for every found animal
    """
    results: dict[str, Any] = detection_model.single_image_detection(image)

    if len(results["detections"].xyxy) == 0:
        return []

    labels: list[str] = [value.split(' ')[0] for value in results['labels']]
    labels_np = np.array(
        labels,
        dtype=np.dtype(object),
    )

    mask: np.ndarray = (results["detections"].confidence > confidence_threshold) & (labels_np == 'animal')
    detection_markup: np.ndarray = results["detections"].xyxy[mask]

    image_height, image_width, _ = image.shape

    markup: list[dict[str, float]] = []
    for current_markup in detection_markup:
        x_upper_left: int = int(current_markup[0])
        y_upper_left: int = int(current_markup[1])
        x_lower_right: int = int(current_markup[2])
        y_lower_right: int = int(current_markup[3])
    
        x_center: float = (x_upper_left + (x_lower_right - x_upper_left) / 2) / image_width
        y_center: float = (y_upper_left + (y_lower_right - y_upper_left) / 2) / image_height
        width: float = (x_lower_right - x_upper_left) / image_width
        height: float = (y_lower_right - y_upper_left) / image_height

        markup.append(
            {
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
            }
        )

    return markup
