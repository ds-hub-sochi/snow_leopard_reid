from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from loguru import logger
from PIL import Image
from PytorchWildlife.models import detection as pw_detection
from tqdm import tqdm


ORIGIANAL_NAMING_TO_RUSSIAN: dict[str, str] = {
    'beaver': 'Бобр',
    'procecat': 'Хорь',
}


def detection_image(  # pylint: disable=too-many-locals
    image: np.ndarray,
    detection_model: pw_detection.MegaDetectorV5,
    confidence_threshold: float = 0.5,
) -> list[dict[str, float]]:
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


@click.command()
@click.option('--path_to_data', type=click.Path(exists=True), help='The path to the data directory')
def add_data(  # pylint: disable=too-many-locals
    path_to_data: str | Path,
) -> None:
    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

    next_stage_index: int = len(glob(str(repository_root_dir / 'data' / 'raw' / 'full_images' / '*'))) + 1
    for subdir in ('full_images', 'detection_labels'):
        (repository_root_dir / 'data' / 'raw' / subdir / f'stage_{next_stage_index}').mkdir(
            exist_ok=True,
            parents=True,
        )

    images_save_dir: Path = repository_root_dir / 'data' / 'raw' / 'full_images' / f'stage_{next_stage_index}'
    markup_save_dir: Path = repository_root_dir / 'data' / 'raw' / 'detection_labels' / f'stage_{next_stage_index}'

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    detection_model: pw_detection.MegaDetectorV5 = pw_detection.MegaDetectorV5(
        device=device,
        pretrained=True,
    )

    path_to_data = Path(path_to_data).resolve()

    subdirs: list[str] = [f.path for f in os.scandir(path_to_data) if f.is_dir()]
    for subdir in subdirs:
        original_label: str = subdir.split('/')[-1]

        label: str = ORIGIANAL_NAMING_TO_RUSSIAN[original_label]

        (markup_save_dir / label).mkdir(
            exist_ok=True,
            parents=True,
        )

        (images_save_dir / label).mkdir(
            exist_ok=True,
            parents=True,
        )

        for img_path in tqdm(glob(f'{subdir}/**', recursive=True)):
            img_path = Path(img_path)
            if img_path.is_file() and img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                img = np.array(Image.open(str(img_path)).convert('RGB'))

                detection_results: list[dict[str, float]] = detection_image(
                    img,
                    detection_model,
                )

                if len(detection_results) > 0:
                    for current_results in detection_results:
                        with open(
                            markup_save_dir / label / f'{img_path.stem}.txt',
                            'w',
                            encoding='utf-8',
                        ) as f:
                            x_center: float = current_results['x_center']
                            y_center: float = current_results['y_center']
                            width: float = current_results['width']
                            height: float = current_results['height']

                            f.write(f'0 {x_center} {y_center} {width} {height}\n')

                    image_save_path: Path = images_save_dir / label / f'{img_path.stem}.JPG'

                    if os.path.exists(str(image_save_path)):
                        logger.warning(f'name collition with the {image_save_path}')

                    Image.fromarray(img).save(image_save_path)

if __name__ == '__main__':
    add_data()  # pylint: disable=no-value-for-parameter
