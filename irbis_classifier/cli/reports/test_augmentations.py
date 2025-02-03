from __future__ import annotations

from pathlib import Path

import albumentations as A
import click
import numpy as np
from loguru import logger

from irbis_classifier.src.plots import create_image_grid
from irbis_classifier.src.training.datasets import AnimalDataset


@click.command()
@click.option(
    '--path_to_data_file',
    type=click.Path(exists=True),
    help='The path to the data file in csv format',
)
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
@click.option(
    '--n_samples',
    type=int,
    default=35,
    help='Numner of samples to produce (original image excluded). Must be a square - 1, like 35 or 15',
)
@click.option('--image_index', type=int, default=4125, help='index of an image you want to display')
def get_figure(
    path_to_data_file: Path | str,
    path_to_save_dir: Path | str,
    n_samples: int,
    image_index: int,
):
    path_to_data_file = Path(path_to_data_file).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    val_transfroms = A.Compose(
        [
            A.LongestMaxSize(
                max_size = 256,
                p = 1.0,
            ),
            A.PadIfNeeded(
                min_height = 256,
                min_width = 256,
                position = 'center',
                border_mode = 0,
                fill = 0,
                p = 1.0,
            ),
            A.Resize(
                height = 224,
                width = 224,
                p = 1.0,
            ),
        ]
    )
    val_dataset: AnimalDataset = AnimalDataset(
        path_to_data_file,
        val_transfroms,
    )

    train_transforms = A.Compose(
        [
            A.LongestMaxSize(
                max_size = 256,
                p = 1.0,
            ),
            A.ToGray(
                num_output_channels = 3,
                p = 0.10,
            ),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit = (-20, 20),
                        sat_shift_limit = (-30, 30),
                        val_shift_limit = (-20, 20),
                        p = 0.85,
                    ),
                    A.Equalize(
                        mode = 'cv',
                        by_channels = True,
                        p = 0.85,
                    ),
                    A.RandomGamma(
                        gamma_limit = (80, 120),
                        p = 0.85,
                    )
                ],
                p = 1.0
            ),
            A.OneOf(
                [
                    A.Defocus(
                        radius = (3, 5),
                        alias_blur = (0.01, 0.02),
                        p = 0.85,
                    ),
                    A.GlassBlur(
                        sigma = 0.5,
                        max_delta = 2,
                        iterations = 1,
                        mode = 'fast',
                        p = 0.85,
                    ),
                    A.Blur(
                        blur_limit = (3, 7),
                        p = 0.85,
                    ),
                ],
                p = 1.0
            ),
            A.OneOf(
                [
                    A.RandomSunFlare(
                        flare_roi = (0.0, 0.0, 1.0, 0.25),
                        num_flare_circles_range = (2, 5),
                        src_radius = 256 // 2,
                        p = 0.05,
                    ),
                    A.RandomRain(
                        p = 0.05,
                    ),
                    A.SaltAndPepper(
                        amount = (0.05, 0.10),
                        p = 0.10,
                    ),
                ],
                p = 1.0,
            ),
            A.RandomCrop(
                height = 224,
                width = 224,
                pad_if_needed = True,
                border_mode = 0,
                fill = 0,
                pad_position = 'center',
                p = 1.0,
            ),
            A.SafeRotate(
                limit = (-30, 30),
                p = 0.85,
            ),
            A.HorizontalFlip(
                p = 0.85,
            ),
        ]
    )
    train_dataset: AnimalDataset = AnimalDataset(
        path_to_data_file,
        train_transforms,
    )

    try:
        assert image_index < len(train_dataset), ''
    except AssertionError:
        logger.error('index must be lower than size of the dataset')
    else:
        images: list[np.ndarray] = []
        images.append(val_dataset[image_index][0])

        titles: list[str] = []
        titles.append('Оригинальное изображение')

        for i in range(n_samples):
            images.append(train_dataset[image_index][0])
            titles.append(f'Вариант #{i + 1}')

        create_image_grid(
            images,
            titles,
            False,
            True,
            path_to_save_dir,
        )


if __name__ == '__main__':
    get_figure()  # pylint: disable=no-value-for-parameter