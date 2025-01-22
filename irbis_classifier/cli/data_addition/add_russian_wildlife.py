from __future__ import annotations

from xml.etree import ElementTree
from glob import glob
from pathlib import Path
from shutil import copy2

import click
import numpy as np
from loguru import logger
from PIL import Image


@click.command()
@click.option('--path_to_data', type=click.Path(exists=True), help='The path to the data directory')
def add_data(path_to_data: str | Path) -> None:  # noqa: R0914
    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

    next_stage_index: int = len(glob(str(repository_root_dir / 'data' / 'raw' / 'full_images' / '*'))) + 1
    for subdir in ('full_images', 'detection_labels'):
        (repository_root_dir / 'data' / 'raw' / subdir / f'stage_{next_stage_index}').mkdir(
            exist_ok=True,
            parents=True,
        )

    path_to_data = Path(path_to_data).resolve()

    for label in [path.split('/')[-1] for path in glob(str(path_to_data / 'images' / '*'))]:
        for subdir in ('full_images', 'detection_labels'):
            (repository_root_dir / 'data' / 'raw' / subdir / f'stage_{next_stage_index}' / label).mkdir(
                exist_ok=True,
                parents=True,
            )

        champion_label_pathes: list[str] = glob(str(path_to_data / 'images' / label / '*'))
        logger.info(f'found {len(champion_label_pathes)} images for the {label} class')

        for images_path in champion_label_pathes:
            filename: str = '.'.join(images_path.split('/')[-1].split('.')[:-1])

            if not (path_to_data / 'markup' / label / f'{filename}.xml').exists():
                logger.warning(f'missing markup for the {label}/{filename} so the image was skipped')
            else:
                image_height, image_width, _ = np.array(Image.open(images_path)).shape

                root = ElementTree.parse(path_to_data / 'markup' / label / f'{filename}.xml').getroot()
                # root = tree.getroot()

                objects = []

                for child in root:
                    if child.tag == 'object':
                        objects.append(child)

                with open(
                    repository_root_dir / 'data' / 'raw' / 'detection_labels' /  # noqa: W504
                    f'stage_{next_stage_index}' / label / f'{filename}.txt',
                    'w',
                    encoding='utf-8',
                ) as markup_file:
                    for obj in objects:
                        bbox = obj[-1]

                        x_min, y_min, x_max, y_max = [int(elem.text) for elem in bbox]  # type: ignore
                        x_center: float = ((x_max + x_min) / 2) / image_width
                        y_center: float = ((y_max + y_min) / 2) / image_height

                        width: float = (x_max - x_min) / image_width
                        height: float = (y_max - y_min) / image_height

                        markup_file.write(f'{0} {x_center} {y_center} {width} {height}\n')

                copy2(
                    images_path,
                    repository_root_dir / 'data' / 'raw' / 'full_images' / f'stage_{next_stage_index}' / label,
                )

        logger.success(f'ended with {label}')


if __name__ == '__main__':
    add_data()  # pylint: disable=no-value-for-parameter
