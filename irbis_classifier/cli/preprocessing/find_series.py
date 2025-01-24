"""
Скрипт, в котором мы вытаскиваем информацию о сериях в фотографиях, чтобы
затем можно было сделать сплит по сериям и не допустить лика.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from irbis_classifier.src.constants import UNIFICATION_MAPPING, CLASSES_TO_USE, LABEL_TO_INDEX
from irbis_classifier.src.utils import filter_non_images, fix_rus_i_naming
from irbis_classifier.src.series_utils import add_series_info


class LabelFilter(Callable[[str], str | None]):  # type: ignore  # pylint: disable=unsupported-binary-operation
    def __init__(
        self,
        unification_mapping: dict[str, str],
        supported_labels: set[str],
    ):
        super().__init__()

        self._unification_mapping: dict[str, str] = unification_mapping
        self._supported_labels: set[str] = supported_labels

    def __call__(
        self,
        label: str
    ) -> str | None:
        label = self._unification_mapping.get(label, label)

        if label not in self._supported_labels:
            return None

        return label


def construct_series(
    path_to_data_dir: Path,
) -> pd.DataFrame | None:
    all_image_paths: list[Path] = list(path_to_data_dir.rglob('*.*'))
    all_image_paths = filter_non_images(all_image_paths)
    logger.info(f'found {len(all_image_paths)} images')

    if len(all_image_paths) == 0:
        logger.warning('found 0 images; stage skipped')

        return None

    species_list: list[str] = [path.parent.name for path in all_image_paths]

    df: pd.DataFrame = pd.DataFrame({'path': all_image_paths, 'specie': species_list})
    df = add_series_info(df)

    # drop photos from blacklist
    # repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()
    # if (repository_root_dir / 'external' / 'blacklist.txt').is_file():
    #     df = drop_blacklist_photos(df, path_to_data_dir)

    df['specie'] = df['specie'].apply(fix_rus_i_naming)

    label_filter: LabelFilter = LabelFilter(
        unification_mapping=UNIFICATION_MAPPING,
        supported_labels=CLASSES_TO_USE,
    )

    df['unified_class'] = df['specie'].map(label_filter)
    df = df[df['unified_class'].notna()]

    df['class_id'] = df['unified_class'].map(LABEL_TO_INDEX)
    df = df[df['class_id'].notna()].reset_index(drop=True)

    return df


@click.command()
@click.option('--path_to_data_dir', type=click.Path(exists=True), help='The path to the data directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
def find_series(
    path_to_data_dir: Path | str,
    path_to_save_dir: Path | str,
) -> None:
    path_to_data_dir = Path(path_to_data_dir).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    stages: list[str] = [f.path.split('/')[-1] for f in os.scandir(path_to_data_dir) if Path(path_to_data_dir).is_dir()]
    for stage in stages:
        logger.info(f'processing {stage}')
        current_stage_df: pd.DataFrame | None = construct_series(path_to_data_dir / stage)
        if current_stage_df is not None:
            current_stage_df.to_csv(
                path_to_save_dir / f'df_{stage}.csv',
                index=False,
            )
            logger.success(f'ended with {stage}')


if __name__ == '__main__':
    find_series()  # pylint: disable=no-value-for-parameter
