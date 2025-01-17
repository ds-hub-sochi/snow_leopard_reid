"""
Скрипт, в котором мы вытаскиваем информацию о сериях в фотографиях, чтобы
затем можно было сделать сплит по сериям и не допустить лика.
"""
from __future__ import annotations

import os
from pathlib import Path

import click
import pandas as pd
from loguru import logger

from irbis_classifier.src.constants import UNIFICATION_MAPPING, CLASSES_TO_USE, LABEL_TO_INDEX
from irbis_classifier.src.utils import filter_non_images, fix_rus_i_naming
from irbis_classifier.src.series_utils import add_series_info


def drop_blacklist_photos(
    df: pd.DataFrame,
    path_to_data: Path,
) -> pd.DataFrame:
    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

    blacklist: list[Path] = []
    with open(repository_root_dir / 'external' / 'blacklist.txt', 'r', encoding='utf-8') as blacklist_file:
        for line in blacklist_file:
            line = line.strip()
            if line:
                blacklist.append(path_to_data / line)

    for item in blacklist:
        # Need reverse transform to fix_rus_i_naming, because
        # item = str(item).replace('й', 'й')
        item = Path(fix_rus_i_naming(str(item)))
        if item in df['path'].values:
            logger.info(f"Элемент '{item}' найден в DataFrame и будет удален.")
            df = df[df['path'] != item]
        else:
            raise ValueError(f"No {item} found to delete.")

    return df.reset_index(drop=True)


class LabelFilter:
    def __init__(
        self,
        unification_mapping: dict[str, str],
        supported_labels: set[str],
    ):
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
) -> pd.DataFrame:
    all_image_paths: list[Path] = list(path_to_data_dir.rglob('*.*'))
    all_image_paths = filter_non_images(all_image_paths)
    logger.info(f'Found {len(all_image_paths)} images')

    species_list: list[str] = [path.parent.name for path in all_image_paths]

    df: pd.DataFrame = pd.DataFrame({'path': all_image_paths, 'specie': species_list})
    df = add_series_info(df)

    # drop photos from blacklist
    # repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()
    # if (repository_root_dir / 'external' / 'blacklist.txt').is_file():
    #     df = drop_blacklist_photos(df, path_to_data_dir)

    df['specie'] = df['specie'].apply(fix_rus_i_naming)

    # Делаем маппинг в классы более высокого уровня, отчищаем от тех, что не учтены в маппере
    label_filter: LabelFilter = LabelFilter(
        unification_mapping=UNIFICATION_MAPPING,
        supported_labels=CLASSES_TO_USE,
    )

    df['unified_class'] = df['specie'].map(label_filter)
    df = df[df['unified_class'].notna()]
    # df = df[df['unified_class'] is not None]
    # Трансформируем классы более высокого уровня в числовые индексы и удаляем то, не учтено в маппере
    # Например: колонок (2 фото) и собачьи (66 фото).
    df['class_id'] = df['unified_class'].map(LABEL_TO_INDEX)
    df = df[df['class_id'].notna()].reset_index(drop=True)

    return df


@click.command()
@click.option('--path_to_data', type=click.Path(exists=True), help='The path to the data directory')
def find_series(path_to_data: str) -> None:
    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()
    (repository_root_dir / 'data' / 'interim' / 'stage_with_series').mkdir(exist_ok=True, parents=True)

    stages: list[str] = [f.path.split('/')[-1] for f in os.scandir(path_to_data) if Path(path_to_data).is_dir()]
    for stage in stages:
        logger.info(f'processing {stage}')
        current_stage_df: pd.DataFrame = construct_series(Path(path_to_data) / stage)
        current_stage_df.to_csv(
            repository_root_dir / 'data' / 'interim' / 'stage_with_series' / f'df_{stage}.csv',
            index=False,
        )


if __name__ == '__main__':
    find_series()  # pylint: disable=no-value-for-parameter
