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

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.utils import filter_non_images, fix_rus_i_naming
from irbis_classifier.src.series_utils import add_series_info


def construct_series(
    path_to_data_dir: Path,
    label_encoder: LabelEncoder,
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

    df['unified_class'] = df['specie'].map(label_encoder.get_unified_label)
    df = df[df['unified_class'].notna()]

    df['class_id'] = df['unified_class'].map(label_encoder.get_index_by_label)
    df = df[df['class_id'].notna()].reset_index(drop=True)

    return df


@click.command()
@click.option(
    '--path_to_data_dir',
    type=click.Path(exists=True),
    help='The path to the data directory',
)
@click.option(
    '--path_to_save_dir',
    type=click.Path(),
    help='The path to the data directory',
)
@click.option(
    '--old_stages',
    type=str,
    help='stages you want to update. Must me a comma-separated string of ints like "1,2"',
)
@click.option(
    '--path_to_unification_mapping_json',
    type=click.Path(exists=True),
    help='The path to the json file with unification mapping',
)
@click.option(
    '--path_to_supported_labels_json',
    type=click.Path(exists=True),
    help='The path to the json file with the list of supported labels',
)
@click.option(
    '--path_to_russian_to_english_mapping_json',
    type=click.Path(exists=True),
    help='The path to the json file with the russian to english mapping',
)
def find_series(
    path_to_data_dir: Path | str,
    path_to_save_dir: Path | str,
    old_stages: str,
    path_to_unification_mapping_json: Path | str,
    path_to_supported_labels_json: Path | str,
    path_to_russian_to_english_mapping_json: Path | str,
) -> None:
    path_to_data_dir = Path(path_to_data_dir).resolve()
    path_to_unification_mapping_json = Path(path_to_unification_mapping_json).resolve()
    path_to_supported_labels_json = Path(path_to_supported_labels_json).resolve()
    path_to_russian_to_english_mapping_json = Path(path_to_russian_to_english_mapping_json).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json,
        path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json,
    )

    old_stages_list: set[str] = {'stage_' + stage_number for stage_number in old_stages.split(',')}

    # old stages' labels must be updated if unification mapping was updated
    stages_to_update: list[str] = list(old_stages_list)

    for stage in stages_to_update:
        logger.info(f'updating {stage}')

        df: pd.DataFrame = pd.read_csv(
            path_to_save_dir / f'df_{stage}.csv',
            index_col = None,
        )

        df_original: pd.DataFrame = df.copy(deep=True)

        df['unified_class'] = df['specie'].map(label_encoder.get_unified_label)
        df = df[df['unified_class'].notna()]

        df['class_id'] = df['unified_class'].map(label_encoder.get_index_by_label)
        df = df[df['class_id'].notna()].reset_index(drop=True)

        if not all(
            [
                df[df.class_id.to_numpy() != df_original.class_id.to_numpy()].shape[0] == 0,
                df_original[df.class_id.to_numpy() != df_original.class_id.to_numpy()].shape[0] == 0,
            ]
        ):
            logger.error(f"During the processing of {stage} some classes' indexes have changed; all the code was stopped")
            return

        df.to_csv(
            path_to_save_dir / f'df_{stage}.csv',
            index=False,
        )
        logger.success(f'ended with {stage}')

    stages: list[str] = [f.path.split('/')[-1] for f in os.scandir(path_to_data_dir) if Path(path_to_data_dir).is_dir()]
    stages = [stage for stage in stages if stage not in old_stages_list]

    for stage in stages:
        logger.info(f'processing {stage}')
        current_stage_df: pd.DataFrame | None = construct_series(  # pylint: disable=too-many-function-args
            path_to_data_dir / stage,
            label_encoder
        )
        if current_stage_df is not None:
            current_stage_df.to_csv(
                path_to_save_dir / f'df_{stage}.csv',
                index=False,
            )
            logger.success(f'ended with {stage}')


if __name__ == '__main__':
    find_series()  # pylint: disable=no-value-for-parameter
