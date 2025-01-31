from __future__ import annotations

from glob import glob
from pathlib import Path
from shutil import rmtree

import click
import pandas as pd
from loguru import logger

from irbis_classifier.src.find_duplicates import DuplicateFinder, DuplicateOpsProcessor, \
    export_dict2json, VIDEOS_KW, IMAGES_KW


@click.command()
@click.option('--path_to_data_dir', type=click.Path(exists=True), help='The path to the data directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
def filter_duplicates(
    path_to_data_dir: str | Path,
    path_to_save_dir: str | Path,
):
    logger.info('duplication finding has started')

    path_to_data_dir = Path(path_to_data_dir).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    temp_dir: Path = path_to_save_dir / 'temp'
    temp_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    stages: list[str] = glob(str(path_to_data_dir / '*'))
    dataframes: list[pd.DataFrame] = [pd.read_csv(file) for file in stages]

    df: pd.DataFrame = pd.concat(
        dataframes,
        axis=0,
        ignore_index=True,
    )
    df.to_csv(temp_dir / 'df.csv')

    finder: DuplicateFinder = DuplicateFinder(str(temp_dir / 'df.csv'))

    img_hashes, video_hashes = finder.find_duplicates()

    export_dict2json(
        {
            IMAGES_KW: img_hashes,
            VIDEOS_KW: video_hashes,
        },
        temp_dir / 'hashes.json',
    )

    ops_processor: DuplicateOpsProcessor = DuplicateOpsProcessor(temp_dir / 'hashes.json')
    df_filtered: pd.DataFrame = ops_processor.remove_duplicates_from_markup_file(temp_dir / 'df.csv')

    correct_pathes: list[str] = list(df_filtered.path)

    for stage in stages:
        current_df: pd.DataFrame = pd.read_csv(stage)
        current_df.query(f'path in {correct_pathes}')

        stage = stage.split('/')[-1]
        current_df.to_csv(
            path_to_save_dir / stage,
            # index=False,
        )

    rmtree(temp_dir)

    logger.success('duplication finding has ended')


if __name__ == "__main__":
    filter_duplicates()  # pylint: disable=no-value-for-parameter
