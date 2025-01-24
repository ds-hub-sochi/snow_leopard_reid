from __future__ import annotations

import re
from glob import glob
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger


def split_and_save(  # pylint: disable=too-many-locals
    path_to_markup_dir: Path,
    stage_files_pathes: list[str],
    dir_to_save: Path,
    train_size: float,
    val_size: float,
) -> None:
    class MarkupExistanceChecker:
        def __init__(
            self,
            path_to_markup_dir: Path,
        ):
            self._path_to_markup_dir: Path = path_to_markup_dir

        def check_markup_existance(
            self,
            filespath: str,
        ) -> bool:
            parts: list[str] = filespath.split('/')
            stage, label, filename = parts[-3:]
            filename = '.'.join(filename.split('.')[:-1])

            markup_filepath = path_to_markup_dir / stage / label / f'{filename}.txt'

            if not Path.is_file(markup_filepath):
                logger.warning(f"markup for {stage}/{label}/{filename} wasn't found")
                return False

            return True

    train_df: pd.DataFrame = pd.DataFrame()
    val_df: pd.DataFrame = pd.DataFrame()
    test_df: pd.DataFrame = pd.DataFrame()

    markup_existance_checker: MarkupExistanceChecker = MarkupExistanceChecker(path_to_markup_dir)

    for stage_file in stage_files_pathes:
        logger.info(f"processing stage #{re.findall(r'[0-9]+', stage_file.split('/')[-1])[0]}")
        current_df: pd.DataFrame = pd.read_csv(stage_file)
        current_df = current_df[current_df.path.apply(markup_existance_checker.check_markup_existance)]

        for specie in sorted(list(set(current_df.specie))):
            specie_subdf: pd.DataFrame = current_df[current_df.specie == specie]
            sequences: list[str] = sorted(list(set(specie_subdf.sequence)))

            sequence_to_length: dict[str, int] = {}

            for sequence in sequences:
                sequence_to_length[sequence] = specie_subdf[specie_subdf.sequence == sequence].shape[0]

            sequence_to_length = dict(sorted(sequence_to_length.items(), key=lambda item: item[1], reverse=True))

            index_i: int = 0

            split_indexes: list[int] = []
            current_subpart_size: int = 0
            while ((index_i < len(sequences)) and (current_subpart_size < round(train_size * specie_subdf.shape[0]))):
                current_subpart_size += sequence_to_length[sequences[index_i]]
                split_indexes.extend(list(specie_subdf[specie_subdf.sequence == sequences[index_i]].index))
                index_i += 1
            train_df = pd.concat([train_df, specie_subdf.loc[np.array(split_indexes)]])

            split_indexes = []
            current_subpart_size = 0
            while ((index_i < len(sequences)) and (current_subpart_size < round(val_size * specie_subdf.shape[0]))):
                current_subpart_size += sequence_to_length[sequences[index_i]]
                split_indexes.extend(list(specie_subdf[specie_subdf.sequence == sequences[index_i]].index))
                index_i += 1
            val_df = pd.concat([val_df, specie_subdf.loc[np.array(split_indexes)]])

            split_indexes = []
            current_subpart_size = 0
            while (index_i < len(sequences)):
                current_subpart_size += sequence_to_length[sequences[index_i]]
                split_indexes.extend(list(specie_subdf[specie_subdf.sequence == sequences[index_i]].index))
                index_i += 1
            test_df = pd.concat([test_df, specie_subdf.loc[np.array(split_indexes)]])

    logger.info(f'train size: {train_df.shape[0]}')
    train_df.to_csv(
        dir_to_save / 'train.csv',
        index=False,
    )

    logger.info(f'val size: {val_df.shape[0]}')
    val_df.to_csv(
        dir_to_save / 'val.csv',
        index=False,
    )

    logger.info(f'test size: {test_df.shape[0]}')
    test_df.to_csv(
        dir_to_save / 'test.csv',
        index=False,
    )

    logger.success('train/val/test split ended')


@click.command()
@click.option(
    '--path_to_dir_with_stages',
    type=click.Path(exists=True),
    help='The path to the directory that contains stages',
)
@click.option('--path_to_markup_dir', type=click.Path(exists=True), help='The path to the markup directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
@click.option('--train_size', default=0.60, help='Train subpart relative size')
@click.option('--val_size', default=0.20, help='Validation subpart relative size')
def run_split(
    path_to_dir_with_stages: Path | str,
    path_to_markup_dir: Path | str,
    path_to_save_dir: Path | str,
    train_size: float,
    val_size: float,
) -> None:
    path_to_dir_with_stages = Path(path_to_dir_with_stages).resolve()
    path_to_markup_dir = Path(path_to_markup_dir).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    stage_files: list[str] = sorted(glob(str(path_to_dir_with_stages / '*.csv')))
    logger.info(f'found {len(stage_files)} stage files')

    split_and_save(
        path_to_markup_dir,
        stage_files,
        path_to_save_dir,
        train_size=train_size,
        val_size=val_size,
    )


if __name__ == '__main__':
    run_split()  # pylint: disable=no-value-for-parameter
