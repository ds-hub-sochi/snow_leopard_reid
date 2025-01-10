from glob import glob
from pathlib import Path

import click
import numpy as np
import pandas as pd
from loguru import logger


def split_and_save(
    repository_root_dir: Path,
    stage_files_pathes: list[str],
    dir_to_save: Path,
    train_size: float,
    val_size: float,
) -> None:
    def check_markup_existance(filespath: str) -> bool:
        parts: list[str] = filespath.split('/')
        stage, label, filename = parts[-3:]
        filename = '.'.join(filename.split('.')[:-1])

        markup_filepath = repository_root_dir / \
            '/'.join(parts[:-4]) / 'detection_labels' / stage / label / f'{filename}.txt'

        if not Path.is_file(markup_filepath):
            logger.warning(f"Markup for {stage}/{label}/{filename} wasn't found")
            return False

        return True

    dir_to_save.mkdir(parents=True, exist_ok=True)

    train_df: pd.DataFrame = pd.DataFrame()
    val_df: pd.DataFrame = pd.DataFrame()
    test_df: pd.DataFrame = pd.DataFrame()

    for stage_file in stage_files_pathes:
        current_df: pd.DataFrame = pd.read_csv(stage_file)
        current_df = current_df[current_df.path.apply(check_markup_existance)]

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
    train_df.to_csv(dir_to_save / "train.csv", index=False)

    logger.info(f'val size: {val_df.shape[0]}')
    val_df.to_csv(dir_to_save / 'val.csv', index=False)

    logger.info(f"test size: {test_df.shape[0]}")
    test_df.to_csv(dir_to_save / 'test.csv', index=False)

    logger.success('train/val/test split ended')


@click.command()
@click.option('--train_size', default=0.60, help='Train subpart relative size')
@click.option('--val_size', default=0.20, help='Validation subpart relative size')
def run_split(train_size: float, val_size: float) -> None:
    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()
    stage_files: list[str] = sorted(
        glob(str(repository_root_dir / 'data' / 'interim' / 'stage_with_series' / '*.csv'))
    )

    split_and_save(
        repository_root_dir,
        stage_files,
        repository_root_dir / 'data' / 'interim' / 'train_val_test_split',
        train_size=train_size,
        val_size=val_size,
    )


if __name__ == '__main__':
    run_split()  # pylint: disable=no-value-for-parameter
