from __future__ import annotations

import re
import pathlib
from glob import glob
from pathlib import Path

import click
import pandas as pd
from loguru import logger


def add_markup(
    split_files_pathes: list[str],
    path_to_save_dir: Path,
    path_to_markup_dir: Path,
    min_relative_size: float,
) -> None:
    for filepath in split_files_pathes:
        filename: str = '.'.join(filepath.split('/')[-1].split('.')[:-1])

        current_df: pd.DataFrame = pd.read_csv(filepath)

        dict_with_markup: dict[str, list[str | int]] = {
            'path' : [],
            'specie' : [],
            'x_center': [],
            'y_center': [],
            'width': [],
            'height': [],
            'stage': [],
        }

        for loc_index in range(current_df.shape[0]):
            current_series: pd.Series = pd.Series(current_df.loc[loc_index])

            parts: list[str] = current_series.path.split('/')
            stage, label, image_filename = parts[-3:]
            image_filename = '.'.join(image_filename.split('.')[:-1])

            markup_filepath: pathlib.Path = path_to_markup_dir / stage / label / f'{image_filename}.txt'

            if (not markup_filepath.is_file()):
                logger.warning(f"markup for {stage}/{label}/{image_filename} wasn't found")

            with open(
                markup_filepath,
                'r',
                encoding='utf-8',
            ) as markup_file:
                for markup_line in markup_file:
                    markup_parts: list[str] = markup_line.split(' ')
                    if float(markup_parts[-1]) > min_relative_size and float(markup_parts[-2]) > min_relative_size:
                        dict_with_markup['path'].append(current_series.path)
                        dict_with_markup['specie'].append(current_series.unified_class)
                        dict_with_markup['x_center'].append(str(float(markup_parts[-4])))
                        dict_with_markup['y_center'].append(str(float(markup_parts[-3])))
                        dict_with_markup['width'].append(str(float(markup_parts[-2])))
                        dict_with_markup['height'].append(str(float(markup_parts[-1])))

                        stage: str = re.findall(
                            r'stage\_[0-9]+',
                            current_series.path,
                        )[0][6:]
                        dict_with_markup['stage'].append(stage)
                    else:
                        logger.warning(f'skipped b-box for an image: {current_series.path}')

        df_with_markup: pd.DataFrame = pd.DataFrame.from_dict(dict_with_markup)
        logger.info(f'{filename} size: {df_with_markup.shape[0]}')

        df_with_markup.to_csv(
            path_to_save_dir / f'{filename}.csv',
            index=False,
        )
    logger.success('markup matched with train/val/test split')


@click.command()
@click.option(
    '--path_to_dir_with_splits',
    type=click.Path(),
    help='The path to the directory with train/val/test splits',
)
@click.option('--path_to_markup_dir', type=click.Path(exists=True), help='The path to the markup directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
@click.option('--min_relative_size', default=0.01, help='Bounding box minimal relative height/width')
def run_matching(
    path_to_dir_with_splits: Path | str,
    path_to_markup_dir: Path | str,
    path_to_save_dir: Path | str,
    min_relative_size: float,
) -> None:
    path_to_dir_with_splits = Path(path_to_dir_with_splits).resolve()
    path_to_markup_dir = Path(path_to_markup_dir).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    splits: list[str] = glob(str(path_to_dir_with_splits / '*.csv'))

    add_markup(
        splits,
        path_to_save_dir,
        path_to_markup_dir,
        min_relative_size,
    )


if __name__ == '__main__':
    run_matching()  # pylint: disable=no-value-for-parameter
