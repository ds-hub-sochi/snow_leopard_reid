from __future__ import annotations

import json
from glob import glob
from loguru import logger
from pathlib import Path
from typing import Any

import click
import pandas as pd

from irbis_classifier.src.utils import sample_from_dataframe


@click.command()
@click.option('--path_to_data_dir', type=click.Path(exists=True), help='The path to the data directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
@click.option(
    '--classes_to_sample_json',
    type=click.Path(exists=True),
    help='json with list of classes that can be sampled',
)
@click.option('--max_sequence_length', type=int, help='sequences with length bigger then this will be combined')
@click.option('--resample_size', type=float, help='size of sequence after resampling. Absolute or a fraction')
def resample_stages(
    path_to_data_dir: Path | str,
    path_to_save_dir: Path | str,
    classes_to_sample_json: Path | str,
    max_sequence_length: int,
    resample_size: int | float,
):
    logger.info('long series resampling has started')

    path_to_data_dir = Path(path_to_data_dir).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    with open(
        classes_to_sample_json,
        'r',
        encoding='utf-8',
    ) as jf:
        classes_to_sample: set[str] = set(json.load(jf))

    stages: list[list] = [path for path in glob(str(path_to_data_dir / '*')) if Path(path).suffix == '.csv']
    for stage in stages:
        stage_name: str = stage.split('/')[-1]

        logger.info(f'processing {stage_name}')

        df: pd.DataFrame = pd.read_csv(stage)
        logger.info(f'dataframe has {df.shape[0]} values')

        unique_sequences: list[str] = list(set(df.sequence))

        new_df_content: list[tuple[Any, ...]] = []

        for sequence in unique_sequences:
            subset_df = df[df.sequence == sequence]
            sequence_length: int = subset_df.shape[0]

            if (subset_df.specie.iloc[0] in classes_to_sample) and (sequence_length > max_sequence_length):
                if resample_size <= 1.0:
                    resampled_df: pd.DataFrame = sample_from_dataframe(
                        subset_df,
                        round(resample_size * sequence_length),
                    )
                else:
                    resampled_df = sample_from_dataframe(
                        subset_df,
                        int(resample_size),
                    )

                new_df_content.extend(list(resampled_df.itertuples(index=False, name=None)))
            else:
                new_df_content.extend(list(subset_df.itertuples(index=False, name=None)))

        resampled_stage_df: pd.DataFrame = pd.DataFrame(
            new_df_content,
            columns=df.columns,
        )

        resampled_stage_df.to_csv(path_to_save_dir / stage_name)
        logger.info(f'dataframe now has {resampled_stage_df.shape[0]} values')
        logger.success(f'{stage_name} processed')


if __name__ == '__main__':
    resample_stages()  # pylint: disable=no-value-for-parameter
