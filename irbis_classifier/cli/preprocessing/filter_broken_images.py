from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from PIL import Image

import click
import numpy as np
import pandas as pd
from loguru import logger

from joblib import Parallel, delayed


def process_serie(current_serie: pd.Series):
    try:
        np.asarray(Image.open(current_serie.path))
    except OSError:
        logger.error(f'found broken image {current_serie.path}')

        return None
    
    return current_serie.to_list()


@click.command()
@click.option('--path_to_data_dir', type=click.Path(exists=True), help='The path to the data directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
def main(
    path_to_data_dir: str | Path,
    path_to_save_dir: str | Path,
):
    logger.info('broken images filtering has started')

    path_to_data_dir = Path(path_to_data_dir).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    stages: list[str] = glob(str(path_to_data_dir / '*'))
    for stage in stages:
        stage_name = stage.split('/')[-1]
        
        logger.info(f'processing {stage_name}')
        
        df: pd.DataFrame = pd.read_csv(stage)

        filtered_lst = list(Parallel(n_jobs=os.cpu_count())(delayed(process_serie)(df.iloc[i]) for i in df.index))
        filtered_lst = [value for value in filtered_lst if value is not None]

        filtered_df = pd.DataFrame(filtered_lst, columns=df.columns)
        filtered_df.to_csv(path_to_save_dir / stage_name, index=False)

        logger.info(f'{stage_name} processed')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter