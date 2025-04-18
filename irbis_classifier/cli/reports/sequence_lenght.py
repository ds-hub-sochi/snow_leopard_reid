from __future__ import annotations

from pathlib import Path

import click

from irbis_classifier.src.plots import create_sequence_length_histogram_comparison


@click.command()
@click.option(
    '--path_to_data_dir_before',
    type=click.Path(exists=True),
    help='The path to the data directory before the resampling',
)
@click.option(
    '--path_to_data_dir_after',
    type=click.Path(exists=True),
    help='The path to the data directory after the resampling',
)
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
@click.option('--max_sequence_length', type=int, help='sequences with length bigger than this will be combined')
def main(
    path_to_data_dir_before: Path | str,
    path_to_data_dir_after: Path | str,
    path_to_save_dir: Path | str,
    max_sequence_length: int,
):
    path_to_data_dir_before = Path(path_to_data_dir_before).resolve()
    path_to_data_dir_after = Path(path_to_data_dir_after).resolve()
    path_to_save_dir = Path(path_to_save_dir).resolve()

    create_sequence_length_histogram_comparison(
        path_to_data_dir_before,
        path_to_data_dir_after,
        False,
        True,
        path_to_save_dir,
        max_sequence_length,
    )


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
