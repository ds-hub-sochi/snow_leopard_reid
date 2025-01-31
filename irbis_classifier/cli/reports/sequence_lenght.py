from __future__ import annotations

from pathlib import Path

import click

from irbis_classifier.src.plots import create_sequence_length_histogram


@click.command()
@click.option('--path_to_data_dir', type=click.Path(exists=True), help='The path to the data directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
@click.option('--filename', type=str, help='Name of a file after saving')
@click.option('--max_sequence_length', type=int, help = 'sequences with length bigger then this will be combined')
def get_figure(
    path_to_data_dir: Path | str,
    path_to_save_dir: Path | str,
    filename: str,
    max_sequence_length: int,
):
    path_to_data_dir = Path(path_to_data_dir).resolve()
    path_to_save_dir = Path(path_to_save_dir).resolve()

    create_sequence_length_histogram(
        path_to_data_dir,
        False,
        True,
        path_to_save_dir / 'sequence_length_histogram',
        filename,
        max_sequence_length=max_sequence_length,
    )


if __name__ == '__main__':
    get_figure()  # pylint: disable=no-value-for-parameter