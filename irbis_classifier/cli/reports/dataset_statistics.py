from __future__ import annotations

from pathlib import Path

import click

from irbis_classifier.src.plots import (
    create_pie_plots_over_split,
    create_classes_difference_bar_plot_over_split,
    create_classes_bar_plot_over_stages,
)


@click.command()
@click.option('--path_to_data_dir', type=click.Path(exists=True), help='The path to the data directory')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
def main(
    path_to_data_dir: Path | str,
    path_to_save_dir: Path | str,
):
    path_to_data_dir = Path(path_to_data_dir).resolve()
    path_to_save_dir = Path(path_to_save_dir).resolve()

    create_pie_plots_over_split(
        path_to_data_dir,
        False,
        True,
        path_to_save_dir / 'splits_classes_distribution',
    )

    create_classes_difference_bar_plot_over_split(
        path_to_data_dir,
        False,
        True,
        path_to_save_dir,
    )

    create_classes_bar_plot_over_stages(
        path_to_data_dir,
        False,
        True,
        path_to_save_dir,
    )


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
