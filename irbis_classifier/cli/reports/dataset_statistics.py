from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt


def create_classes_pie_plot(
    df: pd.DataFrame,
    title: str,
    show: bool,
    filename: str | None = None,
) -> None:
    logger.info(f"Creatin pie plot for the {filename} split")

    unique_species: list[str] = sorted(list(set(df.specie)))

    with sns.color_palette("deep", len(unique_species)):
        labels = unique_species
        sizes = [df[df.specie == label].shape[0] for label in labels]

        _, ax = plt.subplots(figsize=(16, 9))
        ax.pie(sizes, labels=labels)
        plt.title(title)

        if show:
            plt.show()

        if filename is not None:
            repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

            save_dir: Path = repository_root_dir / 'reports' / 'figures' / 'classes_proportion'
            save_dir.mkdir(exist_ok=True, parents=True)

            plt.savefig(save_dir / f'{filename[:-4]}.png')


def create_pie_plots_over_split(
    show: bool,
    save: bool,
) -> None:
    logger.info('Pie plots over split creation is started')
    if not show and not save:
        logger.warning("Pie plot is cancelled due to False value in both 'save' and 'show' options")
        return

    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

    data_dir: Path = repository_root_dir / 'data' / 'processed'

    splits: list[str] = [f.path.split('/')[-1] for f in os.scandir(data_dir) if Path(data_dir).is_dir()]
    splits = [split for split in splits if split.endswith('csv')]

    english_label_to_russian: dict[str, str] = {
        'train': 'обучающей',
        'val': 'валидационной',
        'test': 'тестовой',
    }

    for split in splits:
        create_classes_pie_plot(
            pd.read_csv(data_dir / f'{split}'),
            f'Соотношение классов в {english_label_to_russian[split[:-4]]} выборке',
            show=show,
            filename=split if save else None,
        )

    logger.success("Pie plots are created")


def create_classes_difference_over_split(
    show: bool,
    save: bool,
) -> None:
    # based on
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    logger.info("Bar plot with classes difference creation is started")

    if not show and not save:
        logger.warning("Classes difference bar plot is cancelled due to False value in both 'save' and 'show' options")
        return

    repository_root_dir: Path = Path(__file__).parent.parent.parent.parent.resolve()

    data_dir: Path = repository_root_dir / 'data' / 'processed'

    splits: list[str] = [f.path.split('/')[-1] for f in os.scandir(data_dir) if Path(data_dir).is_dir()]
    splits = [split for split in splits if split.endswith('csv')]

    labels: set[str] = set()
    for split in splits:
        df: pd.DataFrame = pd.read_csv(data_dir / split)
        labels.update(set(df.specie))

    split_to_class_counts: defaultdict[str, list[int]] = defaultdict(list)

    for label in sorted(list(labels)):
        for split in splits:
            df = pd.read_csv(data_dir / split)
            split_to_class_counts[split[:-4]].append(df[df.specie == label].shape[0])

    x: np.ndarray = np.arange(len(labels))
    width: float = 0.25
    multiplier: int = 0

    with sns.color_palette("deep", len(labels)):
        _, ax = plt.subplots(layout='constrained', figsize=(len(labels), 8))

        for attribute, measurement in split_to_class_counts.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, zorder=3)
            ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Количество кропов')
        ax.set_title('Соотношение данных по классам в обучающей/валидационной/тестовой выборке')
        ax.set_xticks(x + width)
        ax.set_xticklabels(sorted(list(labels)), rotation=45)
        ax.legend(loc='upper left')

        ax.grid(linewidth=0.75, zorder=0)

        ax.grid(which="minor", linewidth=0.50, zorder=0)
        ax.minorticks_on()

        if show:
            plt.show()

        if save:
            save_dir: Path = repository_root_dir / 'reports' / 'figures' / 'classes_difference_over_split'
            save_dir.mkdir(exist_ok=True, parents=True)

            plt.savefig(save_dir / 'classes_difference.png')

    logger.success('Bar plot is created')


def collect_figures():
    create_pie_plots_over_split(False, True)
    create_classes_difference_over_split(False, True)


if __name__ == '__main__':
    collect_figures()
