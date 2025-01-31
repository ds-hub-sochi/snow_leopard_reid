from __future__ import annotations

import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt, rcParams

SMALL_SIZE: int = 12
MEDIUM_SIZE: int = 16
BIGGER_SIZE: int = 20

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
rcParams.update({'figure.autolayout': True})


def create_classes_pie_plot(
    df: pd.DataFrame,
    title: str,
    show: bool,
    filename: Path | None,
) -> None:
    unique_species: list[str] = sorted(list(set(df.specie)))

    with sns.color_palette(
        'deep',
        len(unique_species),
    ):
        labels = unique_species
        sizes = [df[df.specie == label].shape[0] for label in labels]

        _, ax = plt.subplots(figsize=(16, 9))
        ax.pie(
            sizes,
            labels=labels,
        )
        plt.title(
            title,
            fontdict={'fontsize': BIGGER_SIZE},
        )

        if show:
            plt.show()

        if filename is not None:
            plt.savefig(f'{str(filename)[:-4]}.png')


def create_pie_plots_over_split(
    data_dir: Path | str,
    show: bool,
    save: bool,
    save_dir: Path | str | None,
) -> None:
    logger.info('pie plots over split creation is started')
    if not show and not save:
        logger.warning("pie plot is cancelled due to False value in both 'save' and 'show' options")
        return

    data_dir = Path(data_dir).resolve()

    splits: list[str] = [f.path.split('/')[-1] for f in os.scandir(data_dir) if data_dir.is_dir()]
    splits = [split for split in splits if split.endswith('csv')]

    if save and save_dir is not None:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

    for split in splits:
        create_classes_pie_plot(
            pd.read_csv(data_dir / split),
            f'Соотношение классов в {split[:-4]} выборке',
            show=show,
            filename=save_dir / split if (save and save_dir is not None) else None,  # type: ignore
        )

    logger.success('pie plots were created')


def create_classes_difference_bar_plot_over_split(
    data_dir: Path | str,
    show: bool,
    save: bool,
    save_dir: Path | str | None,
) -> None:
    # based on
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    logger.info('classes differnce bar plot creation was started')

    if not show and not save:
        logger.warning("classes difference bar plot is cancelled due to False value in both 'save' and 'show' options")
        return

    data_dir = Path(data_dir).resolve()

    splits: list[str] = [f.path.split('/')[-1] for f in os.scandir(data_dir) if data_dir.is_dir()]
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

    with sns.color_palette(
        'deep',
        len(labels),
    ):
        _, ax = plt.subplots(
            layout='constrained',
            figsize=(
                len(labels),
                8,
            ),
        )

        for attribute, measurement in split_to_class_counts.items():
            offset = width * multiplier
            rects = ax.bar(
                x + offset,
                measurement,
                width,
                label=attribute,
                zorder=3,
            )
            ax.bar_label(
                rects,
                padding=3,
            )
            multiplier += 1

        ax.set_title(
            'Соотношение данных по видам в выборках',
            fontdict={'fontsize': BIGGER_SIZE},
        )
        ax.set_ylabel('Количество кропов')
        ax.set_xlabel('Вид')

        ax.set_xticks(x + width)
        ax.set_xticklabels(
            sorted(list(labels)),
            rotation=45,
        )

        ax.legend(loc='upper left')

        ax.grid(
            linewidth=0.75,
            zorder=0,
        )
        ax.grid(
            which='minor',
            linewidth=0.50,
            zorder=0,
        )
        ax.minorticks_on()

        if show:
            plt.show()

        if save and save_dir is not None:
            save_dir = Path(save_dir).resolve()
            save_dir.mkdir(
                exist_ok=True,
                parents=True,
            )

            plt.savefig(
                save_dir / 'classes_difference.png',
                bbox_inches='tight',
            )

    logger.success('classes difference bar plot was created')


def create_classes_bar_plot_over_stages(
    data_dir: Path | str,
    show: bool,
    save: bool,
    save_dir: Path | str | None,
) -> None:
    # based ob
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
    logger.info('stacked bar plot with stages split creation was started')

    if not show and not save:
        logger.warning("classes difference bar plot is cancelled due to False value in both 'save' and 'show' options")
        return

    data_dir = Path(data_dir).resolve()

    splits: list[pd.DataFrame] = [pd.read_csv(path) for path in glob(str(data_dir / '*'))]
    cumulative_df: pd.DataFrame = pd.concat(
        splits,
        ignore_index=True,
        axis=0,
    )

    species: list[str] = sorted(list(set(cumulative_df.specie)))
    stages: list[int] = sorted([int(stage) for stage in list(set(cumulative_df.stage))])

    weight_counts: dict[str, np.ndarray] = {}
    with sns.color_palette(
        'deep',
        len(stages),
    ):
        for stage in stages:
            current_stage_weights: list[int] = []
            stage_subdf: pd.DataFrame = cumulative_df[cumulative_df.stage == stage]
            for specie in species:
                current_stage_weights.append(stage_subdf[stage_subdf.specie == specie].shape[0])

            weight_counts[f'stage_{stage}'] = np.array(current_stage_weights)

        _, ax = plt.subplots(figsize=(len(species), 8))
        bottom = np.zeros(len(species))

        for stage, weight_count in weight_counts.items():  # type: ignore
            _ = ax.bar(
                species,
                weight_count,
                0.5,
                label=stage,
                bottom=bottom,
                zorder=3,
            )
            bottom += weight_count

        ax.set_title(
            "Распределение количества кропов для вида по stage'ам",
            fontdict={'fontsize': BIGGER_SIZE},
        )
        ax.set_ylabel('Количество кропов')
        ax.set_xlabel('Вид')

        ax.set_xticks(np.arange(len(species)))
        ax.set_xticklabels(
            species,
            rotation=45,
        )

        ax.legend(loc='upper right')

        ax.grid(
            linewidth=0.75,
            zorder=0,
        )
        ax.grid(
            which='minor',
            linewidth=0.50,
            zorder=0,
        )
        ax.minorticks_on()

        if show:
            plt.show()

        if save and save_dir is not None:
            save_dir = Path(save_dir).resolve()
            save_dir.mkdir(
                exist_ok=True,
                parents=True,
            )

            plt.savefig(save_dir / 'stage_stacked_barplot.png')

        logger.success('stacked bar plot was created')


def create_sequence_length_histogram(  # pylint: ignore=too-many-positional-arguments
    data_dir: Path | str,
    show: bool,
    save: bool,
    save_dir: Path | str | None,
    filename: str | None,
    max_sequence_length: int = 150,
) -> None:
    logger.info('sequence length histogram creation was started')

    if not show and not save:
        logger.warning("sequence length histogram is cancelled due to False value in both 'save' and 'show' options")
        return

    data_dir = Path(data_dir).resolve()

    length_to_count: defaultdict[int, int] = defaultdict(int)

    stages: list[str] = [path for path in glob(str(data_dir / '*')) if Path(path).suffix == '.csv']
    for stage in stages:
        df: pd.DataFrame = pd.read_csv(stage)
        unique_sequences: list[str] = list(set(df.sequence))

        for sequence in unique_sequences:
            sequence_length: int = df[df.sequence == sequence].shape[0]
            if sequence_length > max_sequence_length:
                length_to_count[max_sequence_length] += 1
            else:
                length_to_count[sequence_length] += 1

    with sns.color_palette(
        'deep',
    ):
        sns.histplot(
            x=list(length_to_count.keys()),
            weights=list(length_to_count.values()),
            bins=max_sequence_length,
        )

        ax = plt.gca()

        ax.grid(
            linewidth=0.75,
            zorder=0,
        )
        ax.grid(
            which='minor',
            linewidth=0.50,
            zorder=0,
        )
        ax.minorticks_on()

        if show:
            plt.show()

        if save and save_dir is not None:
            save_dir = Path(save_dir).resolve()
            save_dir.mkdir(
                exist_ok=True,
                parents=True,
            )

            plt.savefig(save_dir / filename)  # type: ignore

    logger.success('sequence length histogram was created')
