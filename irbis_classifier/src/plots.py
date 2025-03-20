from __future__ import annotations

import os
from collections import defaultdict
from glob import glob
from pathlib import Path
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt, rcParams
from mpl_toolkits.axes_grid1 import ImageGrid

from irbis_classifier.src.testing.test import MetricsEstimations


warnings.filterwarnings('ignore')

EXTRA_SMALL_SIZE: int = 8
SMALL_SIZE: int = 12
MEDIUM_SIZE: int = 16
BIGGER_SIZE: int = 20
LARGE_SIZE: int = 34
EXTREMELY_SUPER_LARGE_SIZE: int = 40

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
rcParams.update(
    {
        'figure.autolayout': True,
    },
)


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


def create_classes_difference_bar_plot_over_split(  # pylint: disable=too-many-locals
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

        paddings: list[int] = (20, 20, 5)

        for (attribute, measurement), padding in zip(split_to_class_counts.items(), paddings):
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
                padding=padding,
                fontsize=SMALL_SIZE,
                label_type='edge',
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


def create_sequence_length_histogram_comparison(  # pylint: disable=too-many-positional-arguments,too-many-locals
    data_dir_before: Path | str,
    data_dir_after: Path | str,
    show: bool,
    save: bool,
    save_dir: Path | str | None,
    max_sequence_length: int = 150,
) -> None:
    logger.info('sequence length histogram creation was started')

    if not show and not save:
        logger.warning("sequence length histogram is cancelled due to False value in both 'save' and 'show' options")
        return

    data_dir_before = Path(data_dir_before).resolve()
    data_dir_after = Path(data_dir_after).resolve()

    data_list: list[tuple[int, int, str]] = []

    for data_dir, label in (
        (data_dir_before, 'before resampling'),
        (data_dir_after, 'after resampling'),
    ):

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

        for length, count in length_to_count.items():
            data_list.append((length, count, label))

    with sns.color_palette(
        'deep',
    ):
        ax = sns.histplot(
            pd.DataFrame(
                data_list,
                columns=(
                    'length',
                    'count',
                    'label',
                ),
            ),
            x='length',
            weights='count',
            hue='label',
            bins=max_sequence_length,
        )

    plt.title('Сравнение гистограмм длин серий')
    plt.xlabel('Длина серии')
    plt.ylabel('Количество серий данной длины')

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

    proper_length: int = len(ax.patches) // 2  # cause two histograms were created

    for i in range(proper_length):
        difference: int = ax.patches[i].get_height() - ax.patches[proper_length + i].get_height()
        ax.annotate(
            f'{"+" if difference > 0 else ""}{difference if difference != 0 else ""}\n',
            (
                ax.patches[proper_length + i].get_x() + ax.patches[proper_length + i].get_width() / 2,
                ax.patches[proper_length + i].get_height(),
            ),
            ha='center',
            va='center',
            color='red' if difference < 0 else 'blue',
            fontweight='bold',
            fontsize=8,
            rotation=30,
        )

    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[-2] = f'{max_sequence_length}+'
    ax.set_xticklabels(labels)

    ax.set_yscale('log')

    if show:
        plt.show()

    if save and save_dir is not None:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        plt.savefig(save_dir / 'sequence_length_histograms_comparison.png')  # type: ignore

    logger.success('sequence length histogram was created')


def create_image_grid(
    images: list[np.ndarray],
    titles: list[str],
    show: bool,
    save: bool,
    save_dir: Path | str | None,
) -> None:
    logger.info('Image grid creation has started')
    assert len(images) == len(titles), \
        logger.error(f'Number of images {len(images)} != number of titles {len(titles)}')

    n_samples: int = len(images)

    assert float(int(np.sqrt(n_samples))**2 == n_samples), \
        logger.error('Number of samples must be a square, like 36 or 25')

    figure: plt.Figure = plt.figure(figsize=(n_samples, n_samples))
    grid: ImageGrid = ImageGrid(
        figure,
        111,
        nrows_ncols=(
            int(np.sqrt(n_samples)),
            int(np.sqrt(n_samples)),
        ),
        axes_pad=0.5,
    )

    for ax, image, title in zip(grid, images, titles):  # type: ignore
        ax.imshow(image)
        ax.set_title(
            title,
            fontsize=24,
        )
        ax.set_axis_off()

    if show:
        plt.show()

    if save and save_dir is not None:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        plt.savefig(save_dir / 'augmented_image_variants.png')  # type: ignore

    logger.success('Image grid was created')


def create_barplot_with_confidence_intervals(  # pylint: disable=too-many-positional-arguments
    f1_score_macro: MetricsEstimations,
    metrics: dict[int, MetricsEstimations],
    metric_max_value: float,
    show: bool,
    save: bool,
    save_dir: str | Path | None,
    metric_name: str | None,
    labels: list[str],
):
    logger.info(f'barplot with the value of {metric_name} over classes creation has started')

    with sns.color_palette(
        'deep',
        len(metrics.keys()),
    ):
        _ = plt.figure(figsize=(len(metrics) * 2, len(metrics)))

        plt.axhline(
            y=metric_max_value,
            zorder=0,
            color='red',
        )

        _ = plt.bar(
            x=list(metrics.keys()),
            height=[metrics[key].point for key in metrics],
            width=0.4,
            color='red',
            align='center',
            alpha=0.5,
            zorder=3,
        )

        plt.errorbar(
            x=list(metrics.keys()),
            y=[metrics[key].point for key in metrics],
            yerr=np.array(
                [
                    (
                        max(
                            metrics[key].point - metrics[key].lower,
                            0 - metrics[key].lower,
                        ),
                        min(
                            metrics[key].upper - metrics[key].point,
                            1 - metrics[key].point,
                        ),
                    ) for key in metrics
                ],
            ).T,
            linestyle='',
            elinewidth=16,
            zorder=4,
        )

        ax = plt.gca()
        ax.set_axisbelow(True)

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

        plt.xticks(
            range(len(list(metrics.keys()))),
            labels,
            rotation=45,
            fontsize=LARGE_SIZE,
        )

        plt.yticks(
            fontsize=EXTREMELY_SUPER_LARGE_SIZE,
        )

        plt.title(
            f'Value of the {metric_name} with confidence intervals \n point estimate = {f1_score_macro.point:.3f}',
            fontsize=EXTREMELY_SUPER_LARGE_SIZE,
        )

        if show:
            plt.show()

    if save and save_dir is not None:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(
            exist_ok=True,
            parents=True,
        )

        plt.savefig(save_dir / f'{metric_name}_over_classes.png')  # type: ignore

    logger.success(f'barplot with the value of {metric_name} over classes was created')


def create_confusion_matrix(  # pylint: disable=too-many-positional-arguments
    confision_matrix: list[list[int]] | npt.NDArray[np.float64],
    labels: list[str] | tuple[str],
    normalize: str | None,
    show: bool,
    save: bool,
    save_dir: str | Path | None,
    title: str | None = None,
) -> None:
    logger.info('confusion matrix creation has started')

    assert normalize in {'over actual', 'over predicted'}, '"normalize" must be either "over_actual" or "over predicted"'

    if normalize == 'over actual':
        confision_matrix = np.array(
            confision_matrix,
            dtype=np.float32,
        )
        sum_over_actual = np.sum(
            confision_matrix,
            axis=1,
        ).reshape(-1, 1)
        confision_matrix /= sum_over_actual

    elif normalize == 'over predicted':
        confision_matrix = np.array(
            confision_matrix,
            dtype=np.float32,
        )
        sum_over_predicted = np.sum(
            confision_matrix,
            axis=0,
        ).reshape(1, -1)
        confision_matrix /= sum_over_predicted

    if normalize is not None:
        def _round(arg: np.float32) -> np.float32:
            return round(arg, 3)

        _round_vectorized = np.vectorize(_round)

        confision_matrix = _round_vectorized(confision_matrix)  # pylint: disable=redefined-variable-type

    labels = labels[:confision_matrix.shape[0]]

    confusion_matrix_as_df: pd.DataFrame = pd.DataFrame(
        confision_matrix,
        index=labels,
        columns=labels,
    )

    plt.figure(figsize=(len(labels), len(labels)))

    axes: plt.axes = sns.heatmap(
        confusion_matrix_as_df,
        annot=True,
        annot_kws={
            'size': MEDIUM_SIZE,
        }
    )

    axes.set_xlabel(
        'Predicted',
        fontsize=LARGE_SIZE,
    )
    axes.set_xticklabels(
        labels,
        rotation=90,
        fontsize=BIGGER_SIZE,
    )

    axes.set_ylabel(
        'Actual',
        fontsize=LARGE_SIZE,
    )
    axes.set_yticklabels(
        labels,
        fontsize=BIGGER_SIZE,
    )

    plt.title(
        'Confusion matrix' if title is None else title,
        fontsize=LARGE_SIZE,
    )

    if show:
        plt.show()

    if save and save_dir is not None:
        save_dir = Path(save_dir).resolve()
        save_dir.mkdir(
            exist_ok=True,
            parents=True,
        )
        if title is None:
            plt.savefig(save_dir / 'confusion_matrix.png')  # type: ignore
        else:
            title = title.lower().replace(' ', '_').replace('\n', '_')
            plt.savefig(save_dir / f'{title}.png')  # type: ignore

    logger.success('confusion matrix was created')
