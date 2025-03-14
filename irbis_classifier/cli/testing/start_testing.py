from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from loguru import logger
from torch.utils.data import DataLoader

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.plots import create_barplot_with_confidence_intervals, create_confusion_matrix
from irbis_classifier.src.testing.test import test_model
from irbis_classifier.src.testing.testers import MetricsEstimations
from irbis_classifier.src.training.datasets import AnimalDataset
from irbis_classifier.src.training.transforms import get_val_transforms


@dataclass
class MetricsJson:
    averaged_f1_macro: MetricsEstimations
    f1_for_classes: dict[str, MetricsEstimations]


@click.command()
@click.option(
    '--path_to_test_csv',
    type=click.Path(exists=True),
    help='path to csv-file with test data',
)
@click.option(
    '--path_to_traced_model',
    type=click.Path(exists=True),
    help ="path to traced model dump",
)
@click.option(
    '--batch_size',
    type=int,
    help ="batch size to use; better be training batch size / number of gpus")
@click.option(
    '--bootstrap_size',
    type=int,
    help='size of a bootstrapped sample',
)
@click.option(
    '--alpha',
    type=float, 
    help='required confidence level',
)
@click.option(
    '--path_to_save_dir',
    type=click.Path(), 
    help='the path to the data directory',
)
@click.option(
    '--path_to_unification_mapping_json',
    type=click.Path(exists=True),
    help='the path to the json file with unification mapping',
)
@click.option(
    '--path_to_supported_labels_json',
    type=click.Path(exists=True),
    help='the path to the json file with the list of supported labels',
)
@click.option(
    '--path_to_russian_to_english_mapping_json',
    type=click.Path(exists=True),
    help='the path to the json file with the russian to english mapping',
)
@click.option(
    '--mean',
    type=str,
    default='0.485,0.456,0.406',
    help='normalization mean; better see model description for the proper values',
)
@click.option(
    '--std',
    type=str,
    default='0.229,0.224,0.225',
    help='normalization standart deviation; better see model description for the proper values',
)
@click.option(
    '--max_size',
    type=int,
    default=256,
    help='max image size; bigger side will be resized to this size',
)
@click.option(
    '--resize',
    type=int,
    default=224,
    help='after the padding applied image will be resized to this size',
)
def run_testing(  # pylint: disable=too-many-positional-arguments,too-many-arguments,too-many-locals
    path_to_test_csv: str | Path,
    path_to_traced_model: str,
    batch_size: int,
    bootstrap_size: int,
    alpha: float,
    path_to_save_dir: str | Path,
    path_to_unification_mapping_json: Path | str,
    path_to_supported_labels_json: Path | str,
    path_to_russian_to_english_mapping_json: Path | str,
    mean='0.485,0.456,0.406',
    std='0.229,0.224,0.225',
    max_size: int = 256,
    resize: int = 224,
) -> None:
    path_to_test_csv = Path(path_to_test_csv).resolve()

    path_to_save_dir = Path(path_to_save_dir).resolve()
    path_to_save_dir = path_to_save_dir / path_to_traced_model[:-3]
    path_to_save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path_to_unification_mapping_json = Path(path_to_unification_mapping_json).resolve()
    path_to_supported_labels_json = Path(path_to_supported_labels_json).resolve()
    path_to_russian_to_english_mapping_json = Path(path_to_russian_to_english_mapping_json).resolve()

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json,
        path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json,
    )

    try:
        mean_lst: list[float] = [float(value) for value in mean.split(',')]
        std_lst: list[float] = [float(value) for value in std.split(',')]
    except ValueError:
        logger.error('check mean and std you have passed; it most be a comma separated string like "0.1,0.2,0.3"')

        return

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: torch.jit.ScriptModule = torch.jit.load(path_to_traced_model)
    model.to(device)
    model.eval()

    test_dataset: AnimalDataset = AnimalDataset(
        path_to_test_csv,
        get_val_transforms(
            mean=mean_lst,
            std=std_lst,
            max_size=max_size,
            resize=resize,
        ),
    )

    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    metrics: list[str] = [
        'f1_score',
        'precision_score',
        'recall_score',
    ]

    metrics_results, confusion_matrix = test_model(
        test_dataloader,
        model,
        metrics,
        bootstrap_size,
        alpha,
    )

    for normalization in ('over actual', 'over predicted'):
        create_confusion_matrix(
            confusion_matrix,
            [label_encoder.get_label_by_index(i) for i in range(label_encoder.get_number_of_classes())],
            normalize=normalization,
            show=False,
            save=True,
            save_dir=path_to_save_dir,
            title=f'Confusion matrix\n(normalized {normalization})'
        )

    for current_metric_name, current_metrics_results in metrics_results.items():
        create_barplot_with_confidence_intervals(
            current_metrics_results[0],
            current_metrics_results[1],
            1.0,
            show=False,
            save=True,
            save_dir=path_to_save_dir,
            metric_name=current_metric_name,
            labels=[label_encoder.get_label_by_index(i) for i in current_metrics_results[1]],
        )

        for index in list(current_metrics_results[1].keys()):
            current_metrics_results[1][label_encoder.get_english_label_by_index(index)] = current_metrics_results[1][index]
            del current_metrics_results[1][index]

        logger.info(
            f'{current_metric_name} macro: point estimations: {current_metrics_results[0].point:.4f}, ' +
            f'lower: {current_metrics_results[0].lower:.4f}, upper: {current_metrics_results[0].upper:.4f}' 
        )

        for label, metric in current_metrics_results[1].items():
            logger.info(
                f'{current_metric_name} for {label}: point estimations: {metric.point:.4f}, ' +
                f'lower: {metric.lower:.4f}, upper: {metric.upper:.4f}',
            )

    with open(
        path_to_save_dir / 'metrics.json',
        'w',
        encoding='utf-8',
    ) as metrics_file:
        json.dump(
            metrics_results,
            metrics_file,
            default=vars,
        )

if __name__ == "__main__":
    run_testing()  # pylint: disable=no-value-for-parameter
