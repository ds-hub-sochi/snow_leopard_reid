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
from irbis_classifier.src.training.datasets import AnimalDataset
from irbis_classifier.src.training.transforms import get_val_transforms


@dataclass
class LabelEncoderParams:
    path_to_unification_mapping_json: str | Path
    path_to_supported_labels_json: str | Path
    path_to_russian_to_english_mapping_json: str | Path

    def __post_init__(self):
        self.path_to_unification_mapping_json = Path(self.path_to_unification_mapping_json).resolve()
        self.path_to_supported_labels_json = Path(self.path_to_supported_labels_json).resolve()
        self.path_to_russian_to_english_mapping_json = Path(self.path_to_russian_to_english_mapping_json).resolve()

@dataclass
class Normalization:
    mean: list[float]
    std: list[float]

@dataclass
class ImageResizing:
    size_before_padding: int
    size_after_padding: int

@dataclass
class ConfidenceIntervalParams:
    bootstrap_size: int
    alpha: float

@dataclass
class TestingConfig:
    path_to_test_csv: str | Path
    path_to_traced_model: str | Path
    path_to_save_dir: str | Path
    batch_size: int
    label_encoder_params: LabelEncoderParams
    normalization: Normalization
    image_resizing: ImageResizing
    metrics_to_use: list[str]
    confidence_interval_params: ConfidenceIntervalParams

    def __post_init__(self):
        self.path_to_test_csv = Path(self.path_to_test_csv).resolve()

        self.path_to_save_dir = Path(self.path_to_save_dir).resolve()
        self.path_to_save_dir = self.path_to_save_dir / self.path_to_traced_model[:-3]
        self.path_to_save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.label_encoder_params = LabelEncoderParams(**self.label_encoder_params)  # pylint: disable=not-a-mapping
        self.normalization = Normalization(**self.normalization)  # pylint: disable=not-a-mapping
        self.image_resizing = ImageResizing(**self.image_resizing)  # pylint: disable=not-a-mapping
        self.confidence_interval_params = ConfidenceIntervalParams(**self.confidence_interval_params)  # pylint: disable=not-a-mapping


@click.command()
@click.option(
    '--path_to_config',
    type=click.Path(exists=True),
    help='path to the json-config file',
)
def run_testing(
    path_to_config: str | Path,
) -> None:
    with open(
        path_to_config,
        'r',
        encoding='utf-8',
    ) as json_file:
        config: TestingConfig = TestingConfig(**json.load(json_file))

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json=config.label_encoder_params.path_to_russian_to_english_mapping_json,
        path_to_supported_classes_json=config.label_encoder_params.path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json=config.label_encoder_params.path_to_russian_to_english_mapping_json,
    )

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: torch.jit.ScriptModule = torch.jit.load(config.path_to_traced_model)
    model.to(device)
    model.eval()

    test_dataset: AnimalDataset = AnimalDataset(
        config.path_to_test_csv,
        get_val_transforms(
            mean=config.normalization.mean,
            std=config.normalization.std,
            max_size=config.image_resizing.size_before_padding,
            resize=config.image_resizing.size_after_padding,
        ),
    )

    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
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
        config.confidence_interval_params.bootstrap_size,
        config.confidence_interval_params.alpha,
    )

    for normalization in ('over actual', 'over predicted'):
        create_confusion_matrix(
            confusion_matrix,
            [label_encoder.get_label_by_index(i) for i in range(label_encoder.get_number_of_classes())],
            normalize=normalization,
            show=False,
            save=True,
            save_dir=config.path_to_save_dir,
            title=f'Confusion matrix\n(normalized {normalization})'
        )

    for current_metric_name, current_metrics_results in metrics_results.items():
        create_barplot_with_confidence_intervals(
            current_metrics_results[0],
            current_metrics_results[1],
            1.0,
            show=False,
            save=True,
            save_dir=config.path_to_save_dir,
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
        config.path_to_save_dir / 'metrics.json',
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
