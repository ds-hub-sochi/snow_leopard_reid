from __future__ import annotations

import os
from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.plots import create_barplot_with_confidence_intervals
from irbis_classifier.src.testing.utils import test_model
from irbis_classifier.src.training.datasets import AnimalDataset
from irbis_classifier.src.training.transforms import val_transfroms


@click.command()
@click.option('--path_to_test_csv', type=click.Path(exists=True), help='path to csv-file with test data')
@click.option('--path_to_weight', type=click.Path(exists=True), help ="path to model's weights")
@click.option('--batch_size', type=int, help ="batch size to use; better be training batch size / number of gpus")
@click.option('--bootstrap_size', type=int, help='size of a bootstrapped sample')
@click.option('--alpha', type=float, help='required confidence level')
@click.option('--path_to_save_dir', type=click.Path(), help='The path to the data directory')
@click.option(
    '--path_to_unification_mapping_json',
    type=click.Path(exists=True),
    help='The path to the json file with unification mapping',
)
@click.option(
    '--path_to_supported_labels_json',
    type=click.Path(exists=True),
    help='The path to the json file with the list of supported labels',
)
@click.option(
    '--path_to_russian_to_english_mapping_json',
    type=click.Path(exists=True),
    help='The path to the json file with the russian to english mapping',
)
def run_testing(  # pylint: disable=too-many-positional-arguments
    path_to_test_csv: str | Path,
    path_to_weight: str | Path,
    batch_size: int,
    bootstrap_size: int,
    alpha: float,
    path_to_save_dir: str | Path,
    path_to_unification_mapping_json: Path | str,
    path_to_supported_labels_json: Path | str,
    path_to_russian_to_english_mapping_json: Path | str,
) -> None:
    path_to_test_csv = Path(path_to_test_csv).resolve()

    path_to_unification_mapping_json = Path(path_to_unification_mapping_json).resolve()
    path_to_supported_labels_json = Path(path_to_supported_labels_json).resolve()
    path_to_russian_to_english_mapping_json = Path(path_to_russian_to_english_mapping_json).resolve()

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json,
        path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json,
    )

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: nn.Module = models.efficientnet_b7(weights = models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        label_encoder.get_number_of_classes(),
    )
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_weight))
    model.eval()

    test_dataset: AnimalDataset = AnimalDataset(
        path_to_test_csv,
        val_transfroms,
    )

    test_dataloader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    f1_score_macro, test_metrics = test_model(
        test_dataloader,
        model,
        bootstrap_size,
        alpha,
    )

    logger.info(f'f1 macro = {f1_score_macro}')

    for label_index, metric in test_metrics.items():
        logger.info(
            f'{label_encoder.get_label_by_index(label_index)}: ' +  
            f'point estimations = {metric.point:.4f}, upper = {metric.upper:.4f}, lower = {metric.lower:.4f}',
        )

    create_barplot_with_confidence_intervals(
        f1_score_macro,
        test_metrics,
        1.0,
        False,
        True,
        path_to_save_dir,
        [label_encoder.get_label_by_index(i) for i in test_metrics],
    )


if __name__ == "__main__":
    run_testing()  # pylint: disable=no-value-for-parameter
