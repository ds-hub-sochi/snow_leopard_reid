from __future__ import annotations

import os
from pathlib import Path

import click
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.models.factory import Factory
from irbis_classifier.src.plots import create_barplot_with_confidence_intervals
from irbis_classifier.src.testing.utils import test_model
from irbis_classifier.src.training.datasets import AnimalDataset
from irbis_classifier.src.training.transforms import get_val_transforms


@click.command()
@click.option('--path_to_test_csv', type=click.Path(exists=True), help='path to csv-file with test data')
@click.option('--model_name', type=str, help ="model name")
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
@click.option(
    '--mean',
    type=str,
    default="0.485,0.456,0.406",
    help='normalization mean',
)
@click.option(
    '--std',
    type=str,
    default='0.229,0.224,0.225',
    help='normalization mean',
)
@click.option(
    '--max_size',
    type=int,
    default=256,
)
@click.option(
    '--resize',
    type=int,
    default=224,
)
def run_testing(  # pylint: disable=too-many-positional-arguments,too-many-arguments
    path_to_test_csv: str | Path,
    model_name: str,
    path_to_weight: str | Path,
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

    model: nn.Module = Factory.build_model(
        model_name,
        label_encoder.get_number_of_classes(),
    )
    model = model.to(device)
    model.load_state_dict(torch.load(path_to_weight))
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
