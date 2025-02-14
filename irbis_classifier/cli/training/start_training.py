"""https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/ - полезная дока"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import click
import comet_ml
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.training import (
    create_train_val_test_datasets,
    setup_experimet,
    Trainer,
    train_transforms,
    val_transfroms,
)

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


@click.command()
@click.option('--path_to_data_dir', type=click.Path(exists=True), help='path to dir with train/val/test csv files')
@click.option(
    '--path_to_checkpoints_dir',
    type=click.Path(exists=True),
    help='path to dir where model checkpoint will be stored',
)
@click.option('--path_to_experiment_config', type=click.Path(exists=True), help='path to experoment config json file')
@click.option('--run_name', type=str, help='name of a run in the comet reports')
@click.option('--batch_size', type=int, help='batch size you want to use; please note that DataParallel is used')
@click.option('--n_epochs', type=int, help='the duration of training in epochs')
@click.option('--lr', type=float, help='learning rate you want to setup for your optimize; exponential way is OK')
@click.option('--device_ids', type=str, help='ids of the devices you want to as a comma separated string; ex. "0,1"')
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
    '--use_scheduler',
    type=bool,
    default=True,
    help="should learning scheduler be used during trainig or not",
)
def start_training(  # pylint: disable=too-many-positional-arguments,too-many-locals,too-many-arguments
    path_to_data_dir: str | Path,
    path_to_checkpoints_dir: str | Path,
    path_to_experiment_config: str | Path,
    run_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device_ids: str,
    path_to_unification_mapping_json: Path | str,
    path_to_supported_labels_json: Path | str,
    path_to_russian_to_english_mapping_json: Path | str,
    use_scheduler: bool = True,
):
    path_to_data_dir = Path(path_to_data_dir).resolve()

    path_to_unification_mapping_json = Path(path_to_unification_mapping_json).resolve()
    path_to_supported_labels_json = Path(path_to_supported_labels_json).resolve()
    path_to_russian_to_english_mapping_json = Path(path_to_russian_to_english_mapping_json).resolve()

    path_to_checkpoints_dir = Path(path_to_checkpoints_dir).resolve()
    path_to_checkpoints_dir = path_to_checkpoints_dir / run_name / datetime.today().strftime('%Y-%m-%d')
    path_to_checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json,
        path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json,
    )

    try:
        device_ids_list: list[int] = [int(id) for id in device_ids.split(',')]
    except ValueError:
        logger.error('check device_ids you have passed; it must be a comma separated string')

        return

    try:
        experiment: comet_ml.CometExperiment = setup_experimet(
            path_to_experiment_config,
            run_name,
        )
    except TypeError as e:
        logger.error(f"error during experimet setup; please check your config's fields: {e}")

        return

    train_dataset, val_dataset, _ = create_train_val_test_datasets(
        path_to_data_dir,
        train_transforms=train_transforms,
        val_transforms=val_transfroms,
    )

    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model: nn.Module = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        26,
    )
    model = nn.DataParallel(
        model,
        device_ids=device_ids_list,
    )
    model = model.to(device)

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
    )

    if use_scheduler:
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_epochs,
        )
    else:
        scheduler = None

    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss()

    scaler: torch.amp.GradScaler = torch.amp.GradScaler()

    parameters = {
        'batch_size': batch_size,
        'initial lerning rate': lr,
        'architecture': model.module.__class__.__name__,
    }

    experiment.log_parameters(parameters)

    trainer: Trainer = Trainer(
        path_to_checkpoints_dir,
        bigger_is_better=True,
    )

    trainer.train(
        model,
        optimizer,
        scheduler,
        criterion,
        scaler,
        n_epochs,
        train_dataloader,
        val_dataloader,
        device,
        experiment,
        label_encoder,
    )

    experiment.log_model(
        run_name,
        str(path_to_checkpoints_dir / 'best_model.pth'),
    )

    experiment.end()


if __name__ == "__main__":
    start_training()  # pylint: disable=no-value-for-parameter
