"""https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/ - полезная дока"""

from __future__ import annotations

import os
import random
from datetime import datetime
from pathlib import Path

import click
import comet_ml
import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader

from irbis_classifier.src.label_encoder import create_label_encoder, LabelEncoder
from irbis_classifier.src.models.factory import Factory
from irbis_classifier.src.training import (
    create_train_val_test_datasets,
    setup_experimet,
    Trainer,
    get_train_transforms,
    get_val_transforms,
)
from irbis_classifier.src.training.losses import LossFactory
from irbis_classifier.src.training.warmup_schedulers import LinearWarmupLR
from irbis_classifier.src.training.weights import get_classes_counts, get_classes_weights


torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


@click.command()
@click.option(
    '--path_to_data_dir',
    type=click.Path(exists=True),
    help='path to dir with train/val/test csv files',
)
@click.option(
    '--path_to_checkpoints_dir',
    type=click.Path(exists=True),
    help='path to dir where model checkpoint will be stored',
)
@click.option(
    '--path_to_experiment_config',
    type=click.Path(exists=True),
    help='path to experoment config json file',
)
@click.option(
    '--model_name',
    type=str,
    help='model you want to use',
)
@click.option(
    '--run_name',
    type=str,
    help='name of a run in the comet reports',
)
@click.option(
    '--batch_size',
    type=int,
    help='batch size you want to use; please note that DataParallel is used',
)
@click.option(
    '--n_epochs',
    type=int,
    help='the duration of training in epochs',
)
@click.option(
    '--lr',
    type=float,
    help='learning rate you want to setup for your optimize; exponential way is OK',
)
@click.option(
    '--device_ids',
    type=str,
    help='ids of the devices you want to as a comma separated string; ex. "0,1"',
)
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
@click.option(
    '--warmup_epochs',
    type=int,
    default=None,
    help="how many warmup epochs must be used",
)
@click.option(
    '--use_weighted_loss',
    type=bool,
    default=False,
    help="use classes's weight to compute loss or not",
)
@click.option(
    '--loss',
    type=str,
    default='CrossEntropyLoss',
    help='which loss to use; for example, CrossEntropyLoss',
)
@click.option(
    '--label_smoothing',
    type=float,
    default=0.0,
    help='label smoothing value',
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
def start_training(  # pylint: disable=too-many-positional-arguments,too-many-locals,too-many-arguments,too-many-statements
    path_to_data_dir: str | Path,
    path_to_checkpoints_dir: str | Path,
    path_to_experiment_config: str | Path,
    model_name: str,
    run_name: str,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device_ids: str,
    path_to_unification_mapping_json: Path | str,
    path_to_supported_labels_json: Path | str,
    path_to_russian_to_english_mapping_json: Path | str,
    use_scheduler: bool = True,
    warmup_epochs: int | None = None,
    use_weighted_loss: bool = False,
    loss: str = 'CrossEntropyLoss',
    label_smoothing: float = 0.0,
    mean='0.485,0.456,0.406',
    std='0.229,0.224,0.225',
    max_size: int = 256,
    resize: int = 224,
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
        logger.error('check device_ids you have passed; it must be a comma separated string like "0,1"')

        return
    
    try:
        mean_lst: list[float] = [float(value) for value in mean.split(',')]
        std_lst: list[float] = [float(value) for value in std.split(',')]
    except ValueError:
        logger.error('check mean and std you have passed; it most be a comma separated string like "0.1,0.2,0.3"')

        return

    try:
        experiment: comet_ml.CometExperiment = setup_experimet(
            path_to_experiment_config,
            run_name,
        )
    except TypeError as error:
        logger.error(f"error during experimet setup; please check your config's fields: {error}")

        return

    train_dataset, val_dataset, _ = create_train_val_test_datasets(
        path_to_data_dir,
        train_transforms=get_train_transforms(
            mean=mean_lst,
            std=std_lst,
            max_size=max_size,
            resize=resize,
        ),
        val_transforms=get_val_transforms(
            mean=mean_lst,
            std=std_lst,
            max_size=max_size,
            resize=resize,
        ),
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

    device: torch.device = torch.device(f'cuda:{device_ids_list[0]}' if torch.cuda.is_available() else 'cpu')

    try:
        model: nn.Module = Factory.build_model(
            model_name,
            label_encoder.get_number_of_classes(),
        )
    except ValueError as error:
        logger.error(f'error duting model creating: {error}')
        experiment.end()

        return

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

    if warmup_epochs is not None:
        warmup_scheduler: LinearWarmupLR | None = LinearWarmupLR(
            optimizer,
            warmup_epochs,
            target_lr=lr,
            initial_lr=1e-8,
        )
    else:
        warmup_scheduler = None

    kwargs = {
        'n_classes': label_encoder.get_number_of_classes(),
        'label_smoothing': label_smoothing,
    }

    if use_weighted_loss:
        weight: torch.Tensor = get_classes_weights(
            get_classes_counts(
                path_to_data_dir / 'train.csv',
                label_encoder.get_number_of_classes(),
            )
        ).to(device)
        kwargs['weight'] = weight

    try:
        criterion: torch.nn.Module = LossFactory().build_loss_funcion(
            loss_name=loss,
            **kwargs,
        )
    except ValueError as error:
        logger.error(f'error during loss creating: {error}')
        experiment.end()

        return

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
        warmup_scheduler,
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

    experiment.log_model(
        run_name,
        str(path_to_checkpoints_dir / 'last_model.pth'),
    )

    experiment.end()


if __name__ == "__main__":
    start_training()  # pylint: disable=no-value-for-parameter
