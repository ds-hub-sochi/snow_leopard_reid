"""https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/ - полезная дока"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import comet_ml
import dotenv
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
class TrainingParams:
    batch_size: int
    n_epochs: int
    lr: float
    use_scheduler: bool
    use_ema_model: bool
    warmup_epochs: int
    gradient_accumulation_steps: int = 1


@dataclass
class LossParams:
    name: str
    use_weights: bool = False
    label_smoothing: float = True


@dataclass
class TrainingConfig:  # pylint: disable=too-many-instance-attributes
    path_to_data_dir: str | Path
    path_to_checkpoints_dir: str | Path
    path_to_experiment_config: str | Path
    model_name: str
    label_encoder_params: LabelEncoderParams
    normalization: Normalization
    image_resizing: ImageResizing
    trainig_params: TrainingParams
    loss_params: LossParams
    device_ids: list[int]
    additional_run_info: str = ""
    run_name: str = ""

    def __post_init__(self):
        self.path_to_data_dir = Path(self.path_to_data_dir).resolve()

        self.path_to_experiment_config = Path(self.path_to_experiment_config).resolve()

        self.label_encoder_params = LabelEncoderParams(**self.label_encoder_params)  # pylint: disable=not-a-mapping
        self.normalization = Normalization(**self.normalization)  # pylint: disable=not-a-mapping
        self.image_resizing = ImageResizing(**self.image_resizing)  # pylint: disable=not-a-mapping
        self.trainig_params = TrainingParams(**self.trainig_params)  # pylint: disable=not-a-mapping
        self.loss_params = LossParams(**self.loss_params)  # pylint: disable=not-a-mapping

        run_name_parts: list[str] = [self.model_name, self.loss_params.name]

        if self.loss_params.label_smoothing != 0.0:
            run_name_parts.append('smoothing')
        
        if self.loss_params.use_weights:
            run_name_parts.append('weights')

        if self.trainig_params.use_scheduler:
            run_name_parts.append('scheduler')

        if self.trainig_params.warmup_epochs != 0:
            run_name_parts.append('warmup')

        run_name_parts.append('gradient_accumulation_steps')
        run_name_parts.append(str(self.trainig_params.gradient_accumulation_steps))

        if self.additional_run_info != '':
            run_name_parts.append(self.additional_run_info)

        self.run_name: str = '_'.join(run_name_parts)

        current_date: str = datetime.today().strftime('%Y-%m-%d')

        self.path_to_checkpoints_dir = Path(self.path_to_checkpoints_dir).resolve()
        self.path_to_checkpoints_dir = self.path_to_checkpoints_dir / self.run_name / current_date
        self.path_to_checkpoints_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


@click.command()
@click.option(
    '--path_to_config',
    type=click.Path(exists=True),
    help='path to the json-config file',
)
def start_training(  # pylint: disable=too-many-statements,too-many-locals
    path_to_config: str | Path,
):
    with open(
        path_to_config,
        'r',
        encoding='utf-8',
    ) as json_file:
        config: TrainingConfig = TrainingConfig(**json.load(json_file))

    repository_root_dir: Path  = Path(__file__).parent.parent.parent.parent.resolve()

    if not os.path.exists(str(repository_root_dir / '.env')):
        os.mknod(str(repository_root_dir / '.env'))

    dotenv_file: str = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)

    dotenv.set_key(
        dotenv_file,
        'LAST_RUN',
        config.run_name,
    )
    dotenv.set_key(
        dotenv_file,
        'TRAIN_BATCH_SIZE',
        str(config.trainig_params.batch_size),
    )
    dotenv.set_key(
        dotenv_file,
        'TRAIN_MAX_SIZE',
        str(config.image_resizing.size_before_padding),
    )
    dotenv.set_key(
        dotenv_file,
        'TRAIN_RESIZE',
        str(config.image_resizing.size_after_padding),
    )
    dotenv.set_key(
        dotenv_file,
        'RUN_DATE',
        datetime.today().strftime('%Y-%m-%d')
    )

    label_encoder: LabelEncoder = create_label_encoder(
        path_to_unification_mapping_json=config.label_encoder_params.path_to_russian_to_english_mapping_json,
        path_to_supported_classes_json=config.label_encoder_params.path_to_supported_labels_json,
        path_to_russian_to_english_mapping_json=config.label_encoder_params.path_to_russian_to_english_mapping_json,
    )

    dotenv.set_key(
        dotenv_file,
        'TRAIN_GPU_COUNT',
        str(len(config.device_ids)),
    )

    dotenv.set_key(
        dotenv_file,
        'TRAIN_MEAN',
        ','.join([str(item) for item in config.normalization.mean]),
    )
    dotenv.set_key(
        dotenv_file,
        'TRAIN_STD',
        ','.join([str(item) for item in config.normalization.std]),
    )

    try:
        experiment: comet_ml.CometExperiment = setup_experimet(
            config.path_to_experiment_config,
            config.run_name,
        )
    except TypeError as error:
        logger.error(f"error during experimet setup; please check your config's fields: {error}")

        return

    train_dataset, val_dataset, _ = create_train_val_test_datasets(
        config.path_to_data_dir,
        train_transforms=get_train_transforms(
            mean=config.normalization.mean,
            std=config.normalization.std,
            max_size_before_padding=config.image_resizing.size_before_padding,
            max_size_after_padding=config.image_resizing.size_after_padding,
        ),
        val_transforms=get_val_transforms(
            mean=config.normalization.mean,
            std=config.normalization.std,
            max_size_before_padding=config.image_resizing.size_before_padding,
            max_size_after_padding=config.image_resizing.size_after_padding,
        ),
    )

    train_dataloader: DataLoader = DataLoader(
        train_dataset,
        batch_size=config.trainig_params.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
    )

    val_dataloader: DataLoader = DataLoader(
        val_dataset,
        batch_size=config.trainig_params.batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    device: torch.device = torch.device(f'cuda:{config.device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    try:
        model: nn.Module = Factory.build_model(
            config.model_name,
            label_encoder.get_number_of_classes(),
        )
    except ValueError as error:
        logger.error(f'error duting model creating: {error}')
        experiment.end()

        return

    model = nn.DataParallel(
        model,
        device_ids=config.device_ids,
    )
    model = model.to(device)

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config.trainig_params.lr,
    )

    if config.trainig_params.use_scheduler:
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.trainig_params.n_epochs,
        )
    else:
        scheduler = None

    if config.trainig_params.warmup_epochs > 0:
        warmup_scheduler: LinearWarmupLR | None = LinearWarmupLR(
            optimizer,
            config.trainig_params.warmup_epochs,
            target_lr=config.trainig_params.lr,
            initial_lr=1e-8,
        )
    else:
        warmup_scheduler = None

    kwargs = {
        'n_classes': label_encoder.get_number_of_classes(),
        'label_smoothing': config.loss_params.label_smoothing,
    }

    if config.loss_params.use_weights:
        weight: torch.Tensor = get_classes_weights(
            get_classes_counts(
                config.path_to_data_dir / 'train.csv',
                label_encoder.get_number_of_classes(),
            )
        ).to(device)
        kwargs['weight'] = weight

    try:
        criterion: torch.nn.Module = LossFactory().build_loss_funcion(
            loss_name=config.loss_params.name,
            **kwargs,
        )
    except ValueError as error:
        logger.error(f'error during loss creating: {error}')
        experiment.end()

        return

    scaler: torch.amp.GradScaler = torch.amp.GradScaler()

    parameters = {
        'batch_size': config.trainig_params.batch_size,
        'initial lerning rate': config.trainig_params.lr,
        'architecture': model.module.__class__.__name__,
    }

    experiment.log_parameters(parameters)

    trainer: Trainer = Trainer(
        config.path_to_checkpoints_dir,
        bigger_is_better=True,
    )

    trainer.train(
        model,
        optimizer,
        scheduler,
        warmup_scheduler,
        criterion,
        scaler,
        config.trainig_params.n_epochs,
        train_dataloader,
        val_dataloader,
        device,
        experiment,
        label_encoder,
        config.trainig_params.use_ema_model,
        config.trainig_params.gradient_accumulation_steps,
    )

    experiment.log_model(
        config.run_name,
        str(config.path_to_checkpoints_dir / 'model_best.pth'),
    )

    experiment.log_model(
        config.run_name,
        str(config.path_to_checkpoints_dir / 'model_last.pth'),
    )

    if config.trainig_params.use_ema_model:
        experiment.log_model(
            config.run_name,
            str(config.path_to_checkpoints_dir / 'ema_model_best.pth'),
        )
        experiment.log_model(
            config.run_name,
            str(config.path_to_checkpoints_dir / 'ema_model_last.pth'),
        )

    experiment.end()


if __name__ == "__main__":
    start_training()  # pylint: disable=no-value-for-parameter
