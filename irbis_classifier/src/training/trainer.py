from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import comet_ml
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score
from torch import autocast
from torch import nn
from tqdm import tqdm

from irbis_classifier.src.label_encoder import LabelEncoder
from irbis_classifier.src.testing.metrics import create_confusion_matrix
from irbis_classifier.src.training import warmup_schedulers


@dataclass
class Logs:
    loss: float


@dataclass
class EvalLogs(Logs):
    f1_score_macro: float
    confusion_matrix: list[list[int]]


class TrainerInterface(ABC):
    """
    A trainer class that trains provided model
    """
    def __init__(
        self,
        path_to_checkpoints_dir: str | Path,
        bigger_is_better: bool = True,
    ):
        self._bigger_is_better: bool = bigger_is_better

        if bigger_is_better:
            self._model_checkpoint_metric: float = 0.0
        else:
            self._model_checkpoint_metric = float('inf')

        self._path_to_checkpoints_dir: Path = Path(path_to_checkpoints_dir).resolve()

    @abstractmethod
    def train(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.optimizer.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        warmup_scheduler: warmup_schedulers.LinearWarmupLR | None,
        criterion: nn.Module,
        scaler: torch.optim.GradScaler,
        n_epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        experiment: comet_ml.CometExperiment,
        label_encoder: LabelEncoder,
        use_ema_model: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        """
        A base function for the model training. Call all the necessary steps inside;

        Args:
            model (torch.nn.Module): model you want to train
            optimizer (torch.optim.optimizer.Optimizer): optimizer to use
            scheduler (torch.optim.lr_scheduler.LRScheduler | None): learning rate scheduler to use
            warmup_scheduler (warmup_schedulers.LinearWarmupLR | None): warmup scheduler to use
            criterion (nn.Module): loss function to use
            scaler (torch.optim.GradScaler): mixed precision scaler
            n_epochs (int): number of training epochs
            train_dataloader (torch.utils.data.DataLoader): train dataloader
            val_dataloader (torch.utils.data.DataLoader): validation dataloader
            device (torch.device): devicec to use
            experiment (comet_ml.CometExperiment): Comet wrapper for logging
            label_encoder (LabelEncoder): LabelEncoder instance that helps to work with labels properly
            use_ema_model (bool, optional): will EMA model be trained or not. Defaults to False.
            gradient_accumulation_steps (int, optional): how much gradient accumulation steps will be used. Defaults to 1.
        """
        pass

    @abstractmethod
    def _training_step(  # pylint: disable=too-many-positional-arguments
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.optimizer.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scaler: torch.optim.GradScaler,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        ema_model: None | torch.optim.swa_utils.AveragedModel,
        gradient_accumulation_steps: int,
    ) -> Logs:
        """
        Used for the training
        """
        pass

    @abstractmethod
    @torch.inference_mode()
    def _evaluation_step(
        self,
        model: torch.nn.Module,
        criterion: nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Logs:
        """
        Used for the validation
        """
        pass

    @abstractmethod
    def _logging_step(  # pylint: disable=too-many-positional-arguments
        self,
        logs: dict[str, float],
        label: str,
        experiment: comet_ml.CometExperiment,
        epoch: int,
        label_encoder: LabelEncoder,
    ) -> None:
        """
        Used for the logging
        """
        pass

    @abstractmethod
    def _saving_step(
        self,
        model: nn.Module,
        metric_value: float,
        model_postfix: str,
    ) -> None:
        """
        Used to save model checkpoints
        """
        pass


class Trainer(TrainerInterface):
    def __init__(  # pylint: disable=useless-parent-delegation
        self,
        path_to_checkpoints_dir: str | Path,
        bigger_is_better: bool = True,
    ):
        super().__init__(
            path_to_checkpoints_dir,
            bigger_is_better,
        )

    def train(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        warmup_scheduler: warmup_schedulers.LinearWarmupLR | None,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scaler: torch.optim.GradScaler,
        n_epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        experiment: comet_ml.CometExperiment,
        label_encoder: LabelEncoder,
        use_ema_model: bool = False,
        gradient_accumulation_steps: int = 1,
    ) -> None: 
        if use_ema_model:
            if self._bigger_is_better:
                self._ema_model_checkpoint_metric: float = 0.0
            else:
                self._ema_model_checkpoint_metric = float('inf')

            ema_model: torch.optim.swa_utils.AveragedModel | None = torch.optim.swa_utils.AveragedModel(
                model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999),
            )
        else:
            ema_model = None

        logger.info(f'training during {n_epochs} epochs has started')

        for epoch in tqdm(range(n_epochs)):
            train_logs: Logs = self._training_step(
                model,
                optimizer,
                criterion,
                scaler,
                train_dataloader,
                device,
                ema_model,
                gradient_accumulation_steps,
            )

            if warmup_scheduler is not None and warmup_scheduler.warmup_epochs > epoch:
                warmup_scheduler.step()
            elif scheduler is not None:
                scheduler.step()

            self._logging_step(
                train_logs,
                'train',
                experiment,
                epoch,
                label_encoder,
            )

            val_logs: EvalLogs = self._evaluation_step(
                model,
                criterion,
                val_dataloader,
                device,
            )

            self._logging_step(
                val_logs,
                'val',
                experiment,
                epoch,
                label_encoder,
            )

            self._saving_step(
                model,
                val_logs.f1_score_macro,
                'model',
            )

            if ema_model is not None:
                ema_val_logs: EvalLogs = self._evaluation_step(
                    ema_model,
                    criterion,
                    val_dataloader,
                    device,
                )

                self._logging_step(
                    ema_val_logs,
                    'val_ema',
                    experiment,
                    epoch,
                    label_encoder,
                )

                self._saving_step(
                    ema_model.module,
                    ema_val_logs.f1_score_macro,
                    'ema_model',
                )

        if ema_model is not None:
            torch.optim.swa_utils.update_bn(
                train_dataloader,
                ema_model,
            )

        logger.success('training has ended')

    def _training_step(  # pylint: disable=too-many-positional-arguments
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: torch.optim.GradScaler,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        ema_model: None | torch.optim.swa_utils.AveragedModel,
        gradient_accumulation_steps: int,
    ) -> dict[str, float]:
        model.train()

        running_loss: list[float] = []

        step: int = 0

        for input_batch, targets in tqdm(train_dataloader):
            input_batch = input_batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast(
                device_type='cuda',
                dtype=torch.float16,
            ):
                predicted_logits: torch.Tensor = model(input_batch)
                loss: torch.Tensor = criterion(
                    predicted_logits,
                    targets,
                )
                loss /= gradient_accumulation_steps

            running_loss.append(loss.item())

            scaler.scale(loss).backward()

            step += 1

            if step == gradient_accumulation_steps:
                step = 0

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if ema_model is not None:
                    ema_model.update_parameters(model)

        if step != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if ema_model is not None:
                ema_model.update_parameters(model)

        return Logs(
            loss=float(np.mean(running_loss)),
        )

    @torch.inference_mode()
    def _evaluation_step(
        self,
        model: torch.nn.Module,
        criterion: nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> EvalLogs:
        model.eval()

        running_loss: list[float] = []

        targets_lst: list[int] = []
        predicted_labels_lst: list[int] = []

        with torch.inference_mode():
            for input_batch, targets in tqdm(val_dataloader):
                input_batch = input_batch.to(device)
                targets = targets.to(device)

                with autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                ):
                    predicted_logits = model(input_batch)
                    loss = criterion(
                        predicted_logits,
                        targets,
                    )

                running_loss.append(loss.item())

                predicted_labels = torch.argmax(
                    predicted_logits,
                    dim=1,
                )

                predicted_labels_lst.extend(predicted_labels.tolist())
                targets_lst.extend(targets.tolist())

        f_score_macro: float = f1_score(
            y_true=targets_lst,
            y_pred=predicted_labels_lst,
            average='macro',
        )

        return EvalLogs(
            loss=float(np.mean(running_loss)),
            f1_score_macro=f_score_macro,
            confusion_matrix=create_confusion_matrix(
                y_true=targets_lst,
                y_predicted=predicted_labels_lst,
            )
        )

    def _logging_step(  # pylint: disable=too-many-positional-arguments
        self,
        logs: Logs,
        label: str,
        experiment: comet_ml.CometExperiment,
        epoch: int,
        label_encoder: LabelEncoder,
    ) -> None:
        if isinstance(
            logs,
            EvalLogs,
        ):
            experiment.log_confusion_matrix(
                matrix=logs.confusion_matrix,
                step=epoch,
                max_categories=len(logs.confusion_matrix),
                max_examples_per_cell=max(max(logs.confusion_matrix[i]) for i in range(len(logs.confusion_matrix))),
                labels=[label_encoder.get_label_by_index(i) for i in range(len(logs.confusion_matrix))],
            )

            delattr(
                logs,
                'confusion_matrix',
            )

        for key, value in logs.__dict__.items():
            experiment.log_metric(
                f'{label}/{key}',
                value,
                epoch,
            )
            logger.info(f'{label}/{key}: {value}')

    def _saving_step(
        self,
        model,
        metric_value: float,
        model_postfix: str,
    ):
        if self._bigger_is_better == (metric_value > getattr(self, f'_{model_postfix}_checkpoint_metric')):
            setattr(self, f'_{model_postfix}_checkpoint_metric', metric_value)

            if isinstance(
                model,
                nn.DataParallel,
            ):
                torch.save(
                    model.module.state_dict(),
                    self._path_to_checkpoints_dir / f'{model_postfix}_best.pth',
                )
            else:
                torch.save(
                    model.state_dict(),
                    self._path_to_checkpoints_dir / f'{model_postfix}_best.pth',
                )

        if isinstance(
            model,
            nn.DataParallel,
        ):
            torch.save(
                model.module.state_dict(),
                self._path_to_checkpoints_dir / f'{model_postfix}_last.pth',
            )
        else:
            torch.save(
                model.state_dict(),
                self._path_to_checkpoints_dir / f'{model_postfix}_last.pth',
            )
