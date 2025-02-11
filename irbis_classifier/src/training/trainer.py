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
from irbis_classifier.src.utils import create_confusion_matrix


@dataclass
class Logs:
    loss: float


@dataclass
class EvalLogs(Logs):
    f1_score_macro: float
    confusion_matrix: list[list[int]]


class TrainerInterface(ABC):
    def __init__(
        self,
        path_to_checkpoints_dir: str | Path,
        bigger_is_better: bool = True,
    ):
        self._bigger_is_better: bool = bigger_is_better

        if bigger_is_better:
            self._checkpoint_metric: float = 0.0
        else:
            self._checkpoint_metric = float('inf')

        self._path_to_checkpoints_dir: Path = Path(path_to_checkpoints_dir).resolve()

    @abstractmethod
    def train(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.optimizer.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scaler: torch.optim.GradScaler,
        n_epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        experiment: comet_ml.CometExperiment,
        model_label: str,
        label_encoder: LabelEncoder,
    ) -> None:
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
    ) -> Logs:
        pass

    @abstractmethod
    @torch.inference_mode()
    def _evaluation_step(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Logs:
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
        pass

    @abstractmethod
    def _saving_step(
        self,
        model: nn.Module,
        metric_value: float,
        model_label: str,
    ) -> None:
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

    def train(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.optimizer.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scaler: torch.optim.GradScaler,
        n_epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        experiment: comet_ml.CometExperiment,
        model_label: str,
        label_encoder: LabelEncoder,
    ) -> None:
        logger.info(f'training during {n_epochs} epochs has started')

        label: str = ''

        for epoch in tqdm(range(n_epochs)):
            train_logs: Logs = self._training_step(
                model,
                optimizer,
                criterion,
                scaler,
                train_dataloader,
                device,
            )

            label = 'train'
            self._logging_step(
                train_logs,
                label,
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

            label = 'val'
            self._logging_step(
                val_logs,
                label,
                experiment,
                epoch,
                label_encoder,
            )

            self._saving_step(
                model,
                val_logs.f1_score_macro,
                model_label,
            )

        logger.success('training has ended')

    def _training_step(  # pylint: disable=too-many-positional-arguments
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scaler: torch.optim.GradScaler,
        train_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        model.train()

        running_loss: list[float] = []

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

            running_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return Logs(
            loss=float(np.mean(running_loss)),
        )

    @torch.inference_mode()
    def _evaluation_step(
        self,
        model: torch.nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        if isinstance(logs, EvalLogs):
            experiment.log_confusion_matrix(
                matrix=logs.confusion_matrix,
                step=epoch,
                max_categories=len(logs.confusion_matrix),
                max_examples_per_cell=max(max(logs.confusion_matrix[i]) for i in range(len(logs.confusion_matrix))),
                labels=[label_encoder.get_label_by_index(i) for i in range(len(logs.confusion_matrix))],
            )

            delattr(logs, 'confusion_matrix')

        for key, value in logs.__dict__.items():
            experiment.log_metric(
                f'{label}/{key}',
                value,
                epoch,
            )
            logger.info(f'{label}/{key}: {value}')

    def _saving_step(
        self,
        model: nn.Module,
        metric_value: float,
        model_label: str,
    ) -> None:
        if self._bigger_is_better == (metric_value > self._checkpoint_metric):
            self._checkpoint_metric = metric_value

            if isinstance(model, nn.DataParallel):
                torch.save(
                    model.module.state_dict(),
                    self._path_to_checkpoints_dir / f'{model_label}_best_model.pth',
                )
            else:
                torch.save(
                    model.state_dict(),
                    self._path_to_checkpoints_dir / f'{model_label}_best_model.pth',
                )

        if isinstance(model, nn.DataParallel):
            torch.save(
                model.module.state_dict(),
                self._path_to_checkpoints_dir / f'{model_label}_last_model.pth',
            )
        else:
            torch.save(
                model.state_dict(),
                self._path_to_checkpoints_dir / f'{model_label}_last_model.pth',
            )
