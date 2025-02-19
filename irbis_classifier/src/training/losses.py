from __future__ import annotations

import inspect
import sys

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()

        self._weight: torch.Tensor | None = weight
        self._alpha: float = alpha
        self._gamma: float = gamma

        if reduction not in {'mean', 'none', 'sum'}:
            logger.error("error during FocalLoss creating: check your 'reduction' parameters")

            raise ValueError("'reduction' must be one of {'mean', 'none', 'sum'}")

        self._reduction: str = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss: torch.Tensor = F.cross_entropy(
            input=inputs,
            target=targets,
            weight=self._weight,
            reduction='none'
        )
        p_t: torch.Tensor = torch.exp(-ce_loss)

        loss_value: torch.Tensor = (self._alpha * (1 - p_t) ** self._gamma) * ce_loss

        if self._reduction == 'mean':
            return loss_value.mean()
        if self._reduction == 'sum':
            return loss_value.sum()

        return loss_value


class LossFactory:
    @staticmethod
    def get_loss(
        loss_name: str,
    ) -> type[torch.nn.Module]:
        if hasattr(torch.nn, loss_name):
            return getattr(
                torch.nn,
                loss_name,
            )
        
        for object_name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and object_name == loss_name:
                return obj

        raise ValueError(f"loss you've provided '{loss_name}' wasn't found")
