from __future__ import annotations

import inspect
import sys

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn


class FocalLoss(nn.Module):
    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        weight: torch.Tensor | None,
        label_smoothing: float = 0.0,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()

        self.weight: torch.Tensor | None = weight
        self.label_smoothing: float = label_smoothing
        self.alpha: float = alpha
        self.gamma: float = gamma

        if reduction not in {'mean', 'none', 'sum'}:
            logger.error("error during FocalLoss creating: check your 'reduction' parameters")

            raise ValueError("'reduction' must be one of {'mean', 'none', 'sum'}")

        self.reduction: str = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        ce_loss: torch.Tensor = F.cross_entropy(
            input=inputs,
            target=targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        p_t: torch.Tensor = torch.exp(-ce_loss)

        loss_value: torch.Tensor = (self.alpha * (1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss_value.mean()
        if self.reduction == 'sum':
            return loss_value.sum()

        return loss_value


class LossFactory:
    """
    A factory that creates a loss instance by its string name
    """
    def _get_loss_class(
        self,
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
    
    def build_loss_funcion(
        self,
        loss_name: str,
        **kwargs,
    ) -> torch.nn.Module:
        try:
            loss_cls: type[torch.nn.Module] = self._get_loss_class(loss_name)
        except ValueError as error:
            raise error
        
        if 'weight' in kwargs and loss_cls in [torch.nn.MultiMarginLoss, FocalLoss]:
            kwargs['weight'] *= kwargs['n_classes']

        init_signature = inspect.signature(loss_cls.__init__)
    
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in init_signature.parameters
        }

        return loss_cls(**filtered_kwargs)
