from __future__ import annotations

import timm
import torch
from torchvision import models

from irbis_classifier.src.models.utils import replace_last_linear


class Factory:
    """
    Factory that can initilize model and load pre-trained weights from torchvision or timm. Last linear layer will
    be changed to brand new with chosen number of classes.
    """
    @staticmethod
    def build_model(
        model_name: str,
        n_classes: int | None,
    ) -> torch.nn.Module | None:
        """
        A main method that creates and initialize model

        Args:
            model_name (str): name of the model you want to use; for example, resnet50 or timm/resnet50.a1_in1k
            n_classes (int | None): number of classes new last linear layer will have

        Raises:
            ValueError: raised if provided model_name wasn't found in both torchvision and timm hubs

        Returns:
            torch.nn.Module | None: a model with pretrained weights and replaced last linear layer
        """
        model: torch.nn.Module | None = None
        if hasattr(models, model_name.lower()):
            model: torch.nn.Module = getattr(models, model_name.lower())(weights='IMAGENET1K_V1')
        else:
            try:
                model = timm.create_model(
                    model_name=model_name,
                    pretrained=True,
                )
            except RuntimeError:
                model = None

        if model is not None:
            if n_classes is not None:
                return replace_last_linear(
                    module=model,
                    n_classes=n_classes,
                )

            return model

        raise ValueError("check the model name you've provided: it wasn't found")
