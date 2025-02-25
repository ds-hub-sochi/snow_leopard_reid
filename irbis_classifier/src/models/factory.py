from __future__ import annotations

import timm
import torch
from torchvision import models

from irbis_classifier.src.models.utils import replace_last_linear


class Factory:
    @staticmethod
    def build_model(
        model_name: str,
        n_classes: int | None,
    ) -> torch.nn.Module | None:
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
