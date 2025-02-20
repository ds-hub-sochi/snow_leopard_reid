from __future__ import annotations

import torch
from torchvision import models


class Factory:
    @staticmethod
    def get_model(
        model_name: str,
        n_classes: int | None,     
    ) -> torch.nn.Module | None:
        model: torch.nn.Module | None = None
        if hasattr(models, model_name.lower()):
            model: torch.nn.Module = getattr(models, model_name.lower())(weights='IMAGENET1K_V1')

        if model is not None:
            if n_classes is not None:
                if hasattr(model, 'classifier'):
                    classifier_layers: list[torch.nn.Module] = list(model.classifier.modules())

                    for i in range(len(classifier_layers) - 1, -1, -1):
                        # print(classifier_layers[i])
                        if isinstance(classifier_layers[i], torch.nn.Linear):
                            classifier_layers[i] = torch.nn.Linear(
                                classifier_layers[i].in_features,
                                n_classes,
                            )

                            model.classifier = torch.nn.Sequential(
                                *classifier_layers,
                            )

                            return model
                        
        raise ValueError("check the model name you've provided; it wasn't found")
