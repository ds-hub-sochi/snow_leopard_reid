import torch
from torchvision import models


class Factory:
    @staticmethod
    def get_model(model_name: str) -> torch.nn.Module:
        if hasattr(models, model_name.lower()):
            return getattr(models, model_name.lower())(weights='IMAGENET1K_V1')
