import numpy as np
import pytest
import torch
from torch import nn
from torchvision import models

from irbis_classifier.src.models.factory import Factory


testdata = (
    ('EfficientNet_B7', models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)),
    ('ConvNeXt_Large', models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)),
    ('Swin_V2_B', models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)),
)


@pytest.mark.parametrize('model_name,model_handwritten', testdata)
def test_torchvision_models_weights_download(
    model_name: str,
    model_handwritten: nn.Module,
):
    model_from_factory: nn.Module = Factory().build_model(model_name, None)

    model_from_factory_named_layers: dict[str, nn.Module] = dict(model_from_factory.named_modules())
    model_handwritten_named_layers: dict[str, nn.Module] = dict(model_handwritten.named_modules())

    for name, layer in model_from_factory_named_layers.items():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            print(layer.bias)
            assert torch.allclose(
                layer.weight,
                model_handwritten_named_layers[name].weight,
            )
            if layer.bias is not None:
                assert np.allclose(
                    layer.bias.detach().numpy(),
                    model_handwritten_named_layers[name].bias.detach().numpy(),
                )
            else:
                assert model_handwritten_named_layers[name].bias is None
