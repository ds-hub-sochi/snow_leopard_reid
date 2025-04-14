from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchvision import models

from irbis_classifier.src.models.utils import replace_last_linear


def test_only_linear():
    model: nn.Module = nn.Linear(10, 5)
    model_modified: nn.Module = replace_last_linear(
        model,
        10,
    )
    
    assert isinstance(model_modified, nn.Linear)
    assert model_modified.out_features == 10
    assert model_modified.out_features != model.out_features


def test_sequential_with_2_linear():
    model: nn.Module = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    model_modified: nn.Module = replace_last_linear(
        model, 10,
    )

    assert isinstance(model_modified[2], nn.Linear)
    assert model_modified[2].out_features == 10
    assert model_modified[2].out_features != model[2].out_features

    assert isinstance(model_modified[0], nn.Linear)
    assert model_modified[0].out_features == model[0].out_features


def test_model_with_nested_sequential_layers():
    model: nn.Module = nn.Sequential(
        nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Linear(15, 20),
            nn.ReLU()
        ),
        nn.Linear(20, 5)
    )

    model_modified: nn.Module = replace_last_linear(
        model, 10,
    )

    assert isinstance(model_modified[2], nn.Linear)
    assert model_modified[2].out_features == 10
    assert model_modified[2].out_features != model[2].out_features

    assert isinstance(model_modified[0][0], nn.Linear)
    assert model_modified[0][0].out_features == model[0][0].out_features

    assert isinstance(model_modified[1][0], nn.Linear)
    assert model_modified[1][0].out_features == model[1][0].out_features


def test_model_with_no_linear_layer():
    model: nn.Module = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(6, 12, kernel_size=3),
        nn.Sigmoid()
    )

    model_modified: nn.Module = replace_last_linear(model, 10)

    for p1, p2 in zip(model.parameters(), model_modified.parameters()):
        assert np.allclose(p1.data.ne(p2.data).sum().item(), 0)


def test_custom_class_with_nested_sequential():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.layer1: nn.Module = nn.Sequential(nn.Linear(5, 15), nn.ReLU())
            self.layer2: nn.Module = nn.Linear(15, 20)
            self.layer3: nn.Module = nn.Sequential(nn.ReLU(), nn.Linear(20, 5))

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            return x

    model: nn.Module = MyModel()
    model_modified: nn.Module = replace_last_linear(model, 10)

    assert isinstance(model_modified.layer3[1], nn.Linear)
    assert model_modified.layer3[1].out_features == 10
    assert model_modified.layer3[1].out_features != model.layer3[1].out_features

    assert isinstance(model_modified.layer2, nn.Linear)
    assert model_modified.layer2.out_features == model.layer2.out_features

    assert isinstance(model_modified.layer1[0], nn.Linear)
    assert model_modified.layer1[0].out_features == model.layer1[0].out_features


def test_model_with_named_sequential_layers():
    model: nn.Module = nn.Sequential(
        OrderedDict(
            [
                ('linear1', nn.Linear(10, 20)),
                ('relu', nn.ReLU()),
                ('linear2', nn.Linear(20, 5)),
            ]
        )
    )

    model_modified: nn.Module = replace_last_linear(model, 10)

    assert isinstance(model_modified.linear2, nn.Linear)
    assert model_modified.linear2.out_features == 10
    assert model_modified.linear2.out_features != model.linear2.out_features

    assert isinstance(model_modified.linear1, nn.Linear)
    assert model_modified.linear1.out_features == model.linear1.out_features


def test_model_with_inner_sequentials_with_named_layers():
    model: nn.Module = nn.Sequential(
        nn.Sequential
            (OrderedDict(
                [
                    ('linear1', nn.Linear(10, 20)),
                    ('relu', nn.ReLU()),
                    ('linear2', nn.Linear(20, 5)),
                ]
            )
        ),
        nn.Sequential(
            OrderedDict(
                [
                    ('linear1', nn.Linear(10, 20)),
                    ('relu', nn.ReLU()),
                    ('linear2', nn.Linear(20, 5)),
                ]
            )
        ),
    )

    model_modified: nn.Module = replace_last_linear(model, 10)

    assert isinstance(model_modified[1].linear2, nn.Linear)
    assert model_modified[1].linear2.out_features == 10
    assert model_modified[1].linear2.out_features != model[1].linear2.out_features

    assert isinstance(model_modified[1].linear1, nn.Linear)
    assert model_modified[1].linear1.out_features == model[1].linear1.out_features

    assert isinstance(model_modified[0].linear1, nn.Linear)
    assert model_modified[0].linear1.out_features == model[0].linear1.out_features

    assert isinstance(model_modified[0].linear2, nn.Linear)
    assert model_modified[0].linear2.out_features == model[0].linear2.out_features


def test_custom_class_with_multiple_linear_layers():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.fc1: nn.Module = nn.Linear(10, 20)
            self.fc2: nn.Module = nn.Linear(20, 30)
            self.seq: nn.Module = nn.Sequential(
                nn.Linear(30, 40),
                nn.ReLU(),
                nn.Linear(40, 50)
            )

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.seq(x)

            return x
        
    model: nn.Module = MyModel()

    model_modified: nn.Module = replace_last_linear(model, 10)

    assert isinstance(model_modified.seq[2], nn.Linear)
    assert model_modified.seq[2].out_features == 10
    assert model_modified.seq[2].out_features != model.seq[2].out_features

    assert isinstance(model_modified.seq[0], nn.Linear)
    assert model_modified.seq[0].out_features == model.seq[0].out_features

    assert isinstance(model_modified.fc1, nn.Linear)
    assert model_modified.fc1.out_features == model.fc1.out_features

    assert isinstance(model_modified.fc2, nn.Linear)
    assert model_modified.fc2.out_features == model.fc2.out_features


def test_real_convnext():
    model: nn.Module = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model_modified: nn.Module = replace_last_linear(model, 10)

    assert isinstance(model_modified.classifier[2], nn.Linear)
    assert model_modified.classifier[2].out_features == 10
    assert model_modified.classifier[2].out_features != model.classifier[2].out_features

    for p1, p2 in zip(list(model.parameters())[:-2], list(model_modified.parameters())[:-2]):
        assert np.allclose(p1.data.ne(p2.data).sum().item(), 0)
