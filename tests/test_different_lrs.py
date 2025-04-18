from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn

from irbis_classifier.src.models.utils import get_last_linear


def test():
    model: nn.Module = nn.Sequential(
        nn.Sequential(
            OrderedDict(
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
                    ('linear1', nn.Linear(5, 20)),
                    ('relu', nn.ReLU()),
                    ('linear2', nn.Linear(20, 5)),
                ]
            )
        ),
    )

    head: nn.Linear = get_last_linear(model)

    head_params: set[torch.Tensor] = set(head.parameters())
    backbone_params: list[torch.Tensor] = [p for p in model.parameters() if p not in head_params]

    head_params_before_training = deepcopy(head_params)
    backbone_params_before_trainig = deepcopy(backbone_params)

    optimizer = torch.optim.AdamW(
        [
            {
                'params': backbone_params,
                'lr': 0.0,
            },
            {
                'params': list(head_params),
                'lr': 1.0,
            },
        ],
    )

    criterion: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

    for i in range(100):
        batch: torch.Tensor = torch.ones((32, 10))
        targets: torch.Tensor = torch.ones((32,)).type(torch.LongTensor)
        predictions: torch.Tensor = model(batch)

        optimizer.zero_grad()

        loss: torch.Tensor = criterion(predictions, targets)
        loss.backward()

        optimizer.step()

    head = get_last_linear(model)

    head_params_after_trainig: set[torch.Tensor] = set(head.parameters())
    backbone_params_after_trainig: list[torch.Tensor] = [p for p in model.parameters() if p not in head_params_after_trainig]

    for i, j in zip(backbone_params_before_trainig, backbone_params_after_trainig):
        assert torch.allclose(i, j)

    for i, j in zip(head_params_before_training, head_params_after_trainig):
        assert not torch.allclose(i, j)
