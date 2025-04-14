import warnings

import torch
from numpy import allclose

from irbis_classifier.src.training import warmup_schedulers


warnings.simplefilter(
    'ignore',
    DeprecationWarning,
)


def test_linear_warmup_simple_test():
    base_lr = 1e-4
    warmup_steps = 10
    start_lr = 1e-8

    total_steps = 50  # Total number of training steps

    model: torch.nn.Module = torch.nn.Linear(
        10,
        10,
    )
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    warmup_scheduler: warmup_schedulers.LinearWarmupLR = warmup_schedulers.LinearWarmupLR(
        optimizer,
        warmup_epochs = warmup_steps,
        target_lr = base_lr,
        initial_lr = start_lr,
    )

    learning_rates = []

    lr = optimizer.param_groups[0]['lr']
    learning_rates.append(lr)

    for step in range(total_steps):
        optimizer.step()

        if step < warmup_scheduler.warmup_epochs:
            warmup_scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        learning_rates.append(lr)

    assert learning_rates[0] == start_lr
    assert learning_rates[warmup_steps] == base_lr

    for i in range(1, warmup_steps - 1):
        assert allclose(
            learning_rates[i] - learning_rates[i - 1],
            learning_rates[i + 1] - learning_rates[i],
            atol=1e-5,   
        )

    for lr in learning_rates[warmup_steps:]:
        assert lr == base_lr
