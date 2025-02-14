from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupLR(_LRScheduler):
    """
    Linear learning rate warmup scheduler.

    Increases the learning rate linearly from a low initial value (1e-8 by default)
    to the chosen target learning rate during the warmup period.
    After the warmup period, the learning rate remains constant at the target value.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): Number of epochs for linear warmup.
        target_lr (float): Target learning rate to reach after warmup.
        initial_lr (float, optional): Initial learning rate at the start of warmup. Defaults to 1e-8.
        last_epoch (int, optional): The index of the last epoch when resuming training. Defaults to -1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Example:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        warmup_scheduler = LinearWarmupLR(optimizer, warmup_epochs=5, target_lr=1e-3)
        for epoch in range(10):
            train(...)
            warmup_scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        target_lr: float,
        initial_lr: float = 1e-8,
        last_epoch: int = -1,
    ):  
        self.warmup_epochs: int = warmup_epochs
        self.target_lr: float = target_lr
        self.initial_lr: float = initial_lr

        super().__init__(
            optimizer,
            last_epoch,
        )

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            warmup_lr = self.initial_lr + (self.target_lr - self.initial_lr) * progress

            return [warmup_lr for _ in self.optimizer.param_groups]

        return [self.target_lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            warmup_lr = self.initial_lr + (self.target_lr - self.initial_lr) * progress

            return [warmup_lr for _ in self.optimizer.param_groups]

        return [self.target_lr for _ in self.optimizer.param_groups]
