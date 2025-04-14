from __future__ import annotations

from copy import deepcopy

from torch import nn


def replace_last_linear(
    module: nn.Module,
    n_classes: int,
) -> nn.Module:
    """
    This function replaces last linear layer of given model with brand new layer with chosen output size

    Args:
        module (nn.Module): the model whose last line layer you want to replace
        n_classes (int): output size of the new linear layer

    Returns:
        nn.Module: model with replaced last linear layer
    """
    if isinstance(module, nn.Linear):
        in_features: int = module.in_features

        return nn.Linear(
            in_features,
            n_classes,
        )

    module_copy: nn.Module = deepcopy(module)

    last_linear_name: str | None = None
    last_linear_parent: nn.Module | None = None

    def _find_last_linear(
        model: nn.Module,
    ):
        nonlocal last_linear_name, last_linear_parent

        for child_name, child in model.named_children():
            if isinstance(child, nn.Linear):
                last_linear_name = child_name
                last_linear_parent = model
            else:
                _find_last_linear(child)

    _find_last_linear(module_copy)

    if last_linear_name is not None and last_linear_parent is not None:
        in_features = getattr(last_linear_parent, last_linear_name).in_features
        setattr(last_linear_parent, last_linear_name, nn.Linear(in_features, n_classes))

    return module_copy
