from __future__ import annotations

from copy import deepcopy

from torch import nn


def get_last_linear(
    module: nn.Module,
    return_additional_info: bool = False,
) -> nn.Linear | tuple[nn.Linear, nn.Module, str] | None:
    """
    This function returns last linear layer of a given model if it exists. Also it can return a parent module \
    and found layer's name inside it.

    Args:
        module (nn.Module): model where you want to find last linear layer
        return_additional_info (bool, optional): return also a parent module and found layer's name inside it. \
        Defaults to False.

    Returns:
        nn.Linear | tuple[nn.Linear, nn.Module, str] | None: found layer + additional info or None if it was not found
    """

    if isinstance(module, nn.Linear):
        return module

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

    _find_last_linear(module)

    if last_linear_name is not None and last_linear_parent is not None:
        if return_additional_info:
            return getattr(last_linear_parent, last_linear_name), last_linear_parent, last_linear_name
        
        return getattr(last_linear_parent, last_linear_name)

    return None


def replace_last_linear(
    module: nn.Module,
    n_classes: int,
    reuse_previous_weights: bool = False
) -> nn.Module:
    """
    This function finds last (the closest one to the model output layer) linear layer and replaces it
    with the new one that has specified number of out_features.

    Args:
        module (nn.Module): model which last linear layer you want to replace.
        n_classes (int): number of out_features new layer will have.
        reuse_previous_weights (bool, optional): some of the new layer's weight will be initialized \
            used previous layer's weight. Defaults to False.

    Raises:
        ValueError: raised if reuse_previous_weights is True, but old layer has more out_features than the new one.
    Returns:
        nn.Module: model copy with re-initialized last linear layer.
    """

    module_copy: nn.Module = deepcopy(module)

    result: nn.Linear | tuple[nn.Linear, nn.Module, str] | None = get_last_linear(
        module_copy,
        return_additional_info=True,
    )

    if result is None:
        return module_copy

    if isinstance(result, nn.Linear):
        last_linear_layer, last_linear_layer_parent, last_linear_layer_name = result, None, None
    else:
        last_linear_layer, last_linear_layer_parent, last_linear_layer_name = result

    if last_linear_layer is not None:
        in_features = last_linear_layer.in_features

        new_layer: nn.Linear = nn.Linear(
            in_features,
            n_classes,
        )

        if reuse_previous_weights:
            previous_n_classes: int = last_linear_layer.out_features

            if previous_n_classes > n_classes:
                raise ValueError(
                    "can't reuse previously trained weights when" + \
                    "new number of classes is smaller the the previous one",
                )
            new_layer.weight.data[:previous_n_classes, :] = last_linear_layer.weight.data
        
        if last_linear_layer_name is None and last_linear_layer_parent is None:
            return new_layer

        setattr(
            last_linear_layer_parent,
            last_linear_layer_name,
            new_layer,
        )

        return module_copy
