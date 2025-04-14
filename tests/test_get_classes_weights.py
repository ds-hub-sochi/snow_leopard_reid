from collections import OrderedDict

import numpy as np
import pytest
import torch

from irbis_classifier.src.training.weights import get_classes_weights


testdata = (
    2, 
    3,
    4,
    5,
)

@pytest.mark.parametrize("n_classes", testdata)
def test_all_classes_equal_count(
    n_classes: int,
):
    counts: OrderedDict[int, int] = {}
    for i in range(n_classes):
        counts[i] = n_classes

    weights: torch.Tensor = get_classes_weights(counts)

    for w in weights.tolist():
        assert np.allclose(w, 1/n_classes, atol=1e-5)


testdata = (
    2,
    3,
    4,
    5,
)

@pytest.mark.parametrize("n_times_bigger", testdata)
def test_one_classe_is_bigger(
    n_times_bigger: int,
):
    counts: OrderedDict[int, int] = {}
    for i in range(5):
        counts[i] = 10

    counts[0] *= n_times_bigger

    weights: torch.Tensor = get_classes_weights(counts)
    weights_lst: list[float] = weights.tolist()

    assert np.allclose(weights[0] * n_times_bigger, weights[1], atol=1e-5), "more counts -> smaller weight"

    for i in range(2, len(weights_lst)):
        assert np.allclose(weights_lst[i], weights_lst[i-1], atol=1e-5), "equal counts -> equal weights"

    assert np.allclose(sum(weights), 1.0, atol=1e-5), "weights sum must be equal to 1"


testdata = (
    3,
    4,
    5,
    6,
)

@pytest.mark.parametrize("n_classes", testdata)
def test_all_classes_different_count(
    n_classes: int,
):
    counts: OrderedDict[int, int] = {}
    for i in range(n_classes):
        counts[i] = i + 1

    weights: torch.Tensor = get_classes_weights(counts)
    weights_lst: list[float] = weights.tolist()

    for i in range(1, len(weights_lst)):
        assert np.allclose(weights_lst[i] * (i + 1), weights_lst[0], atol=1e-5), "weights must be inverse proportioned"

    assert np.allclose(sum(weights), 1.0, atol=1e-5), "weights sum must be equal to 1"
