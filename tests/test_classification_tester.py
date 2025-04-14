import numpy as np
import pytest
from sklearn.metrics import f1_score

from irbis_classifier.src.testing.testers import ClassificationTester, MetricsEstimations


def test_binary_simple():
    tester: ClassificationTester = ClassificationTester()

    y_true: list[int] = [1, 1, 1, 0, 0]
    y_predicted: list[int] = [1, 1, 0, 0, 1]

    answer: dict[int, MetricsEstimations] = {
        0: MetricsEstimations(1/2, 1/2, 1/2),
        1: MetricsEstimations(2/3, 2/3, 2/3),
    }

    results: dict[str, MetricsEstimations] = tester.get_estimation_over_class(
        f1_score,
        y_true,
        y_predicted,
    )

    assert answer.keys() == results.keys()

    for label, result in results.items():
        assert np.allclose(
            result.point,
            answer[label].point,
            rtol=1e-5,
            atol=1e-5,
        )


def test_multyclass_simple():
    tester: ClassificationTester = ClassificationTester()

    y_true: list[int] = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    y_predicted: list[int] = [1, 2, 3, 1, 2, 3, 1, 2, 3]

    answer: dict[int, MetricsEstimations] = {
        1: MetricsEstimations(1/3, 1/3, 1/3),
        2: MetricsEstimations(1/3, 1/3, 1/3),
        3: MetricsEstimations(1/3, 1/3, 1/3),
    }

    results: dict[str, MetricsEstimations] = tester.get_estimation_over_class(
        f1_score,
        y_true,
        y_predicted,
    )

    assert answer.keys() == results.keys()

    for label, result in results.items():
        assert np.allclose(
            result.point,
            answer[label].point,
            rtol=1e-5,
            atol=1e-5,
        )


test_data = (
    (
        [1, 1, 1,],
        [1, 1, 1,],
        {
            1: MetricsEstimations(1, 1, 1),
        },
    ),
    (
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        {
            0: MetricsEstimations(1, 1, 1),
            1: MetricsEstimations(1, 1, 1),
        },
    ),       
)

@pytest.mark.parametrize("y_true,y_predicted,answer", test_data)
def test_binary_ideal_classification(
    y_true: list[int],
    y_predicted: list[int],
    answer: list[int],
):
    tester: ClassificationTester = ClassificationTester()
    results: dict[str, MetricsEstimations] = tester.get_estimation_over_class(f1_score, y_true, y_predicted)

    assert answer.keys() == results.keys()

    for label, result in results.items():
        assert np.allclose(
            result.point,
            answer[label].point,
            rtol=1e-5,
            atol=1e-5,
        )
        assert np.allclose(
            result.lower,
            answer[label].lower,
            atol=3e-2,
        )
        assert np.allclose(
            result.upper,
            answer[label].upper,
            atol=3e-2,
        )
