import numpy as np
import pytest

from src.prediction.evaluation import (
    classification_metrics,
    regression_metrics,
)


def test_classification_metrics_return_expected_values():
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]

    results = classification_metrics(y_true, y_pred)

    assert results["accuracy"] == pytest.approx(4 / 6)
    assert results["precision"] == pytest.approx(2 / 3)
    assert results["recall"] == pytest.approx(2 / 3)
    assert results["f1_score"] == pytest.approx(2 / 3)
    assert results["confusion_matrix"] == [[2, 1], [1, 2]]
    assert "classification_report" in results


def test_classification_metrics_calculate_roc_auc():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.3, 0.7, 0.9]

    results = classification_metrics(y_true, y_pred, y_prob)

    assert results["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_use_selected_probability_column():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = [
        [0.9, 0.1],
        [0.2, 0.8],
        [0.7, 0.3],
        [0.1, 0.9],
    ]

    results = classification_metrics(
        y_true,
        y_pred,
        y_prob,
        positive_class_index=1,
    )

    assert results["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_skip_roc_auc_without_probabilities():
    results = classification_metrics(
        [0, 1, 0, 1],
        [0, 1, 1, 1],
    )

    assert "roc_auc" not in results


def test_classification_metrics_skip_roc_auc_for_multiclass_data():
    results = classification_metrics(
        [0, 1, 2, 0, 1, 2],
        [0, 1, 1, 0, 2, 2],
        y_prob=[0.2, 0.8, 0.7, 0.1, 0.6, 0.9],
    )

    assert "roc_auc" not in results
    assert results["confusion_matrix"] == [
        [2, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ]


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([], []),
        ([0, 1], [0]),
    ],
)
def test_classification_metrics_reject_empty_or_unequal_inputs(
    y_true, y_pred
):
    with pytest.raises(ValueError):
        classification_metrics(y_true, y_pred)


def test_classification_metrics_reject_two_dimensional_targets():
    with pytest.raises(ValueError, match="one-dimensional"):
        classification_metrics(
            [[0], [1]],
            [[0], [1]],
        )


def test_classification_metrics_reject_invalid_probability_column():
    with pytest.raises(ValueError, match="positive class index"):
        classification_metrics(
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            y_prob=[[0.8, 0.2], [0.2, 0.8],
                    [0.7, 0.3], [0.1, 0.9]],
            positive_class_index=2,
        )


def test_regression_metrics_return_expected_values():
    y_true = [3.0, -0.5, 2.0, 7.0]
    y_pred = [2.5, 0.0, 2.0, 8.0]

    results = regression_metrics(y_true, y_pred)

    assert results["mae"] == pytest.approx(0.5)
    assert results["mse"] == pytest.approx(0.375)
    assert results["rmse"] == pytest.approx(np.sqrt(0.375))
    assert results["r2_score"] == pytest.approx(0.948608137)


def test_regression_metrics_return_none_for_single_sample_r2():
    results = regression_metrics([10.0], [8.0])

    assert results["mae"] == pytest.approx(2.0)
    assert results["mse"] == pytest.approx(4.0)
    assert results["rmse"] == pytest.approx(2.0)
    assert results["r2_score"] is None


@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([], []),
        ([1.0, 2.0], [1.0]),
    ],
)
def test_regression_metrics_reject_empty_or_unequal_inputs(
    y_true, y_pred
):
    with pytest.raises(ValueError):
        regression_metrics(y_true, y_pred)


def test_regression_metrics_reject_two_dimensional_targets():
    with pytest.raises(ValueError, match="one-dimensional"):
        regression_metrics(
            [[1.0], [2.0]],
            [[1.0], [2.0]],
        )

@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, 2.0], [1.0, np.inf]),
    ],
)
def test_regression_metrics_reject_non_finite_values(y_true, y_pred):
    with pytest.raises(ValueError):
        regression_metrics(y_true, y_pred)