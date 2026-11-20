import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_model.train import best_threshold_mcc, expected_calibration_error, inspection_budget_recall


def test_best_threshold_mcc_finds_a_separable_threshold():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 900 + [1] * 100)
    y_prob = np.concatenate([rng.uniform(0, 0.4, 900), rng.uniform(0.6, 1.0, 100)])
    threshold, mcc = best_threshold_mcc(y_true, y_prob)
    assert 0.4 <= threshold <= 0.6
    assert mcc > 0.9


def test_best_threshold_mcc_is_near_zero_for_random_scores():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 1000)
    y_prob = rng.random(1000)
    _, mcc = best_threshold_mcc(y_true, y_prob)
    assert mcc < 0.3  # a random score should not find strong separation


def test_inspection_budget_recall_catches_all_top_ranked_failures():
    y_true = pd.Series([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1])
    recall = inspection_budget_recall(y_true, y_prob, budget_frac=0.2)
    assert recall == 1.0  # top 2 (20%) are exactly the 2 real failures


def test_inspection_budget_recall_misses_when_failures_rank_low():
    y_true = pd.Series([0] * 9 + [1])
    y_prob = np.array([0.9] * 9 + [0.01])
    recall = inspection_budget_recall(y_true, y_prob, budget_frac=0.2)
    assert recall == 0.0


def test_expected_calibration_error_is_zero_for_perfectly_calibrated_bins():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_prob = np.array([0.6] * 10)  # matches the true positive rate exactly
    ece = expected_calibration_error(y_true, y_prob, n_bins=1)
    assert abs(ece) < 1e-9


def test_expected_calibration_error_is_positive_when_overconfident():
    y_true = np.array([0] * 9 + [1])
    y_prob = np.array([0.95] * 10)
    ece = expected_calibration_error(y_true, y_prob, n_bins=1)
    assert ece > 0.5
