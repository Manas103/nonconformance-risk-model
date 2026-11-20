"""Trains the LightGBM nonconformance risk model, the logistic-regression
and random-inspection baselines, calibrates the LightGBM model with
isotonic regression, and measures the cold-start generalization gap.
Every metric is computed once per run and returned in one dict; nothing
here is iterated against the test set to chase a target.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

from .features import cold_start_test_mask, time_ordered_split

FEATURE_EXCLUDE = {"Id", "station_path", "Response"}


def feature_columns(df):
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


def best_threshold_mcc(y_true, y_prob):
    """The classification threshold that maximizes MCC on the given
    (validation) set; MCC needs a hard decision, and 0.5 is meaningless
    at a 0.5%-ish positive rate."""
    best_t, best_mcc = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        mcc = matthews_corrcoef(y_true, (y_prob >= t).astype(int))
        if mcc > best_mcc:
            best_t, best_mcc = t, mcc
    return best_t, best_mcc


def inspection_budget_recall(y_true, y_prob, budget_frac=0.05):
    n = len(y_true)
    n_flagged = max(1, int(n * budget_frac))
    order = np.argsort(-y_prob)
    flagged = order[:n_flagged]
    n_failures = y_true.sum()
    if n_failures == 0:
        return float("nan")
    return float(y_true.iloc[flagged].sum() / n_failures) if hasattr(y_true, "iloc") else \
        float(y_true[flagged].sum() / n_failures)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    n = len(y_true)
    y_true_arr = np.asarray(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true_arr[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def run(df, max_attempts_note=""):
    train_df, val_df, test_df = time_ordered_split(df)
    cols = feature_columns(df)
    report = {"n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df)}

    t0 = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        n_jobs=8,
        verbosity=-1,
    )
    model.fit(train_df[cols], train_df["Response"])
    report["lgbm_train_time_s"] = time.time() - t0

    val_prob = model.predict_proba(val_df[cols])[:, 1]
    test_prob = model.predict_proba(test_df[cols])[:, 1]
    threshold, _ = best_threshold_mcc(val_df["Response"].values, val_prob)
    test_mcc = matthews_corrcoef(test_df["Response"], (test_prob >= threshold).astype(int))
    report["lgbm"] = {
        "test_mcc": test_mcc,
        "threshold": float(threshold),
        "inspection_budget_recall_5pct": inspection_budget_recall(test_df["Response"], test_prob),
    }

    # logistic regression baseline: needs imputation, LightGBM does not
    lr_train = train_df[cols].fillna(0.0)
    lr_val = val_df[cols].fillna(0.0)
    lr_test = test_df[cols].fillna(0.0)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(lr_train, train_df["Response"])
    lr_val_prob = lr.predict_proba(lr_val)[:, 1]
    lr_test_prob = lr.predict_proba(lr_test)[:, 1]
    lr_threshold, _ = best_threshold_mcc(val_df["Response"].values, lr_val_prob)
    report["logistic_regression_baseline"] = {
        "test_mcc": matthews_corrcoef(test_df["Response"], (lr_test_prob >= lr_threshold).astype(int)),
        "threshold": float(lr_threshold),
    }

    # random inspection baseline: a uniform random score per part
    rng = np.random.default_rng(0)
    random_prob = rng.random(len(test_df))
    random_threshold, _ = best_threshold_mcc(val_df["Response"].values, rng.random(len(val_df)))
    report["random_inspection_baseline"] = {
        "test_mcc": matthews_corrcoef(test_df["Response"], (random_prob >= random_threshold).astype(int)),
        "inspection_budget_recall_5pct": inspection_budget_recall(test_df["Response"], random_prob),
    }

    # isotonic calibration, fit on val, applied to test
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_prob, val_df["Response"])
    calibrated_test_prob = iso.predict(test_prob)
    report["calibration"] = {
        "ece_uncalibrated": expected_calibration_error(test_df["Response"], test_prob),
        "ece_isotonic_calibrated": expected_calibration_error(test_df["Response"], calibrated_test_prob),
    }

    # cold-start: test rows whose station_path never appears in train
    cold_mask = cold_start_test_mask(train_df, test_df)
    report["cold_start"] = {
        "n_cold_start_rows": int(cold_mask.sum()),
        "fraction_of_test": float(cold_mask.mean()),
    }
    if cold_mask.sum() > 10:
        cold_prob = test_prob[cold_mask.values]
        cold_true = test_df.loc[cold_mask, "Response"]
        if cold_true.nunique() > 1:
            report["cold_start"]["test_mcc"] = matthews_corrcoef(cold_true, (cold_prob >= threshold).astype(int))
        else:
            report["cold_start"]["test_mcc"] = None
            report["cold_start"]["note"] = "cold-start slice has only one class present; MCC undefined"

    report["failure_rate"] = {
        "train": float(train_df["Response"].mean()),
        "val": float(val_df["Response"].mean()),
        "test": float(test_df["Response"].mean()),
    }
    report["n_features"] = len(cols)
    return report, model
