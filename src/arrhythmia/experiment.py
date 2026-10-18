"""Reproducible model comparison, held-out evaluation and research artifacts."""

import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score, make_scorer)
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .data import FEATURES, LABELS, audit, read_csv, subject_groups
from .models import FeatureCleaner, TwoStageClassifier, TabularMLP


def pipeline(model, *, scale=False, clip=False, drop_future=False):
    steps = [("features", FeatureCleaner(clip=clip, drop_future=drop_future))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("model", model)])


def candidates(seed, trees):
    def forest(weight="balanced_subsample"):
        return RandomForestClassifier(n_estimators=trees, min_samples_leaf=2,
                                      class_weight=weight, random_state=seed, n_jobs=2)
    return {
        "majority_baseline": pipeline(DummyClassifier(strategy="most_frequent")),
        "linear_svm_C0.1": pipeline(LinearSVC(C=0.1, class_weight="balanced", dual="auto",
                                             max_iter=20000, random_state=seed), scale=True),
        "linear_svm_C1": pipeline(LinearSVC(C=1, class_weight="balanced", dual="auto",
                                           max_iter=20000, random_state=seed), scale=True),
        "random_forest_unweighted": pipeline(forest(None)),
        "random_forest_balanced": pipeline(forest()),
        "random_forest_clipped": pipeline(forest(), clip=True),
        "random_forest_without_post_RR": pipeline(forest(), drop_future=True),
        "tabular_mlp": pipeline(TabularMLP(random_state=seed), scale=True),
        "two_stage_forest": pipeline(TwoStageClassifier(trees, seed)),
    }


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(model, X, y):
    prediction = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0)),
        "balanced_accuracy_fixed_labels": float(recall_score(y, prediction, labels=LABELS,
                                                             average="macro", zero_division=0)),
        "classification_report": classification_report(y, prediction, labels=LABELS,
                                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, prediction, labels=LABELS).tolist(),
        "label_order": LABELS,
    }


def run(data, output, *, seed=42, quick=False):
    """Use CV on development subjects to select once, then evaluate untouched test subjects."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = read_csv(data)
    X, y, groups = df[FEATURES], df.type, subject_groups(df)
    train, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                       .split(X, y, groups))
    if set(y.iloc[train]) != set(LABELS):
        raise ValueError("Development split lacks a class; cannot train all models.")
    inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                 .split(X.iloc[train], y.iloc[train], groups.iloc[train]))
    for fit, val in inner:
        if set(y.iloc[train].iloc[fit]) != set(LABELS):
            raise ValueError("A CV training fold lacks a class; use a different documented split.")
        assert not set(groups.iloc[train].iloc[fit]) & set(groups.iloc[train].iloc[val])
    assert not set(groups.iloc[train]) & set(groups.iloc[test])
    summary = audit(df)
    summary["csv_sha256"] = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    save_json(output / "data_audit.json", summary)
    split = pd.DataFrame({"row_index": df.index, "record": df.record, "subject_group": groups,
                          "partition": "development"})
    split.loc[test, "partition"] = "test"
    split["cv_fold"] = -1
    for fold, (_, val) in enumerate(inner):
        split.loc[train[val], "cv_fold"] = fold
    split.to_csv(output / "split_manifest.csv", index=False)
    scoring = {"macro_f1": make_scorer(f1_score, labels=LABELS, average="macro", zero_division=0, pos_label="VEB"),
               "accuracy": "accuracy"}
    models = candidates(seed, 30 if quick else 120)
    records, fold_records, convergence = [], [], []
    start = time.perf_counter()
    for name, model in models.items():
        print(f"Cross-validating {name}...", flush=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            scores = cross_validate(model, X.iloc[train], y.iloc[train], cv=inner,
                                    scoring=scoring, n_jobs=1, error_score="raise")
        convergence.extend({"model": name, "message": str(w.message)} for w in caught
                           if issubclass(w.category, ConvergenceWarning))
        records.append({"model": name, "cv_macro_f1": float(scores["test_macro_f1"].mean()),
                        "cv_macro_f1_std": float(scores["test_macro_f1"].std()),
                        "cv_accuracy": float(scores["test_accuracy"].mean()),
                        "fit_seconds": float(scores["fit_time"].sum())})
        for fold in range(len(inner)):
            fold_records.append({"model": name, "fold": fold,
                                 "macro_f1": float(scores["test_macro_f1"][fold]),
                                 "accuracy": float(scores["test_accuracy"][fold])})
        pd.DataFrame(records).to_csv(output / "model_comparison.csv", index=False)
    comparison = pd.DataFrame(records).sort_values("cv_macro_f1", ascending=False, kind="stable")
    winner_name = comparison.iloc[0].model
    winner = clone(models[winner_name]).fit(X.iloc[train], y.iloc[train])
    baseline = clone(models["majority_baseline"]).fit(X.iloc[train], y.iloc[train])
    result = evaluate(winner, X.iloc[test], y.iloc[test])
    result.update({"selected_model": winner_name, "baseline": evaluate(baseline, X.iloc[test], y.iloc[test]),
                   "seed": seed, "quick": quick, "development_rows": len(train), "test_rows": len(test),
                   "development_class_counts": y.iloc[train].value_counts().to_dict(),
                   "test_class_counts": y.iloc[test].value_counts().to_dict(),
                   "test_records": sorted(int(x) for x in df.record.iloc[test].unique()),
                   "selection": "Highest mean fixed-label macro-F1 on three subject-disjoint development folds",
                   "convergence_warnings": convergence,
                   "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                                "numpy": np.__version__, "pandas": pd.__version__}})
    # A deliberately optimistic comparator, independent of selection. This does not estimate unseen-subject performance.
    print("Running beat-random diagnostic...", flush=True)
    row_train, row_test = train_test_split(np.arange(len(df)), stratify=y, test_size=0.2, random_state=seed)
    row_model = clone(models["random_forest_balanced"]).fit(X.iloc[row_train], y.iloc[row_train])
    result["beat_random_diagnostic"] = evaluate(row_model, X.iloc[row_test], y.iloc[row_test])
    result["beat_random_diagnostic"]["shared_subject_groups"] = len(set(groups.iloc[row_train]) & set(groups.iloc[row_test]))
    # Feature importance is measured on development CV fold 0 only, leaving test interpretation descriptive.
    print("Computing development-fold permutation importance...", flush=True)
    fit, val = inner[0]
    importance_model = clone(models[winner_name]).fit(X.iloc[train].iloc[fit], y.iloc[train].iloc[fit])
    importance = permutation_importance(importance_model, X.iloc[train].iloc[val], y.iloc[train].iloc[val],
                                       scoring=scoring["macro_f1"], n_repeats=3,
                                       max_samples=min(3000, len(val)), random_state=seed, n_jobs=1)
    pd.DataFrame({"feature": FEATURES, "mean_macro_f1_decrease": importance.importances_mean,
                  "std": importance.importances_std}).sort_values("mean_macro_f1_decrease", ascending=False).to_csv(
                      output / "feature_importance.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(fold_records).to_csv(output / "cv_folds.csv", index=False)
    result["elapsed_seconds"] = time.perf_counter() - start
    save_json(output / "metrics.json", result)
    joblib.dump({"pipeline": winner, "features": FEATURES, "labels": LABELS,
                 "selected_model": winner_name, "versions": result["versions"],
                 "trained_on": "Development subjects only; test subjects remain excluded"}, output / "model.joblib")
    figures(df, comparison, result, output)
    print(f"Selected {winner_name}: held-out macro-F1={result['macro_f1']:.4f}", flush=True)
    return result


def figures(df, comparison, result, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    df.type.value_counts().reindex(LABELS).plot.bar(ax=axes[0], color=["#356b8c", "#d18a37", "#b74f54"])
    axes[0].set(title="Class imbalance (log scale)", ylabel="Beats", yscale="log", xlabel="Class")
    pd.crosstab(df.record, df.type).reindex(columns=LABELS).plot.bar(stacked=True, ax=axes[1], width=0.85)
    axes[1].set(title="Beats by record", ylabel="Beats", xlabel="Record")
    axes[1].tick_params(axis="x", labelsize=6)
    fig.savefig(output / "data_overview.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    table = comparison.sort_values("cv_macro_f1")
    ax.barh(table.model, table.cv_macro_f1, xerr=table.cv_macro_f1_std, color="#356b8c")
    ax.set(xlabel="Development macro-F1 (mean ± fold SD)", xlim=(0, 1), title="Subject-disjoint model comparison")
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    matrix = np.asarray(result["confusion_matrix"])
    ConfusionMatrixDisplay(matrix, display_labels=LABELS).plot(ax=axes[0], colorbar=False, values_format="d")
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    ConfusionMatrixDisplay(normalized, display_labels=LABELS).plot(ax=axes[1], colorbar=False, values_format=".2f")
    axes[0].set_title("Held-out subjects: counts")
    axes[1].set_title("Held-out subjects: row fractions")
    fig.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(fig)
