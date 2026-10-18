"""Fold-local feature processing and a two-stage classifier."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


class FeatureCleaner(TransformerMixin, BaseEstimator):
    """Learn duplicate columns, optional quantile clipping and medians on training data only."""

    def __init__(self, clip=False, drop_future=False):
        self.clip = clip
        self.drop_future = drop_future

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        candidates = [c for c in X if not (self.drop_future and c.endswith("post-RR"))]
        kept = []
        for col in candidates:
            if not any(X[col].equals(X[other]) for other in kept):
                kept.append(col)
        self.columns_ = kept
        values = X[kept]
        self.medians_ = values.median().fillna(0)
        self.lower_ = values.quantile(0.005).fillna(0)
        self.upper_ = values.quantile(0.995).fillna(0)
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        if set(X.columns) != set(self.feature_names_in_):
            raise ValueError("Feature schema differs from the fitted schema.")
        values = X.loc[:, self.columns_].fillna(self.medians_)
        if self.clip:
            values = values.clip(self.lower_, self.upper_, axis=1)
        return values.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")
        return np.asarray(self.columns_, dtype=object)


class TwoStageClassifier(ClassifierMixin, BaseEstimator):
    """Supervised normal/other gate followed by Q/VEB classification; not anomaly detection."""

    def __init__(self, n_estimators=120, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if set(self.classes_) != {"N", "Q", "VEB"}:
            raise ValueError("Two-stage training requires N, Q and VEB.")
        base = RandomForestClassifier(n_estimators=self.n_estimators,
                                      class_weight="balanced_subsample",
                                      min_samples_leaf=2, n_jobs=2,
                                      random_state=self.random_state)
        other = y != "N"
        self.gate_ = clone(base).fit(X, other)
        self.subtype_ = clone(base).fit(X[other], y[other])
        return self

    def predict(self, X):
        check_is_fitted(self, "gate_")
        other = self.gate_.predict(X)
        result = np.full(len(X), "N", dtype="<U3")
        if other.any():
            result[other] = self.subtype_.predict(X[other])
        return result


class TabularMLP(ClassifierMixin, BaseEstimator):
    """Encode string targets to avoid sklearn 1.7's early-stopping string-label error."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.model_ = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=100,
                                    early_stopping=True, random_state=self.random_state,
                                    batch_size=min(512, max(1, int(len(X) * 0.9))))
        self.model_.fit(X, self.encoder_.transform(y))
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.encoder_.inverse_transform(self.model_.predict(X))
