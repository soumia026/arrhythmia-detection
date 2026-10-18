"""Behavioral checks for leakage boundaries, validation and artifact inference."""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold

from arrhythmia.cli import predict
from arrhythmia.data import FEATURES, LABELS, read_csv, subject_groups
from arrhythmia.experiment import candidates, pipeline
from arrhythmia.models import FeatureCleaner, TwoStageClassifier


def fixture():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, len(FEATURES))), columns=FEATURES)
    frame["record"] = np.repeat([101, 105, 201, 202, 203, 208], 20)
    frame["type"] = np.tile(["N", "Q", "VEB"], 40)
    return frame


class ProjectTests(unittest.TestCase):
    def test_schema_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            df = fixture()
            df.to_csv(path, index=False)
            self.assertEqual(read_csv(path).shape, (120, 34))
            df.loc[0, FEATURES[0]] = np.inf
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Infinite"):
                read_csv(path)
            df = fixture().drop(columns=[FEATURES[0]])
            df.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Schema"):
                read_csv(path)

    def test_duplicate_headers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("record,record\n101,101\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_csv(path)

    def test_preprocessing_learns_only_training_values(self):
        train = pd.DataFrame({"a": [0., 2., 4.], "duplicate": [0., 2., 4.], "b": [2., 3., 4.]})
        cleaner = FeatureCleaner(clip=True).fit(train)
        self.assertEqual(cleaner.columns_, ["a", "b"])
        validation = pd.DataFrame({"a": [np.nan, 1e6], "duplicate": [999., 999.], "b": [3., 3.]})
        transformed = cleaner.transform(validation)
        self.assertEqual(transformed[0, 0], 2.)
        self.assertLess(transformed[1, 0], 4.)
        self.assertEqual(cleaner.medians_["a"], 2.)
        with self.assertRaisesRegex(ValueError, "schema"):
            cleaner.transform(validation.drop(columns="b"))

    def test_subject_boundaries_include_related_records(self):
        df = fixture()
        groups = subject_groups(df)
        self.assertEqual(groups[df.record == 201].iloc[0], groups[df.record == 202].iloc[0])
        for train, test in StratifiedGroupKFold(3, shuffle=True, random_state=42).split(df, df.type, groups):
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))

    def test_future_features_removed_and_order_stable(self):
        x = fixture()[FEATURES]
        cleaner = FeatureCleaner(drop_future=True).fit(x)
        self.assertFalse(any(c.endswith("post-RR") for c in cleaner.columns_))
        np.testing.assert_allclose(cleaner.transform(x), cleaner.transform(x[x.columns[::-1]]))

    def test_all_candidate_models_and_rare_validation_class(self):
        from sklearn.base import clone
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import f1_score, make_scorer
        df = fixture()
        x, y = df[FEATURES], df.type
        for name, model in candidates(42, 3).items():
            fitted = clone(model).fit(x, y)
            self.assertTrue(set(fitted.predict(x[:5])).issubset(set(LABELS)), name)
        # Validation data may legitimately have no Q beats; fixed-label scoring must still work.
        train = np.arange(60)
        val = np.array([i for i in range(60, 120) if y.iloc[i] != "Q"])
        scores = cross_validate(pipeline(DummyClassifier()), x, y, cv=[(train, val)],
                                scoring=make_scorer(f1_score, labels=LABELS, average="macro",
                                                    zero_division=0, pos_label="VEB"), error_score="raise")
        self.assertTrue(np.isfinite(scores["test_score"]).all())

    def test_two_stage_requires_all_classes(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            TwoStageClassifier().fit(np.zeros((2, 2)), ["N", "VEB"])

    def test_artifact_roundtrip_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            df = fixture()
            model = pipeline(DummyClassifier(strategy="most_frequent")).fit(df[FEATURES], df.type)
            joblib.dump({"pipeline": model}, directory / "model.joblib")
            df[FEATURES].to_csv(directory / "new.csv", index=False)
            result = predict(directory / "model.joblib", directory / "new.csv", directory / "pred.csv")
            np.testing.assert_array_equal(result.predicted_type, model.predict(df[FEATURES]))
            self.assertEqual(len(pd.read_csv(directory / "pred.csv")), len(df))


if __name__ == "__main__":
    unittest.main()
