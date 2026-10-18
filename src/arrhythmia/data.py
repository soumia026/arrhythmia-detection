"""Strict CSV validation and subject grouping, shared by training and inference."""

from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ["N", "Q", "VEB"]
SUFFIXES = ["pre-RR", "post-RR", "pPeak", "tPeak", "rPeak", "sPeak", "qPeak",
            "qrs_interval", "pq_interval", "qt_interval", "st_interval",
            *[f"qrs_morph{i}" for i in range(5)]]
FEATURES = [f"{lead}_{suffix}" for lead in (0, 1) for suffix in SUFFIXES]


def read_csv(path: str | Path, *, training: bool = True) -> pd.DataFrame:
    """Reject broken schemas and nonnumeric/infinite features; allow missing features for imputation."""
    import csv
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not supported.")
    frame = pd.read_csv(path)
    required = FEATURES + (["record", "type"] if training else [])
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(FEATURES + ["record", "type"]))
    if missing or extra:
        raise ValueError(f"Schema mismatch: missing={missing}, unexpected={extra}")
    if frame.empty:
        raise ValueError("The CSV contains no rows.")
    try:
        frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("All ECG features must be numeric.") from exc
    if np.isinf(frame[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values are not supported.")
    if training:
        if frame[["record", "type"]].isna().any().any():
            raise ValueError("Record IDs and labels cannot be missing.")
        record = pd.to_numeric(frame.record, errors="raise")
        if not np.isfinite(record).all() or not (record == np.floor(record)).all():
            raise ValueError("Record IDs must be finite integers.")
        frame["record"] = record.astype(int)
        if set(frame.type) != set(LABELS):
            raise ValueError(f"Training data must contain exactly these labels: {LABELS}")
    return frame


def subject_groups(frame: pd.DataFrame) -> pd.Series:
    """PhysioNet identifies records 201 and 202 as recordings from one subject."""
    return frame.record.replace({202: 201})


def audit(frame: pd.DataFrame) -> dict:
    x = frame[FEATURES]
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "feature_count": len(FEATURES), "records": int(frame.record.nunique()),
        "subject_groups": int(subject_groups(frame).nunique()),
        "class_counts": {k: int(v) for k, v in frame.type.value_counts().items()},
        "missing_features": int(x.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "identical_feature_pairs": [[a, b] for i, a in enumerate(FEATURES)
                                    for b in FEATURES[i + 1:] if x[a].equals(x[b])],
        "class_record_counts": {k: int(v) for k, v in frame.groupby("type").record.nunique().items()},
    }
