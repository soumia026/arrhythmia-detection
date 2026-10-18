"""Command-line training and schema-checked batch prediction."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURES, read_csv
from .experiment import run


def predict(model_path, data_path, output):
    # joblib uses pickle: only load artifacts you created or trust.
    artifact = joblib.load(model_path)
    df = read_csv(data_path, training=False)
    labels = artifact["pipeline"].predict(df[FEATURES])
    result = pd.DataFrame({"row_index": df.index, "predicted_type": labels})
    if "record" in df:
        result.insert(1, "record", df.record)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Compare models and save an evaluated pipeline")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--output", default=Path("artifacts/run"), type=Path)
    train.add_argument("--seed", default=42, type=int)
    train.add_argument("--quick", action="store_true", help="30 trees rather than 120; all rows and folds retained")
    inference = sub.add_parser("predict", help="Predict from extracted features (trusted local model only)")
    inference.add_argument("--model", required=True, type=Path)
    inference.add_argument("--data", required=True, type=Path)
    inference.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        run(args.data, args.output, seed=args.seed, quick=args.quick)
    else:
        predict(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
