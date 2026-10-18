# Arrhythmia detection from ECG features

A reproducible machine-learning project by **KOUADRI Soumia**, using a derived MIT-BIH heartbeat feature table to classify **N, VEB and Q** labels.

The revised project compares classical models, a small tabular neural network and a two-stage classifier. It evaluates unseen subjects, reports class-specific errors, and provides a saved-pipeline prediction command. This is research software; the data does not support reliable rare-class clinical detection.

## Start here

- [Executed analysis notebook](notebooks/arrhythmia_analysis.ipynb): dataset audit, experiments, figures, results and conclusions.
- [Results report](reports/RESULTS.md): measured performance and interpretation.
- [Dataset requirements](data/README.md): exact input schema, provenance limitations and setup.
- [Model card](MODEL_CARD.md): intended use, evaluation and limitations.
- [Original notebook](archive/original_arrhythmia.ipynb): historical work, with stale outputs cleared.

## Run locally

Use Python 3.11 or 3.12. From the repository root:

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -e ".[notebook]"
arrhythmia train --data "path/to/MIT-BIH Arrhythmia Database new.csv" --output artifacts/run
arrhythmia predict --model artifacts/run/model.joblib --data "path/to/new_features.csv" --output artifacts/predictions.csv
python -m unittest discover -s tests -v
jupyter lab notebooks/arrhythmia_analysis.ipynb
```

Training runs on CPU. `--quick` uses 30 instead of 120 trees, while retaining all input rows and folds. It produces a separate experiment, so use a separate output directory. The published results use the full 120-tree configuration and seed 42. The SVM is linear to avoid the original hours-long kernel-SVC search.

The notebook displays committed results without the CSV. Set `ARRHYTHMIA_DATA` to the CSV path and `RUN_TRAINING = True` in the notebook to regenerate them. The raw feature CSV and trained joblib model stay local; neither is included in the repository. Retraining requires the matching feature table, whose extraction procedure was not supplied. Unit tests use synthetic data.

## Evaluation

The input contains **97,107 beats**, **44 records** and **43 subject groups** after merging records 201/202, which [PhysioNet identifies as one subject](https://physionet.org/content/mitdb/1.0.0/). Class counts are 90,083 N, 7,009 VEB and only 15 Q.

One preselected grouped fold is held out; three grouped folds on the remaining subjects compare nine candidates. Duplicate-column removal, imputation, optional quantile clipping and scaling are learned inside training folds. Selection uses mean macro-F1 over the fixed label set N/Q/VEB, with undefined class F1 assigned zero. Some validation folds have no Q beats, so fold variability must be interpreted cautiously.

Experiments include:

- Majority baseline, two linear-SVM settings, Random Forest and a two-hidden-layer MLP.
- Unweighted versus class-weighted forests, and training-only quantile clipping.
- Removing post-RR to examine dependence on the next heartbeat.
- A supervised normal/other gate followed by Q/VEB classification.
- A separate beat-random split diagnostic to illustrate how shared-subject evaluation differs.

The saved pipeline is fitted on development subjects only. The test set is used for final evaluation, never candidate selection. Accuracy, macro-F1, fixed-label macro recall, per-class precision/recall, confusion matrices, split manifests and development-fold permutation importance are exported. Fold standard deviations are descriptive, not confidence intervals.

## What changed from the original

- Fixed undefined results variables and replaced incomplete training cells with a reusable package and CLI.
- Replaced SMOTE before cross-validation with an explicit class-weighting comparison; preprocessing is fitted inside each fold.
- Replaced random beat splitting as the primary evaluation with subject-disjoint splitting, including the shared 201/202 subject.
- Completed model comparisons, the two-stage proposal, outlier-clipping and future-RR experiments, interpretation and conclusion.
- Added schema validation, prediction, behavioral tests and GitHub Actions checks.
- Corrected the row count, lead-versus-beat terminology, and unsupported claims about feature duplication and feature-extraction provenance.

## Scope and limitations

This project classifies **already extracted features**. It does not detect R peaks from raw ECG, denoise waveforms, infer an arrhythmia episode, or provide a streaming clinical detector. The source CSV's label mapping and feature definitions cannot be fully verified, so labels are retained as provided. N should not automatically be interpreted as clinical normality, and Q is not a well-supported diagnostic category here.

Only 15 Q beats are available, including seven in the test set. More rare-class data, an independently verified extraction pipeline, repeated or external subject-level evaluation, and demographic metadata are needed for stronger claims. Post-RR uses the next beat; other feature-extraction windows are also undocumented. Temporal sequence models and waveform CNNs remain future work requiring additional data.

Only load joblib model files you created or trust. Prediction returns class labels, not calibrated diagnostic probabilities. No open-source license is granted by this repository; dataset redistribution rights were not established. AI assistance was used in the original notebook and this revision; experiment outputs come from actual executed code.

## References

- [Moody and Mark, MIT-BIH Arrhythmia Database v1.0.0, PhysioNet](https://physionet.org/content/mitdb/1.0.0/).
- [scikit-learn: avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn: grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
- [scikit-learn: permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
