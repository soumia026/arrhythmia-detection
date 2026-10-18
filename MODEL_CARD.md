# Model card

## Intended use

Educational research and reproducible comparison of classifiers on a supplied MIT-BIH-derived feature table. The input is a row of 32 numeric extracted ECG features; output is one of N, Q or VEB. No raw waveform processing or clinical diagnosis is provided.

## Training and selection

See `reports/metrics.json` for the selected candidate and exact package versions. Nine predefined candidates are compared using three subject-disjoint development CV folds. Mean fixed-label macro-F1 selects one winner. The outer test fold is fixed before comparison (seed 42, first fold of five-fold StratifiedGroupKFold). Records 201 and 202 share a subject group.

Fold-local processing learns duplicate columns and feature medians; SVM/MLP pipelines scale inputs. Forests compare unweighted and class-weighted fitting, optional training-quantile clipping, removal of post-RR, and a supervised two-stage architecture. The small MLP encodes target strings internally because scikit-learn 1.7.1's early-stopping scoring expects numeric labels.

The exported model remains trained only on development subjects, matching its reported held-out evaluation. Retraining with different seeds, tree counts or package versions constitutes a new experiment.

## Evaluation evidence

`reports/RESULTS.md`, the executed notebook and `reports/metrics.json` contain observed test metrics. `reports/model_comparison.csv` and `reports/cv_folds.csv` contain development comparisons. `reports/split_manifest.csv` records every original row's assignment and subject group; `-1` marks the outer test rows. The dataset SHA-256 identifies the exact source file. The beat-random diagnostic is separate from model selection and has overlapping subjects and a different test population.

## Limitations

- 15 Q examples overall and seven held-out Q beats cannot establish dependable rare-class detection. Some validation folds have no Q support; fixed-label scoring assigns an undefined F1 zero.
- Extracted feature definitions, units, annotation mapping, upstream preprocessing and licensing are not fully documented. No complete raw-waveform-to-feature reproduction is possible from the supplied files.
- This is a small historical subject cohort; demographic subgroup performance is unknown. There is no external validation or prospective clinical evaluation.
- Input rows may depend on neighboring beats. Post-RR explicitly observes the next heartbeat. The feature-removal ablation does not establish real-time operation.
- Permutation importance is computed on one development validation fold with a limited sample. It measures predictive sensitivity and is not a causal or Q-specific explanation.
- No calibrated confidence scores, clinical decision thresholds, deployment service or monitoring system are supplied.

## Inference contract

Supply the complete feature schema in `src/arrhythmia/data.py`; `record` and `type` are optional during prediction. Columns can be reordered. Missing values use saved training medians, and unknown/missing columns, infinity and nonnumeric features fail validation. Output CSV retains row indexes and optional record IDs. Only load trusted joblib artifacts.

## Provenance

Author: KOUADRI Soumia. Original research notebook preserved under `archive/`. The original notebook disclosed AI assistance; AI assistance was also used to revise, implement and test the project. Source data stays local and is excluded from Git because distribution rights for this particular derived table were not established.
