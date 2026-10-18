# Dataset setup

Copy your feature CSV here as `mitbih_features.csv`, or pass its existing path to `arrhythmia train --data`.

The supplied file has 97,107 rows, 44 record IDs, 32 numeric ECG features, `record`, and `type` (N, Q, VEB).
Records 201 and 202 are grouped as one subject, giving 43 subject groups.
It contains 90,083 N, 7,009 VEB and 15 Q examples, with no missing values or exact duplicate rows.

This is a derived feature table, not the original waveform database. The original feature-extraction code,
label mapping, preprocessing history and distribution license were not supplied. Feature names suggest
RR intervals, wave amplitudes, segment durations and morphology coefficients for two ECG leads;
their precise definitions and units cannot be independently confirmed from the CSV.

The original MIT-BIH database is at https://physionet.org/content/mitdb/1.0.0/ . Downloading its raw
records will **not** reproduce this feature table without the missing extraction procedure.
Do not substitute an unrelated MIT-BIH CSV (such as 187-sample beat arrays): schemas differ.

The CSV and trained binary models are excluded from Git. This avoids claiming redistribution rights
for an extract whose source and license are unknown. The repository includes its SHA-256 fingerprint
and aggregate results to identify the exact evaluated input. Obtaining or supplying a compatible
feature table is required to retrain; code tests use generated fixtures and need no external data.
