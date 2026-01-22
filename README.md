# AKI Prediction Script

This repository contains a simple end-to-end pipeline to **predict Acute Kidney Injury (AKI)** from patient demographic data and historical creatinine measurements.

The script reads training and test CSV files, extracts meaningful patient-level features from creatinine time series, trains a LightGBM model, and outputs AKI predictions for the test set.

---

## What this script does

At a high level, the script:

1. Reads patient data from CSV files
2. Extracts features from creatinine history (baseline, trend, variability, etc.)
3. Trains a LightGBM classifier using labeled training data
4. Predicts AKI for unseen patients
5. Writes predictions (`y` / `n`) to an output CSV

The goal is to keep the pipeline lightweight, interpretable, and easy to run from the command line.

---

## Features used for prediction

For each patient, the following features are computed:

* Age
* Sex (binary encoded: male = 1, female = 0)
* Creatinine baseline (first available value)
* Last creatinine measurement
* Change in creatinine (last − baseline)
* Mean creatinine
* Standard deviation of creatinine
* Creatinine trend (slope over time using linear regression)

If creatinine history is missing or insufficient, the script safely falls back to zeros.

---

## Model details

* Algorithm: LightGBM (`LGBMClassifier`)
* Objective: Binary classification (AKI vs no AKI)
* Output: Probability-based prediction with a 0.5 threshold
* Label encoding:

  * `y` → AKI present
  * `n` → AKI not present

---

## Input data requirements

### Training CSV

The training file must contain:

* `age`
* `sex`
* `aki` (label: `y` or `n`)
* One or more creatinine result columns:

  * `creatinine_result_0`, `creatinine_result_1`, ...
* Corresponding date columns:

  * `creatinine_date_0`, `creatinine_date_1`, ...

### Test CSV

The test file should contain the same fields **except** the `aki` column.

---

## How to run

### Basic usage

```bash
python3 predict_aki.py \
  --train /data/training.csv \
  --input test.csv \
  --output aki.csv
```

### Arguments

| Argument   | Description                    | Default              |
| ---------- | ------------------------------ | -------------------- |
| `--train`  | Path to training CSV           | `/data/training.csv` |
| `--input`  | Path to test CSV               | `test.csv`           |
| `--output` | Path to output predictions CSV | `aki.csv`            |

---

## Output format

The output file is a CSV with a single column:

```csv
aki
y
n
y
n
```

Each row corresponds to a patient in the test dataset.

---

## Dependencies

Make sure the following Python packages are installed:

* pandas
* numpy
* scikit-learn
* lightgbm

You can install them with:

```bash
pip install pandas numpy scikit-learn lightgbm
```

---

## Notes & assumptions

* Creatinine dates are expected in `YYYY-MM-DD` format.
* Linear regression is used to estimate creatinine trend over time.
* Missing values are handled conservatively by filling with zeros.
* This script is intended for experimentation and prototyping, not direct clinical use.

---

## Future improvements

* Better handling of irregular time gaps in creatinine measurements
* Explicit feature normalization
* Model evaluation metrics and cross-validation
* Support for additional lab values
