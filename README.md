# SWEMLS Coursework 1: Acute Kidney Injury Prediction

## Overview

This project implements a model to predict **Acute Kidney Injury (AKI)** from patient blood test data. The system was developed as part of the SWEMLS coursework for South Riverside Hospital, which aims to alert clinical teams when a patient's condition deteriorates. For this coursework, we focus specifically on AKI as a proxy for general deterioration.

The model is trained on patient demographic information (age, sex) and historical creatinine blood test results. Predictions are evaluated using the **F3 score**, prioritizing the detection of false negatives (patients with AKI who might otherwise be overlooked).

---

## Dataset

* **training.csv**: Contains patient age, sex, historical creatinine results, and AKI diagnosis (`y` or `n`).
* **test.csv**: Contains patient data without AKI labels. The model predicts AKI for these patients.

> The training data is provided in `/data/training.csv` during evaluation.
> The `aki.csv` output must correspond row-for-row with `test.csv`.

---

## Features

The following patient-level features are extracted:

| Feature               | Description                                  |
| --------------------- | -------------------------------------------- |
| `age`                 | Patient age                                  |
| `sex`                 | Binary encoding (1 = male, 0 = female)       |
| `creatinine_baseline` | First recorded creatinine value              |
| `creatinine_last`     | Most recent creatinine value                 |
| `creatinine_delta`    | Change from baseline to last measurement     |
| `creatinine_mean`     | Mean of historical creatinine values         |
| `creatinine_std`      | Standard deviation of creatinine values      |
| `creatinine_trend`    | Linear trend (slope) of creatinine over time |

The trend is computed via **linear regression** on date-encoded creatinine measurements.

---

## Model

* **Algorithm**: LightGBM (`LGBMClassifier`)
* **Hyperparameters**:

  * `n_estimators=300`
  * `learning_rate=0.05`
  * `num_leaves=31`
  * `max_depth=-1`
  * `random_state=42`
* Trained to classify patients as AKI-positive or AKI-negative based on extracted features.
* Threshold for prediction: **0.5** (can be adjusted).

---

## Usage

Clone the repository, build the Docker image, and run the model using Docker commands.

### Build Docker Image

```bash
docker build -t coursework1 .
```

### Run Model Inference

```bash
docker run -v ${PWD}:/data coursework1
```

* Input and output paths inside the container are `/data/test.csv` and `/data/aki.csv` respectively.
* Predictions are written as `y` for AKI-positive and `n` for AKI-negative.

### Run Model Validation

To validate the model using the training data and compute the F3 score:

```bash
docker run -v ${PWD}:/data coursework1 python model.py --validate --train /data/training.csv
```

* Pass/fail status is reported based on the NHS baseline F3 (~0.73).

---

## Dependencies

* Python >= 3.12
* `pandas`
* `numpy`
* `scikit-learn`
* `lightgbm`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Evaluation

* **Metric**: F3 score (prioritizes false negatives).
* **Target**: F3 >= 0.73 (NHS baseline).
* Engineering quality is also assessed (clean code, error handling, reproducibility).

---

## Notes

* The hospital dataset is simulated for coursework purposes.
* False positives are less critical than false negatives, but still relevant.
* Any third-party libraries used are justified as safe for clinical-style deployment.

---

## Repository Structure

```
.
├── model.py          # Main model implementation
├── training.csv      # Training data (provided during evaluation)
├── test.csv          # Test data
├── aki.csv           # Output predictions
├── requirements.txt  # Python dependencies
└── Dockerfile        # Docker configuration
```

---

## Author

* Minakshee Narayankar
* SWEMLS MSc Coursework 1
