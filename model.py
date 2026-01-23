#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

def parse_dates_to_ordinal(dates):
    """
    Convert date strings to ordinal numbers for regression.

    Args:
        dates: Array-like of date strings in 'YYYY-MM-DD' format.

    Returns:
        np.ndarray or None: Ordinal representation of dates shaped (-1, 1), or None if parsing fails.

    """
    try:
        return pd.to_datetime(dates, format="%Y-%m-%d").map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    except Exception:
        return None

def compute_creatinine_trend(creatinine_dates, creatinine_history):
    """
    Compute the linear trend (slope) of creatinine values over time.

    Args:
        creatinine_dates: Array-like of date strings corresponding to measurements.
        creatinine_history: Array-like of creatinine measurements.

    Returns:
        float: Slope of creatinine over time. Returns 0.0 if not enough data.
    """
    if len(creatinine_dates) >= 2:
        date_ordinals = parse_dates_to_ordinal(creatinine_dates)
        if date_ordinals is not None:
            return LinearRegression().fit(date_ordinals, creatinine_history).coef_[0]
    return 0.0

def compute_patient_features(df):
    """
    Extract patient-level features from demographics and creatinine history.

    Args:
        df: Pandas DataFrame containing patient data, including creatinine results and dates.

    Returns:
        pd.DataFrame: Patient features including age, sex, creatinine baseline, last measurement, delta, mean, std, and trend.
    """
    patient_features = []

    creatinine_columns = [c for c in df.columns if 'creatinine_result' in c]
    date_columns = [c for c in df.columns if 'creatinine_date' in c]

    for patient in df.itertuples(index=False):
        creatinine_history = np.array([getattr(patient, c) for c in creatinine_columns if pd.notnull(getattr(patient, c))])
        creatinine_dates = np.array([getattr(patient, c) for c in date_columns if pd.notnull(getattr(patient, c))])

        if len(creatinine_history) == 0:
            baseline = last_measurement = delta = mean_val = std_val = trend = 0.0
        else:
            baseline = creatinine_history[0]
            last_measurement = creatinine_history[-1]
            delta = last_measurement - baseline
            mean_val = np.mean(creatinine_history)
            std_val = np.std(creatinine_history)
            trend = compute_creatinine_trend(creatinine_dates, creatinine_history)

        sex_binary = 1 if str(getattr(patient, 'sex')).lower() in ['m', 'male'] else 0
        age_val = getattr(patient, 'age', 0)

        patient_features.append({
            'age': age_val,
            'sex': sex_binary,
            'creatinine_baseline': baseline,
            'creatinine_last': last_measurement,
            'creatinine_delta': delta,
            'creatinine_mean': mean_val,
            'creatinine_std': std_val,
            'creatinine_trend': trend
        })

    return pd.DataFrame(patient_features)

def train_aki_predictor(features, labels):
    """
    Train a LightGBM model to predict Acute Kidney Injury (AKI).

    Args:
        features: Pandas DataFrame of patient features.
        labels: Array-like of binary labels (0/1) indicating AKI occurrence.

    Returns:
        lgb.LGBMClassifier: Trained LightGBM model.
    """
    aki_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42
    )
    aki_model.fit(features, labels)
    return aki_model

def evaluate_model(features, labels, threshold=0.5):
    """
    Split data, train model, and compute F3 score on validation set.

    Args:
        features: Pandas DataFrame of patient features.
        labels: Array-like of binary labels (0/1) indicating AKI occurrence.
        threshold: Float, probability threshold for converting predicted probabilities to binary labels.

    Returns:
        float: F3 score on the validation set. Returns 0.0 if validation set has a single class.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = train_aki_predictor(X_train, y_train)
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)

    if len(np.unique(y_val)) < 2:
        print("Warning: Validation set has a single class; F3 undefined")
        return 0.0

    f3 = fbeta_score(y_val, preds, beta=3)
    return f3

def main():
    parser = argparse.ArgumentParser(description="Predict Acute Kidney Injury from patient data")
    parser.add_argument("--input", default="test.csv", help="Path to test CSV")
    parser.add_argument("--output", default="aki.csv", help="Path to write predictions CSV")
    parser.add_argument("--train", default="/data/training.csv", help="Path to training CSV")
    parser.add_argument("--validate", action="store_true", help="Run automated validation instead of inference")
    args = parser.parse_args()

    try:
        train_data = pd.read_csv(args.train)
    except FileNotFoundError as e:
        print(f"Error loading training data: {e}")
        return

    # Extract features
    train_features = compute_patient_features(train_data)
    train_features.fillna(0, inplace=True)
    train_labels = train_data['aki'].map({'y':1, 'n':0})

    # ---------------- VALIDATION MODE ----------------
    if args.validate:
        f3 = evaluate_model(train_features, train_labels)
        print(f"Validation F3 score: {f3:.3f}")

        NHS_BASELINE_F3 = 0.73
        if f3 >= NHS_BASELINE_F3:
            print("STATUS: PASS — Model meets expected quality")
            exit(0)
        else:
            print("STATUS: FAIL — Model below expected quality")
            exit(1)

    # ---------------- INFERENCE MODE ----------------
    try:
        test_data = pd.read_csv(args.input)
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return

    test_features = compute_patient_features(test_data)
    test_features.fillna(0, inplace=True)

    aki_model = train_aki_predictor(train_features, train_labels)
    aki_probs = aki_model.predict_proba(test_features)[:, 1]
    aki_preds = (aki_probs >= 0.5).astype(int)

    output = ["y" if p == 1 else "n" for p in aki_preds]
    pd.DataFrame({"aki": output}).to_csv(args.output, index=False)

    print(f"Predictions written to {args.output}")

if __name__ == "__main__":
    main()