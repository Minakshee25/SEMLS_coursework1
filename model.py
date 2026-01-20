#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

def parse_dates_to_ordinal(dates):
    """Convert date strings to ordinal numbers for regression. Returns None if parsing fails."""
    try:
        return pd.to_datetime(dates, format="%Y-%m-%d").map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    except Exception:
        return None

def compute_creatinine_trend(creatinine_dates, creatinine_history):
    """Compute linear trend (slope) of creatinine over time."""
    if len(creatinine_dates) >= 2:
        date_ordinals = parse_dates_to_ordinal(creatinine_dates)
        if date_ordinals is not None:
            return LinearRegression().fit(date_ordinals, creatinine_history).coef_[0]
    return 0.0

def compute_patient_features(df):
    """Extract patient-level features from demographic info and creatinine history."""
    patient_features = []

    creatinine_columns = [c for c in df.columns if 'creatinine_result' in c]
    date_columns = [c for c in df.columns if 'creatinine_date' in c]

    for patient in df.itertuples(index=False):
        creatinine_history = getattr(patient, 'creatinine_result_0', np.array([0.0]))
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
    """Train LightGBM model to predict AKI."""
    aki_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42
    )
    aki_model.fit(features, labels)
    return aki_model

def main():
    parser = argparse.ArgumentParser(description="Predict Acute Kidney Injury from patient data")
    parser.add_argument("--input", default="test.csv", help="Path to test CSV")
    parser.add_argument("--output", default="aki.csv", help="Path to write predictions CSV")
    parser.add_argument("--train", default="/data/training.csv", help="Path to training CSV")
    args = parser.parse_args()

    try:
        train_data = pd.read_csv(args.train)
        test_data = pd.read_csv(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Extract features
    train_features = compute_patient_features(train_data)
    train_labels = train_data['aki'].map({'y':1, 'n':0})
    test_features = compute_patient_features(test_data)

    train_features.fillna(0, inplace=True)
    test_features.fillna(0, inplace=True)

    # Train model
    aki_model = train_aki_predictor(train_features, train_labels)

    # Generate predictions
    aki_probabilities = aki_model.predict_proba(test_features)[:,1]
    predicted_labels = (aki_probabilities >= 0.5).astype(int)
    predicted_yn = ['y' if x == 1 else 'n' for x in predicted_labels]

    # Save output
    pd.DataFrame({'aki': predicted_yn}).to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")

if __name__ == "__main__":
    main()
