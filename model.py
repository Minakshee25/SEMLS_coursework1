#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
# from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

def parse_dates(dates):
    """Convert dates to ordinal for regression. Returns None if parsing fails."""
    try:
        return pd.to_datetime(dates, format="%Y-%m-%d").map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    except Exception:
        return None

def compute_slope(dates, values):
    """Compute slope of values over dates using linear regression."""
    if len(dates) >= 2:
        dates_ordinal = parse_dates(dates)
        if dates_ordinal is not None:
            return LinearRegression().fit(dates_ordinal, values).coef_[0]
    return 0.0

def extract_features(df):
    """
    Extract features from creatinine measurements and demographic data.
    Returns a DataFrame where each row corresponds to the features of one patient.
    """
    feature_rows = []
    creatinine_cols = [c for c in df.columns if 'creatinine_result' in c]
    date_cols = [c for c in df.columns if 'creatinine_date' in c]
    
    for _, row in df.iterrows():
        creatinine_vals = row[creatinine_cols].dropna().values
        dates = row[date_cols].dropna().values

        if len(creatinine_vals) == 0:
            baseline = last = delta = mean = std = slope = 0.0
        else:
            baseline = creatinine_vals[0]
            last = creatinine_vals[-1]
            delta = last - baseline
            mean = np.mean(creatinine_vals)
            std = np.std(creatinine_vals)
            slope = compute_slope(dates, creatinine_vals)

        sex = 1 if str(row['sex']).lower() in ['m', 'male'] else 0
        age = row.get('age', 0)

        features = {
            'age': row['age'],
            'sex': sex,
            'creatinine_baseline': baseline,
            'creatinine_last': last,
            'creatinine_delta': delta,
            'creatinine_mean': mean,
            'creatinine_std': std,
            'creatinine_slope': slope
        }
        feature_rows.append(features)

    return pd.DataFrame(feature_rows)

def train_model(X_train, y_train):
    """Train LightGBM classifier and return trained model."""
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def main():
    parser = argparse.ArgumentParser(description="AKI prediction pipeline")
    parser.add_argument("--input", default="test.csv", help="Path to input CSV")
    parser.add_argument("--output", default="aki.csv", help="Path to output CSV")
    parser.add_argument("--train", default="/data/training.csv", help="Path to training CSV")
    args = parser.parse_args()

    # Load datasets
    try:
        train = pd.read_csv(args.train)
        test = pd.read_csv(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Feature extraction
    X_train = extract_features(train)
    y_train = train['aki'].map({'y':1, 'n':0})
    X_test = extract_features(test)

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Train model
    model = train_model(X_train, y_train)

    # Predict on test set
    probs_test = model.predict_proba(X_test)[:,1]
    # Use default threshold 0.5
    preds_test = (probs_test >= 0.5).astype(int)
    preds_yn = ['y' if p==1 else 'n' for p in preds_test]

    # Save output
    pd.DataFrame({'aki': preds_yn}).to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")

if __name__ == "__main__":
    main()
