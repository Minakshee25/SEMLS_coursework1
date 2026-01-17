#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

def extract_features(df):
    feature_rows = []
    for _, row in df.iterrows():
        # Creatinine columns
        creatinine_cols = [c for c in df.columns if 'creatinine_result' in c]
        date_cols = [c for c in df.columns if 'creatinine_date' in c]
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
            if len(dates) >= 2:
                try:
                    dates_ordinal = pd.to_datetime(dates).map(pd.Timestamp.toordinal).values.reshape(-1,1)
                    slope = LinearRegression().fit(dates_ordinal, creatinine_vals).coef_[0]
                except:
                    slope = 0.0
            else:
                slope = 0.0

        sex = 1 if str(row['sex']).lower() in ['m', 'male'] else 0

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()

    # Load datasets
    train = pd.read_csv("/data/training.csv")
    test = pd.read_csv(flags.input)

    # Feature extraction
    X_train = extract_features(train)
    y_train = train['aki'].map({'y':1, 'n':0})
    X_test = extract_features(test)

    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Train model on all data
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    probs_test = model.predict_proba(X_test)[:,1]
    # Use default threshold 0.5
    preds_test = (probs_test >= 0.5).astype(int)
    preds_yn = ['y' if p==1 else 'n' for p in preds_test]

    # Save output
    pd.DataFrame({'aki': preds_yn}).to_csv(flags.output, index=False)
    print(f"Predictions written to {flags.output}")

if __name__ == "__main__":
    main()
