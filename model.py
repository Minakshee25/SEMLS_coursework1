#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------
# F3 score function
# ---------------------------
def f3_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    if precision == 0 and recall == 0:
        return 0.0
    return 10 * (precision * recall) / (9 * precision + recall)

# ---------------------------
# Feature extraction
# ---------------------------
def extract_features(df):
    feature_rows = []
    for _, row in df.iterrows():
        # Creatinine results and dates
        creatinine_cols = [c for c in df.columns if 'creatinine_result' in c]
        date_cols = [c for c in df.columns if 'creatinine_date' in c]
        creatinine_vals = row[creatinine_cols].dropna().values
        dates = row[date_cols].dropna().values

        if len(creatinine_vals) == 0:
            # No data, fill zeros
            baseline = last = delta = mean = std = slope = 0.0
        else:
            baseline = creatinine_vals[0]
            last = creatinine_vals[-1]
            delta = last - baseline
            mean = np.mean(creatinine_vals)
            std = np.std(creatinine_vals)
            # Trend / slope
            if len(dates) >= 2:
                try:
                    dates_ordinal = pd.to_datetime(dates).map(pd.Timestamp.toordinal).values.reshape(-1,1)
                    slope = LinearRegression().fit(dates_ordinal, creatinine_vals).coef_[0]
                except:
                    slope = 0.0
            else:
                slope = 0.0

        # Encode sex
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

# ---------------------------
# Main function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()

    # Load datasets
    train = pd.read_csv("/data/training.csv")  # always train from training.csv
    test = pd.read_csv(flags.input)

    # Extract features
    X = extract_features(train)
    y = train['aki'].map({'y':1, 'n':0})
    X_test = extract_features(test)

    # Fill missing values
    X.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Train / validation split for F3 threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Threshold tuning on validation
    probs_val = model.predict_proba(X_val)[:,1]
    best_f3 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds_thresh = (probs_val >= thresh).astype(int)
        score = f3_score(y_val, preds_thresh)
        if score > best_f3:
            best_f3 = score
            best_thresh = thresh

    # Predict on test set
    probs_test = model.predict_proba(X_test)[:,1]
    preds_test = (probs_test >= best_thresh).astype(int)
    preds_yn = ['y' if p==1 else 'n' for p in preds_test]

    # Write output
    pd.DataFrame({'aki': preds_yn}).to_csv(flags.output, index=False)
    print(f"Predictions written to {flags.output} | Threshold={best_thresh:.2f} | Validation F3={best_f3:.3f}")

if __name__ == "__main__":
    main()
