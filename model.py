#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime, timedelta
import statistics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test.csv")
    parser.add_argument("--output", default="aki.csv")
    flags = parser.parse_args()
    with open(flags.input, newline="") as fh_in, open(flags.output, "w", newline="") as fh_out:
        reader = csv.DictReader(fh_in)
        writer = csv.writer(fh_out)
        writer.writerow(("aki",))

        # determine creatinine columns from header
        date_cols = [c for c in reader.fieldnames if c.startswith("creatinine_date_")]
        result_cols = [c for c in reader.fieldnames if c.startswith("creatinine_result_")]

        for row in reader:
            # collect (datetime, value) pairs
            measures = []
            for dcol, rcol in zip(date_cols, result_cols):
                d = row.get(dcol, "")
                v = row.get(rcol, "")
                if d and v:
                    try:
                        dt = datetime.fromisoformat(d)
                        val = float(v)
                        measures.append((dt, val))
                    except Exception:
                        continue

            if not measures:
                writer.writerow(("n",))
                continue

            measures.sort(key=lambda x: x[0])
            idx_dt, idx_val = measures[-1]

            # 48-hour rule: increase >= 26.5 umol/L within 48 hours
            two_days_ago = idx_dt - timedelta(days=2)
            aki = False
            for dt, val in measures:
                if dt < idx_dt and dt >= two_days_ago:
                    if idx_val - val >= 26.5:
                        aki = True
                        break

            if aki:
                writer.writerow(("y",))
                continue

            # 7-day rule: 1.5x baseline within 7 days
            seven_days_ago = idx_dt - timedelta(days=7)
            prev_7d = [v for (dt, v) in measures if dt < idx_dt and dt >= seven_days_ago]
            if prev_7d:
                baseline7 = min(prev_7d)
                if idx_val >= 1.5 * baseline7:
                    writer.writerow(("y",))
                    continue

            # 8-365 day historical baseline: use median of 8-365 day window if available
            hist_start = idx_dt - timedelta(days=365)
            hist_end = idx_dt - timedelta(days=8)
            hist_vals = [v for (dt, v) in measures if dt >= hist_start and dt <= hist_end]
            if hist_vals:
                try:
                    baseline_hist = statistics.median(hist_vals)
                except Exception:
                    baseline_hist = min(hist_vals)
                if idx_val >= 1.5 * baseline_hist:
                    writer.writerow(("y",))
                    continue

            writer.writerow(("n",))

if __name__ == "__main__":
    main()