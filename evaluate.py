"""
evaluate.py
-----------
Measures how well Subscription Guardian works against the KNOWN answer key
(data/ground_truth.csv). This is the number you show the judges.

Two things are measured:
  1. Subscription detection  -> did we find the recurring charges and reject
                                the one-off spends?  (precision / recall)
  2. Hidden detection        -> of the truly hidden subscriptions, how many
                                did we flag?

Run:
    python evaluate.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import recurrence  # noqa: E402
import classifier  # noqa: E402
import risk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main():
    txns = pd.read_csv(os.path.join(HERE, "data", "transactions.csv"))
    gt = pd.read_csv(os.path.join(HERE, "data", "ground_truth.csv"))

    scored = risk.score(classifier.classify(recurrence.find_recurring(txns)))
    detected_subs = set(scored["merchant"])
    detected_hidden = set(scored[scored["is_hidden"]]["merchant"])

    true_subs = set(gt[gt["is_subscription"]]["merchant"])
    all_merchants = set(gt["merchant"])
    true_hidden = set(gt[gt["is_hidden"]]["merchant"])

    # --- Subscription detection ---
    tp = len(detected_subs & true_subs)
    fp = len(detected_subs - true_subs)
    fn = len(true_subs - detected_subs)
    tn = len((all_merchants - true_subs) - detected_subs)
    p, r, f1 = _prf(tp, fp, fn)

    print("=" * 56)
    print(" SUBSCRIPTION DETECTION")
    print("=" * 56)
    print(f"  true subscriptions : {len(true_subs)}")
    print(f"  correctly detected : {tp}")
    print(f"  false positives    : {fp}  (one-off spends wrongly flagged)")
    print(f"  missed             : {fn}")
    print(f"  precision={p:.0%}  recall={r:.0%}  F1={f1:.0%}")

    # --- Hidden detection ---
    h_tp = len(detected_hidden & true_hidden)
    h_fp = len(detected_hidden - true_hidden)
    h_fn = len(true_hidden - detected_hidden)
    hp, hr, hf1 = _prf(h_tp, h_fp, h_fn)

    print("\n" + "=" * 56)
    print(" HIDDEN SUBSCRIPTION DETECTION  (the whole point)")
    print("=" * 56)
    print(f"  truly hidden       : {len(true_hidden)}")
    print(f"  correctly flagged  : {h_tp}  ->  {h_tp}/{len(true_hidden)}")
    print(f"  false alarms       : {h_fp}")
    print(f"  missed             : {h_fn}")
    print(f"  precision={hp:.0%}  recall={hr:.0%}  F1={hf1:.0%}")

    if h_fn:
        print(f"\n  missed hidden subs: {sorted(true_hidden - detected_hidden)}")
    if h_fp:
        print(f"  false alarms      : {sorted(detected_hidden - true_hidden)}")

    print("\nHeadline for the pitch:")
    print(f"  → Detected {h_tp}/{len(true_hidden)} hidden subscriptions "
          f"with {fp} false positives across {len(all_merchants)} merchants.")


if __name__ == "__main__":
    main()
