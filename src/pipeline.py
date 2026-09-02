"""
pipeline.py
-----------
Ties the four detection stages into one call, so the API and the dashboard
share exactly the same logic:

    transactions.csv
        -> recurrence.find_recurring   (step 02)
        -> classifier.classify         (step 03)
        -> risk.score                  (step 04)
        -> scored subscriptions  (+ agent uses these in step 05)
"""

from __future__ import annotations

import os

import pandas as pd

from .recurrence import find_recurring
from .classifier import classify
from .risk import score, portfolio_summary


def _default_csv() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "data", "transactions.csv")


def run(txns: pd.DataFrame | None = None, csv_path: str | None = None):
    """Run the full detection pipeline. Returns (scored_df, summary_dict)."""
    if txns is None:
        txns = pd.read_csv(csv_path or _default_csv())
    scored = score(classify(find_recurring(txns)))
    return scored, portfolio_summary(scored)


# Allow running as a plain script too (python src/pipeline.py) without the
# package-relative imports, for quick checks.
if __name__ == "__main__":
    import importlib
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    recurrence = importlib.import_module("recurrence")
    classifier = importlib.import_module("classifier")
    risk = importlib.import_module("risk")
    df = pd.read_csv(_default_csv())
    scored = risk.score(classifier.classify(recurrence.find_recurring(df)))
    print(scored[["merchant", "monthly_cost", "risk_band", "is_hidden"]].to_string(index=False))
    print(risk.portfolio_summary(scored))
