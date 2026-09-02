"""
recurrence.py  (Architecture step 02)
-------------------------------------
The Recurrence Engine.

Goal: from a flat list of transactions, find groups of charges that REPEAT
in a subscription-like way -> same merchant, near-constant amount, and a
regular time gap between charges (e.g. every ~30 days).

A grocery bill also repeats, but its AMOUNT is all over the place and the gaps
are irregular. A subscription is boring and predictable -> that predictability
is exactly the signal we exploit.

Output: one row per (merchant) that looks recurring, with features the
classifier and risk scorer will use downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _coefficient_of_variation(values: np.ndarray) -> float:
    """std / mean. Low = very consistent (subscription-like)."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or values.mean() == 0:
        return 1.0
    return float(values.std() / values.mean())


def _regularity(gaps: np.ndarray) -> float:
    """How regular are the day-gaps between charges? 1.0 = perfectly regular."""
    gaps = np.asarray(gaps, dtype=float)
    if len(gaps) == 0:
        return 0.0
    if gaps.mean() == 0:
        return 0.0
    cv = gaps.std() / gaps.mean()
    return float(max(0.0, 1.0 - cv))  # small variation -> close to 1


def find_recurring(
    txns: pd.DataFrame,
    min_occurrences: int = 3,
    max_amount_cv: float = 0.15,
    min_regularity: float = 0.5,
) -> pd.DataFrame:
    """
    Detect recurring charges.

    A merchant is considered recurring when:
      - it appears >= min_occurrences times,
      - its amounts are near-constant (coefficient of variation <= max_amount_cv),
      - and the gaps between charges are fairly regular.

    Returns a DataFrame of candidate subscriptions with useful features.
    """
    df = txns.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    results = []
    for merchant, g in df.groupby("merchant"):
        g = g.sort_values("date")
        n = len(g)
        if n < min_occurrences:
            continue

        amounts = g["amount"].to_numpy()
        dates = g["date"].to_numpy()
        gaps = np.diff(dates).astype("timedelta64[D]").astype(float)

        amount_cv = _coefficient_of_variation(amounts)
        regularity = _regularity(gaps)
        median_gap = float(np.median(gaps)) if len(gaps) else 0.0

        # Core recurrence test.
        is_recurring = (amount_cv <= max_amount_cv) and (regularity >= min_regularity)
        if not is_recurring:
            continue

        last_date = pd.Timestamp(dates[-1])
        next_charge = last_date + pd.Timedelta(days=round(median_gap)) if median_gap else pd.NaT

        results.append({
            "merchant": merchant,
            "category": g["category"].mode().iat[0],
            "occurrences": n,
            "typical_amount": round(float(np.median(amounts)), 2),
            "amount_cv": round(amount_cv, 4),
            "median_gap_days": round(median_gap, 1),
            "regularity": round(regularity, 3),
            "currency": g["currency"].mode().iat[0],
            "method": g["method"].mode().iat[0],
            "first_seen": pd.Timestamp(dates[0]).date().isoformat(),
            "last_seen": last_date.date().isoformat(),
            "next_charge_est": next_charge.date().isoformat() if pd.notna(next_charge) else None,
            "monthly_cost": round(float(np.median(amounts)) * (30.0 / median_gap), 2) if median_gap else None,
        })

    cols = ["merchant", "category", "occurrences", "typical_amount", "amount_cv",
            "median_gap_days", "regularity", "currency", "method",
            "first_seen", "last_seen", "next_charge_est", "monthly_cost"]
    return pd.DataFrame(results, columns=cols).sort_values("monthly_cost", ascending=False, ignore_index=True)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txns = pd.read_csv(os.path.join(here, "data", "transactions.csv"))
    rec = find_recurring(txns)
    print(f"Found {len(rec)} recurring merchants:\n")
    print(rec.to_string(index=False))
