"""
generate_data.py
-----------------
Builds a realistic synthetic transaction history for one user, with:
  - regular one-off spends (groceries, fuel, food delivery, ...)  -> NOT subscriptions
  - "known" subscriptions the user is aware of (Netflix, gym, ...)
  - "hidden" subscriptions -> the villain of our story:
        * free trials that silently turned paid
        * foreign credit-card recurring charges with no consent screen
        * a forgotten old subscription

We also save a ground-truth file so we can MEASURE how well the detector works.

Run:
    python src/generate_data.py
Output:
    data/transactions.csv        -> the input our system sees
    data/ground_truth.csv        -> the answer key (for evaluation only)
"""

import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# The history window: ~8 months so a 30-day subscription repeats 7-8 times.
START = date(2026, 1, 1)
END = date(2026, 8, 25)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------------------
# 1. Merchant catalogues
# ---------------------------------------------------------------------------

# One-off / irregular merchants -> these should NOT be flagged as subscriptions,
# even though some (like groceries) show up often. Amounts vary a lot.
ONE_OFF_MERCHANTS = [
    ("BigBasket",        "groceries",     (250, 2200)),
    ("Zomato",           "food",          (120, 900)),
    ("Swiggy",           "food",          (110, 850)),
    ("Uber",             "transport",     (60, 700)),
    ("IndianOil",        "fuel",          (500, 3000)),
    ("Amazon",           "shopping",      (150, 6000)),
    ("Myntra",           "shopping",      (400, 4500)),
    ("Local Pharmacy",   "health",        (80, 1200)),
    ("Metro Card",       "transport",     (100, 500)),
]

# Known subscriptions: user knowingly signed up. Domestic, recognisable.
# (name, category, amount, interval_days, is_foreign, is_trial_to_paid)
KNOWN_SUBS = [
    ("Netflix",        "entertainment", 649.0,  30, False, False),
    ("Spotify",        "entertainment", 119.0,  30, False, False),
    ("CultFit Gym",    "fitness",       1300.0, 30, False, False),
    ("Jio Recharge",   "telecom",       299.0,  28, False, False),
]

# HIDDEN subscriptions: the ones Subscription Guardian must catch.
# Each has a "reason" describing why it is sneaky (used only in ground truth).
HIDDEN_SUBS = [
    # Foreign recurring charge on credit card, no consent screen (Palkin's $23 story).
    # ~ $23 -> stored in INR at ~1920, small monthly drift like a forex conversion.
    ("Quillbot Premium",  "productivity", 1920.0, 30, True,  False, "foreign card autopay, no consent screen"),
    ("Canva Pro (US)",    "productivity", 3080.0, 30, True,  False, "foreign recurring, billed in USD"),
    # Free trial that silently converted to paid after 1 month.
    ("Notion AI",         "productivity", 800.0,  30, False, True,  "free trial silently turned paid"),
    ("Grammarly",         "productivity", 1000.0, 30, False, True,  "trial converted to paid subscription"),
    # A forgotten old subscription the user never uses anymore.
    ("Hotstar",           "entertainment", 299.0, 30, False, False, "forgotten, unused for months"),
]


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def _daterange_points(start, end, interval_days, jitter=2):
    """Yield roughly-periodic dates from start..end, every interval_days
    with a small +/- jitter (real autopay isn't perfectly on the dot)."""
    d = start + timedelta(days=random.randint(0, interval_days))
    while d <= end:
        shift = random.randint(-jitter, jitter)
        point = d + timedelta(days=shift)
        if start <= point <= end:
            yield point
        d += timedelta(days=interval_days)


def _amount_with_drift(base, foreign):
    """Subscriptions are near-constant. Foreign ones drift a little (forex)."""
    if foreign:
        return round(base * random.uniform(0.97, 1.04), 2)
    # tiny occasional rounding difference
    return round(base + random.choice([0, 0, 0, 0.0, -0.0]), 2)


# ---------------------------------------------------------------------------
# 3. Build the rows
# ---------------------------------------------------------------------------

def build():
    rows = []

    # --- one-off spends: frequent but irregular amounts, no fixed cadence ---
    n_oneoff = 520
    span_days = (END - START).days
    for _ in range(n_oneoff):
        name, cat, (lo, hi) = random.choice(ONE_OFF_MERCHANTS)
        d = START + timedelta(days=random.randint(0, span_days))
        amt = round(random.uniform(lo, hi), 2)
        rows.append({
            "date": d,
            "merchant": name,
            "category": cat,
            "amount": amt,
            "currency": "INR",
            "method": random.choice(["UPI", "UPI", "DebitCard", "CreditCard"]),
        })

    # --- known subscriptions ---
    for name, cat, amt, interval, foreign, _trial in KNOWN_SUBS:
        for d in _daterange_points(START, END, interval):
            rows.append({
                "date": d,
                "merchant": name,
                "category": cat,
                "amount": _amount_with_drift(amt, foreign),
                "currency": "INR",
                "method": "UPI",  # domestic autopay -> shows up on UPI mandate
            })

    # --- hidden subscriptions ---
    for name, cat, amt, interval, foreign, trial, _reason in HIDDEN_SUBS:
        # A trial-to-paid one starts LATER (after a 1-month free trial).
        sub_start = START
        if trial:
            sub_start = START + timedelta(days=random.randint(60, 100))
        for d in _daterange_points(sub_start, END, interval):
            rows.append({
                "date": d,
                "merchant": name,
                "category": cat,
                "amount": _amount_with_drift(amt, foreign),
                "currency": "USD" if foreign else "INR",
                # Foreign hidden ones ride on the credit card (the blind spot).
                "method": "CreditCard" if foreign else "CreditCard",
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{100000 + i}" for i in range(len(df))])
    return df


def build_ground_truth():
    gt = []
    for name, *_rest in KNOWN_SUBS:
        gt.append({"merchant": name, "is_subscription": True, "is_hidden": False, "reason": "known / user-aware"})
    for name, cat, amt, interval, foreign, trial, reason in HIDDEN_SUBS:
        gt.append({"merchant": name, "is_subscription": True, "is_hidden": True, "reason": reason})
    for name, cat, rng in ONE_OFF_MERCHANTS:
        gt.append({"merchant": name, "is_subscription": False, "is_hidden": False, "reason": "one-off spend"})
    return pd.DataFrame(gt)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = build()
    gt = build_ground_truth()

    txn_path = os.path.join(DATA_DIR, "transactions.csv")
    gt_path = os.path.join(DATA_DIR, "ground_truth.csv")
    df.to_csv(txn_path, index=False)
    gt.to_csv(gt_path, index=False)

    n_hidden = int(gt["is_hidden"].sum())
    n_sub = int(gt["is_subscription"].sum())
    print(f"Wrote {len(df)} transactions -> {txn_path}")
    print(f"Merchants: {df['merchant'].nunique()} | subscriptions: {n_sub} "
          f"(hidden: {n_hidden}) | ground truth -> {gt_path}")


if __name__ == "__main__":
    main()
