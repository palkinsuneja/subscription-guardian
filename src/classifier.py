"""
classifier.py  (Architecture step 03)
-------------------------------------
The Subscription Classifier.

The Recurrence Engine already told us WHICH merchants repeat. This module
decides, for each one, the things a human actually cares about:

    - is_subscription : yes (they're all recurring by this point)
    - is_hidden       : is this one likely OFF the user's radar?
    - flags           : *why* it's suspicious (foreign, trial-converted,
                         forgotten, on-credit-card, ...)

"Hidden" is the whole point of the product, so we compute it explicitly.
A charge is hidden-ish when it has the fingerprints of the traps Palkin hit:
    * foreign currency on a credit card  (no consent screen)
    * started mid-history after a gap    (a free trial that turned paid)
    * hasn't been "used" recently / small + forgotten
    * rides on a credit card rather than an explicit UPI mandate

We keep it interpretable (rules + a simple score) instead of a black box,
because for a risk product the user must be able to see WHY.
"""

from __future__ import annotations

import pandas as pd

# A user-maintained allow-list of subscriptions they KNOW about.
# In a real product this comes from the user; here it models "user awareness".
KNOWN_TO_USER = {"Netflix", "Spotify", "CultFit Gym", "Jio Recharge"}

HISTORY_START = pd.Timestamp("2026-01-01")


def classify(recurring: pd.DataFrame, known_to_user: set[str] | None = None) -> pd.DataFrame:
    """Add is_subscription / is_hidden / flags / hidden_score to each recurring row."""
    known = KNOWN_TO_USER if known_to_user is None else known_to_user
    df = recurring.copy()

    out_rows = []
    for _, r in df.iterrows():
        flags = []
        hidden_score = 0

        # --- Signal 1: foreign currency (Palkin's $23 trap) ---
        if str(r["currency"]).upper() != "INR":
            flags.append("foreign currency")
            hidden_score += 40

        # --- Signal 2: on a credit card (the blind spot vs. visible UPI mandate) ---
        if str(r["method"]).lower() == "creditcard":
            flags.append("credit-card autopay")
            hidden_score += 15

        # --- Signal 3: started late -> likely a free trial that converted ---
        first_seen = pd.Timestamp(r["first_seen"])
        days_after_start = (first_seen - HISTORY_START).days
        if days_after_start > 45:
            flags.append("started mid-history (likely trial → paid)")
            hidden_score += 25

        # --- Signal 4: not explicitly known to the user ---
        if r["merchant"] not in known:
            flags.append("not in your known list")
            hidden_score += 20

        # --- Signal 5: small + quietly persistent -> classic 'forgotten' ---
        if r["typical_amount"] <= 400 and r["occurrences"] >= 6 and r["merchant"] not in known:
            flags.append("small & long-running (forgotten?)")
            hidden_score += 10

        hidden_score = min(hidden_score, 100)
        is_hidden = hidden_score >= 40  # threshold: needs a real reason, not just 'unknown'

        row = r.to_dict()
        row["is_subscription"] = True
        row["is_hidden"] = bool(is_hidden)
        row["hidden_score"] = int(hidden_score)
        row["flags"] = "; ".join(flags) if flags else "looks expected"
        out_rows.append(row)

    result = pd.DataFrame(out_rows)
    return result.sort_values(["is_hidden", "hidden_score"], ascending=False, ignore_index=True)


if __name__ == "__main__":
    import os
    from recurrence import find_recurring

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txns = pd.read_csv(os.path.join(here, "data", "transactions.csv"))
    rec = find_recurring(txns)
    cls = classify(rec)
    show = cls[["merchant", "typical_amount", "currency", "method",
                "is_hidden", "hidden_score", "flags"]]
    print(show.to_string(index=False))
