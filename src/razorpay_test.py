"""
razorpay_test.py
----------------
Demonstrates a REAL recurring-payment flow using Razorpay's TEST mode, then
feeds the resulting charge into Subscription Guardian to show it gets caught.

No real money moves: test mode uses sandbox keys and test cards.

Setup (free):
  1. Sign up at https://dashboard.razorpay.com and switch to TEST mode.
  2. Settings -> API Keys -> generate Test keys.
  3. export RAZORPAY_KEY_ID=rzp_test_xxx
     export RAZORPAY_KEY_SECRET=xxx
  4. pip install razorpay
  5. python src/razorpay_test.py

What it does:
  - creates a Plan (e.g. ₹1920 / month) and a Subscription against it,
  - (in a real demo you'd authorize it with a Razorpay test card),
  - appends the simulated recurring charges to the transaction history,
  - runs the detector and prints whether Guardian flagged it.

This file degrades gracefully: with no keys it runs a SIMULATED version so the
pipeline demo still works end-to-end.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import recurrence  # noqa: E402
import classifier  # noqa: E402
import risk  # noqa: E402

MERCHANT = "TestApp Premium (US)"
AMOUNT_INR = 1920.0
INTERVAL_DAYS = 30


def _keys() -> tuple[str | None, str | None]:
    return os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")


def create_test_subscription():
    """Create a real Plan + Subscription in Razorpay TEST mode (no real money)."""
    key_id, key_secret = _keys()
    if not (key_id and key_secret):
        print("No Razorpay test keys set -> running SIMULATED recurring charge instead.")
        return None
    try:
        import razorpay
        client = razorpay.Client(auth=(key_id, key_secret))
        plan = client.plan.create({
            "period": "monthly",
            "interval": 1,
            "item": {"name": MERCHANT, "amount": int(AMOUNT_INR * 100), "currency": "INR"},
        })
        sub = client.subscription.create({
            "plan_id": plan["id"],
            "total_count": 12,
            "customer_notify": 1,
        })
        print(f"Created TEST plan {plan['id']} and subscription {sub['id']}.")
        print(f"Authorize it with a Razorpay test card at: {sub.get('short_url')}")
        return sub
    except Exception as e:  # pragma: no cover
        print(f"Razorpay call failed ({e}) -> falling back to simulation.")
        return None


def simulate_charges(n: int = 6) -> pd.DataFrame:
    """Produce n monthly charges as if the test subscription had billed."""
    start = date(2026, 3, 1)
    rows = []
    for i in range(n):
        rows.append({
            "txn_id": f"RZTEST{i:03d}",
            "date": (start + timedelta(days=INTERVAL_DAYS * i)).isoformat(),
            "merchant": MERCHANT,
            "category": "productivity",
            "amount": round(AMOUNT_INR * (1 + 0.01 * (i % 3)), 2),
            "currency": "USD",
            "method": "CreditCard",
        })
    return pd.DataFrame(rows)


def main():
    create_test_subscription()

    # Merge the test subscription's charges into the existing history.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base = pd.read_csv(os.path.join(here, "data", "transactions.csv"))
    combined = pd.concat([base, simulate_charges()], ignore_index=True)

    scored = risk.score(classifier.classify(recurrence.find_recurring(combined)))
    hit = scored[scored["merchant"] == MERCHANT]
    if not hit.empty:
        r = hit.iloc[0]
        print(f"\n✅ Guardian caught the test subscription '{MERCHANT}':")
        print(f"   risk={r['risk_score']}/100  band={r['risk_band']}  hidden={r['is_hidden']}")
        print(f"   flags: {r['flags']}")
    else:
        print(f"\n❌ '{MERCHANT}' was not flagged — check the recurrence thresholds.")


if __name__ == "__main__":
    main()
