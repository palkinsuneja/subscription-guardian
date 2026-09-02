"""
risk.py  (Architecture step 04)
-------------------------------
The Risk Scorer.

Turns the classifier's signals into a single 0-100 RISK score per subscription,
plus a band (LOW / MEDIUM / HIGH) and a short "upcoming charge" warning when a
charge is due soon. This is the number the dashboard leads with.

Risk is not the same as "hidden": a known, domestic Netflix charge is not
hidden AND not risky. A foreign, credit-card, trial-converted charge you never
recognised is both hidden and high-risk.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def _band(score: int) -> str:
    if score >= 60:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


def score(classified: pd.DataFrame, today: date | None = None) -> pd.DataFrame:
    """Add risk_score, risk_band and days_to_next / upcoming warning."""
    today = today or date(2026, 8, 25)
    df = classified.copy()

    rows = []
    for _, r in df.iterrows():
        # Start from the hidden score, then layer risk-specific weights.
        risk = int(r.get("hidden_score", 0))

        # Foreign charges carry more *financial* risk (forex + hard to cancel).
        if str(r["currency"]).upper() != "INR":
            risk += 10

        # Bigger money at stake -> higher risk.
        monthly = float(r.get("monthly_cost") or r["typical_amount"])
        if monthly >= 2000:
            risk += 15
        elif monthly >= 1000:
            risk += 8

        risk = max(0, min(100, risk))

        # Upcoming-charge warning.
        days_to_next = None
        upcoming = None
        if r.get("next_charge_est"):
            nxt = pd.Timestamp(r["next_charge_est"]).date()
            days_to_next = (nxt - today).days
            if 0 <= days_to_next <= 7:
                upcoming = f"₹{monthly:,.0f} due in {days_to_next} day(s) on {nxt.isoformat()}"

        row = r.to_dict()
        row["risk_score"] = risk
        row["risk_band"] = _band(risk)
        row["days_to_next"] = days_to_next
        row["upcoming_warning"] = upcoming
        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values("risk_score", ascending=False, ignore_index=True)


def portfolio_summary(scored: pd.DataFrame) -> dict:
    """Headline numbers for the top of the dashboard."""
    total_monthly = float(pd.to_numeric(scored["monthly_cost"], errors="coerce").fillna(
        scored["typical_amount"]).sum())
    hidden = scored[scored["is_hidden"]]
    return {
        "total_subscriptions": int(len(scored)),
        "hidden_count": int(scored["is_hidden"].sum()),
        "high_risk_count": int((scored["risk_band"] == "HIGH").sum()),
        "total_monthly_spend": round(total_monthly, 2),
        "hidden_monthly_spend": round(float(pd.to_numeric(
            hidden["monthly_cost"], errors="coerce").fillna(hidden["typical_amount"]).sum()), 2),
    }


if __name__ == "__main__":
    import os
    from recurrence import find_recurring
    from classifier import classify

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txns = pd.read_csv(os.path.join(here, "data", "transactions.csv"))
    scored = score(classify(find_recurring(txns)))
    show = scored[["merchant", "monthly_cost", "risk_score", "risk_band",
                   "is_hidden", "upcoming_warning"]]
    print(show.to_string(index=False))
    print("\nPortfolio:", portfolio_summary(scored))
