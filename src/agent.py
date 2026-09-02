"""
agent.py  (Architecture step 05)
--------------------------------
The AI Agent.

Two jobs:
  1. alert()  -> turn a flagged subscription into a short, human warning +
                 a clear "should I cancel?" recommendation.
  2. ask()    -> answer free-text questions about the user's subscriptions
                 ("what are all my hidden subscriptions?", "how much am I
                 losing per month?") grounded in THEIR data.

Design choice: the agent uses Groq (Llama) when a GROQ_API_KEY is available,
so you get real natural-language generation for the demo. If no key is set,
it falls back to a deterministic template so the project ALWAYS runs -- no
dead demo in front of judges. Same interface either way.
"""

from __future__ import annotations

import json
import os

import pandas as pd

GROQ_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# LLM plumbing (optional)
# ---------------------------------------------------------------------------

def _groq_available() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def _groq_chat(system: str, user: str) -> str | None:
    """Call Groq if configured; return None on any failure so callers fall back."""
    if not _groq_available():
        return None
    try:
        from groq import Groq  # imported lazily so the module loads without the dep
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. Per-subscription alert
# ---------------------------------------------------------------------------

_ALERT_SYSTEM = (
    "You are Subscription Guardian, a calm, trustworthy assistant that protects "
    "people from silent recurring charges. Be concise (2-3 sentences), specific, "
    "and end with a clear recommendation: CANCEL, REVIEW, or KEEP. Never invent "
    "numbers beyond what you are given."
)


def _template_alert(row: dict) -> str:
    name = row["merchant"]
    amt = row.get("monthly_cost") or row["typical_amount"]
    band = row["risk_band"]
    flags = row.get("flags", "")
    rec = "CANCEL" if row["risk_score"] >= 60 else ("REVIEW" if row["is_hidden"] else "KEEP")
    uw = row.get("upcoming_warning")
    warn = f" {uw}." if isinstance(uw, str) and uw else ""
    reason = f" Why it stood out: {flags}." if row.get("is_hidden") else ""
    return (f"[{band} RISK] {name} is charging you about ₹{float(amt):,.0f}/month.{reason}{warn} "
            f"Recommendation: {rec}.")


def alert(row: dict) -> str:
    """Human-readable alert for one scored subscription."""
    user = (
        "Write a one-paragraph alert for this recurring charge.\n"
        f"Data: {json.dumps({k: row.get(k) for k in ['merchant','monthly_cost','typical_amount','currency','method','risk_band','risk_score','is_hidden','flags','upcoming_warning']}, default=str)}"
    )
    return _groq_chat(_ALERT_SYSTEM, user) or _template_alert(row)


# ---------------------------------------------------------------------------
# 2. Free-text Q&A grounded in the user's subscriptions
# ---------------------------------------------------------------------------

_QA_SYSTEM = (
    "You are Subscription Guardian. Answer ONLY from the JSON subscription data "
    "provided. Be concrete, use the numbers given, and keep it short. If the data "
    "doesn't contain the answer, say so."
)


def _context_json(scored: pd.DataFrame) -> str:
    keep = ["merchant", "monthly_cost", "typical_amount", "currency", "method",
            "risk_band", "risk_score", "is_hidden", "flags", "next_charge_est"]
    keep = [c for c in keep if c in scored.columns]
    return scored[keep].to_json(orient="records")


def _template_answer(scored: pd.DataFrame, question: str) -> str:
    q = question.lower()
    hidden = scored[scored["is_hidden"]]
    monthly_col = pd.to_numeric(scored["monthly_cost"], errors="coerce").fillna(scored["typical_amount"])

    if "hidden" in q or "unknown" in q or "surprise" in q:
        if hidden.empty:
            return "Good news — I found no hidden subscriptions."
        names = ", ".join(f"{m} (₹{c:,.0f}/mo)" for m, c in
                          zip(hidden["merchant"], pd.to_numeric(hidden["monthly_cost"],
                              errors="coerce").fillna(hidden["typical_amount"])))
        total = pd.to_numeric(hidden["monthly_cost"], errors="coerce").fillna(hidden["typical_amount"]).sum()
        return (f"You have {len(hidden)} hidden subscription(s): {names}. "
                f"Together that's ~₹{total:,.0f}/month leaking quietly.")

    if "how much" in q or "total" in q or "spend" in q or "losing" in q:
        total = monthly_col.sum()
        htotal = pd.to_numeric(hidden["monthly_cost"], errors="coerce").fillna(hidden["typical_amount"]).sum()
        return (f"You're spending ~₹{total:,.0f}/month across {len(scored)} subscriptions. "
                f"Of that, ~₹{htotal:,.0f}/month is on {len(hidden)} hidden one(s) you may not have chosen.")

    if "foreign" in q or "dollar" in q or "usd" in q:
        foreign = scored[scored["currency"].str.upper() != "INR"]
        if foreign.empty:
            return "No foreign-currency subscriptions found."
        names = ", ".join(foreign["merchant"])
        return f"Foreign-currency charges: {names}. These bill in {foreign['currency'].iloc[0]} and are the hardest to notice."

    if "cancel" in q or "risk" in q or "high" in q:
        high = scored[scored["risk_band"] == "HIGH"]
        if high.empty:
            return "Nothing is high-risk right now."
        names = ", ".join(high["merchant"])
        return f"I'd review/cancel these HIGH-risk charges first: {names}."

    # default
    total = monthly_col.sum()
    return (f"You have {len(scored)} subscriptions (~₹{total:,.0f}/month), "
            f"{len(hidden)} of them hidden. Ask me about hidden ones, foreign charges, "
            f"total spend, or what to cancel.")


def ask(scored: pd.DataFrame, question: str) -> str:
    """Answer a natural-language question about the user's subscriptions."""
    user = f"Subscriptions JSON:\n{_context_json(scored)}\n\nQuestion: {question}"
    return _groq_chat(_QA_SYSTEM, user) or _template_answer(scored, question)


if __name__ == "__main__":
    import os as _os
    from recurrence import find_recurring
    from classifier import classify
    from risk import score

    here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    txns = pd.read_csv(_os.path.join(here, "data", "transactions.csv"))
    scored = score(classify(find_recurring(txns)))

    print("MODE:", "Groq LLM" if _groq_available() else "offline template (no GROQ_API_KEY)")
    print("\n--- Sample alerts ---")
    for _, r in scored[scored["is_hidden"]].head(3).iterrows():
        print("•", alert(r.to_dict()))

    print("\n--- Q&A ---")
    for q in ["What are my hidden subscriptions?",
              "How much am I losing per month?",
              "Which foreign charges do I have?"]:
        print(f"\nQ: {q}\nA: {ask(scored, q)}")
