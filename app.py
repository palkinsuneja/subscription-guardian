"""
app.py  (Streamlit dashboard — Architecture step 06)
----------------------------------------------------
The face of Subscription Guardian.

Shows:
  - headline numbers (total subs, hidden count, monthly leak)
  - every subscription, HIDDEN ones highlighted in red
  - upcoming charges due soon
  - an "Ask Guardian" box for natural-language questions

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import recurrence  # noqa: E402
import classifier  # noqa: E402
import risk  # noqa: E402
import agent  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transactions.csv")

st.set_page_config(page_title="Subscription Guardian", page_icon="🛡️", layout="wide")


@st.cache_data
def load_scored():
    txns = pd.read_csv(DATA)
    scored = risk.score(classifier.classify(recurrence.find_recurring(txns)))
    return scored, risk.portfolio_summary(scored)


scored, summary = load_scored()

st.title("🛡️ Subscription Guardian")
st.caption("Catches the recurring charges running silently in the background — "
           "before the money is pulled.")

# --- headline metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Subscriptions found", summary["total_subscriptions"])
c2.metric("Hidden / unnoticed", summary["hidden_count"])
c3.metric("Monthly spend", f"₹{summary['total_monthly_spend']:,.0f}")
c4.metric("Hidden leak / month", f"₹{summary['hidden_monthly_spend']:,.0f}",
          help="Money going to subscriptions you may not have chosen.")

mode = "Groq LLM" if agent._groq_available() else "offline template"
st.info(f"AI agent mode: **{mode}**  ·  set GROQ_API_KEY for live LLM alerts.", icon="🤖")

# --- upcoming charges ---
upcoming = scored[scored["upcoming_warning"].apply(lambda x: isinstance(x, str) and bool(x))]
if not upcoming.empty:
    st.subheader("⏰ Charges due soon")
    for _, r in upcoming.iterrows():
        st.warning(f"**{r['merchant']}** — {r['upcoming_warning']}")

# --- hidden subscriptions (the star of the show) ---
st.subheader("🚨 Hidden subscriptions")
hidden = scored[scored["is_hidden"]]
if hidden.empty:
    st.success("No hidden subscriptions detected.")
else:
    for _, r in hidden.iterrows():
        with st.container(border=True):
            top = st.columns([3, 1, 1])
            top[0].markdown(f"### :red[{r['merchant']}]")
            top[1].metric("Risk", f"{r['risk_score']}/100")
            top[2].metric("₹/month", f"{float(r['monthly_cost'] or r['typical_amount']):,.0f}")
            st.caption(f"**Why flagged:** {r['flags']}")
            st.write(agent.alert(r.to_dict()))

# --- all subscriptions table ---
st.subheader("All detected subscriptions")


def _row_style(row):
    color = "background-color: rgba(176,42,31,0.12)" if row["is_hidden"] else ""
    return [color] * len(row)


table = scored[["merchant", "category", "monthly_cost", "currency", "method",
                "risk_band", "risk_score", "is_hidden", "next_charge_est", "flags"]].copy()
st.dataframe(table.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)

# --- ask guardian ---
st.subheader("💬 Ask Guardian")
q = st.text_input("Ask about your subscriptions",
                  placeholder="e.g. What are my hidden subscriptions? How much am I losing per month?")
if q:
    st.write(agent.ask(scored, q))
