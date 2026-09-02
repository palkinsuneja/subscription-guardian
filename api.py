"""
api.py  (FastAPI backend)
-------------------------
Exposes the detection pipeline over HTTP so the dashboard (or any client)
can consume it. Steps 02-05 of the architecture live behind these endpoints.

Run:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs for interactive docs.

Endpoints:
    GET  /health                 -> liveness
    GET  /subscriptions          -> all detected subscriptions, risk-scored
    GET  /subscriptions/hidden   -> only the hidden ones
    GET  /summary                -> portfolio headline numbers
    POST /ask   {"question": ..} -> natural-language answer over your data
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import recurrence  # noqa: E402
import classifier  # noqa: E402
import risk  # noqa: E402
import agent  # noqa: E402

app = FastAPI(title="Subscription Guardian API", version="1.0")

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transactions.csv")


def _scored() -> pd.DataFrame:
    txns = pd.read_csv(_DATA)
    return risk.score(classifier.classify(recurrence.find_recurring(txns)))


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records: NaN/NaT -> None (raw NaN isn't valid JSON)."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "llm": "groq" if agent._groq_available() else "offline-template"}


@app.get("/subscriptions")
def subscriptions():
    return _records(_scored())


@app.get("/subscriptions/hidden")
def hidden():
    s = _scored()
    return _records(s[s["is_hidden"]])


@app.get("/summary")
def summary():
    return risk.portfolio_summary(_scored())


@app.post("/ask")
def ask(q: Question):
    return {"question": q.question, "answer": agent.ask(_scored(), q.question)}
