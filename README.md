# 🛡️ Subscription Guardian

**An AI risk agent that catches the recurring charges running silently in the background — and warns you *before* the money is pulled.**

Razorpay AI Buildathon 2026 · Track 2: AI Risk Manager · by Palkin Suneja

---

## The problem

When you set up UPI autopay, the screen clearly tells you *"this is recurring."*
But on a **credit card** — especially **foreign subscriptions** — autopay switches
on quietly. No warning, no consent screen. Next month, the money is simply gone.

> Real story that started this project: a foreign subscription taken on a credit
> card, believed to be a one-month purchase. Nowhere did it say autopay. The next
> month, ~$23 was pulled automatically. It had to be hunted down and cancelled
> manually — but that month's money was already lost.

Free trials that silently turn paid, old subscriptions you forgot, and foreign
recurring charges nobody is watching — that blind spot is what Subscription
Guardian fills.

## What it does

1. **Detects recurring charges** - finds charges that repeat with a subscription-like fingerprint (near-constant amount + regular interval), while ignoring irregular one-off spends like groceries.
2. **Separates hidden from known** - flags the subscriptions likely *off your radar* (foreign, trial-converted, forgotten, credit-card autopay).
3. **Warns before the charge hits** - surfaces charges due in the next few days.
4. **Risk-scores every merchant** - a 0–100 score with LOW / MEDIUM / HIGH bands.
5. **Answers your questions** - an AI agent answers *"what are my hidden subscriptions?"*, *"how much am I losing per month?"* grounded in your data.

## How it works (architecture)

```
transactions.csv
    │
    ▼
[02] Recurrence Engine     src/recurrence.py   → find repeating charges (amount + interval)
    │
    ▼
[03] Subscription Classifier src/classifier.py → subscription? hidden? why? (rules + score)
    │
    ▼
[04] Risk Scorer           src/risk.py         → 0–100 risk + upcoming-charge warnings
    │
    ▼
[05] AI Agent              src/agent.py        → plain-language alerts + Q&A (Groq/Llama)
    │
    ▼
[06] Dashboard             app.py (Streamlit)  ← served via api.py (FastAPI)
```

## Results (on synthetic ground-truth data)

Run `python evaluate.py`:

```
SUBSCRIPTION DETECTION   precision=100%  recall=100%   (0 false positives across 18 merchants)
HIDDEN DETECTION         5 / 5 hidden subscriptions correctly flagged, 0 false alarms
```

> Numbers are on a labelled **synthetic** dataset (`src/generate_data.py`) where the
> hidden subscriptions are known in advance, so detection can be measured honestly.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the synthetic transaction history (with hidden subs planted)
python src/generate_data.py

# 2. See the detection + accuracy
python evaluate.py

# 3. Run the dashboard
streamlit run app.py

# (optional) run the API
uvicorn api:app --reload      # docs at http://127.0.0.1:8000/docs
```

### Enable the live LLM (optional)
The AI agent uses **Groq (Llama 3.3)** when a key is present, and falls back to
deterministic templates otherwise (so the demo never breaks):

```bash
export GROQ_API_KEY=your_key_here
```

### Razorpay test-mode demo (optional)
Shows a **real recurring payment** in Razorpay TEST mode being caught by the detector
(no real money — sandbox keys + test cards):

```bash
export RAZORPAY_KEY_ID=rzp_test_xxx
export RAZORPAY_KEY_SECRET=xxx
python src/razorpay_test.py
```

## Project structure

```
subscription-guardian/
├── data/
│   ├── transactions.csv      # generated input
│   └── ground_truth.csv      # answer key (for evaluation)
├── src/
│   ├── generate_data.py      # synthetic transactions + planted hidden subs
│   ├── recurrence.py         # [02] recurrence engine
│   ├── classifier.py         # [03] subscription / hidden classifier
│   ├── risk.py               # [04] risk scorer
│   ├── agent.py              # [05] Groq/Llama agent (+ offline fallback)
│   ├── razorpay_test.py      # Razorpay test-mode integration
│   └── pipeline.py           # ties the stages together
├── api.py                    # FastAPI backend
├── app.py                    # Streamlit dashboard
├── evaluate.py               # accuracy vs ground truth
├── requirements.txt
└── README.md
```

## Scope — deliberately focused

**Not** building fraud/QR detection, failed-payment recovery, or a full finance
dashboard. One problem - hidden recurring charges - solved deeply.

## Next steps
- Harden the dataset with trickier decoys (price hikes, annual plans, near-monthly rent) to stress-test detection.
- Learn the "known to user" list from user feedback instead of a static allow-list.
- Push real Razorpay webhook events through the same pipeline.
