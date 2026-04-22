# LotSense

Local car price estimator + automated Marketplace conversation bot.
Collects Facebook Marketplace listings in your area, trains a mileage-based model to estimate fair market value, and handles buyer conversations automatically until they ask for your location.

---

## What it does

### Price estimator
Pulls local Marketplace listings, extracts mileage and asking price from listing text, and trains an XGBoost quantile regression model. Returns a low/mid/high price range grounded in your actual local market — not national book value averages. Mileage is the primary driver; year provides within-generation context.

### Conversation bot
Classifies incoming buyer messages by intent (serious buyer, lowballer, scammer), answers listing questions from a per-car knowledge base, negotiates within your set floor price, and automatically hands off to you the moment a buyer asks to meet or requests your location.

### Feedback loop
Every completed sale feeds back into the price model. Over time it calibrates specifically to your local market, your categories, and your selling patterns in ways no generic tool can replicate.

---

## Repo structure

```
lotsense/
├── data/
│   ├── collector.py        # FB session token + listing fetcher
│   ├── parser.py           # mileage + price text extraction
│   └── db.py               # local SQLite handler (30-day staleness)
├── pricing/
│   ├── estimator.py        # XGBoost quantile regression (low/mid/high)
│   └── features.py         # mileage + log(mileage) + year features
├── bot/
│   ├── classifier.py       # intent classifier (serious / lowball / scam)
│   ├── conversation.py     # response generation + negotiation logic
│   └── handoff.py          # location request detection
├── .env.example
├── requirements.txt
└── main.py
```

## Build phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Data collection + price model | done |
| 2 | Conversation bot + intent classifier | planned |
| 3 | Connect pricing context to negotiation logic | planned |

---

## Stack

| Component | Library |
|-----------|---------|
| Price regression | `XGBoost` |
| Intent classification | `BERT-base` fine-tuned |
| Conversation state | `LangChain` |
| Local data store | `SQLite` |

---

## How the two components connect

```
Price estimator output
        ↓
  Market value:  $8,400
  Your floor:    $7,800
  Listed at:     $9,000
        ↓
Conversation bot uses this context
  → Knows how much room exists to negotiate
  → Knows when an offer is insulting vs reasonable
  → Holds firm intelligently, not arbitrarily
  → Hands off to you when buyer asks to meet
```

---

## Data and usage note

This tool is built for personal use. Requests are scoped to your own listings and messages and kept at human-like rates. The intent is to automate your own normal Marketplace activity, not to bulk-collect other users' data.
