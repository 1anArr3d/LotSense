# LotSense

Local car price estimator + automated FB Marketplace conversation bot.

---

## What it does

### Price estimator
Trains an XGBoost quantile regression model on Copart auction data. Returns a low/mid/high street price range. The training target is Copart's ACV (retail estimate) discounted 25% to reflect street market prices. Monotone constraints enforce correct depreciation direction. 400+ comps per vehicle in ~10 pages is sufficient for reliable results.

### Conversation bot
Classifies incoming buyer messages by intent (serious buyer, lowballer, scammer), answers listing questions from a per-car knowledge base, negotiates within your set floor price, and hands off to you when a buyer asks to meet or requests your location.

---

## Model design

| Feature | Role |
|---------|------|
| `year` | input |
| `mileage` | input |
| `log(mileage)` | engineered input |
| `ACV × 0.75` | **training target** (street price proxy) |

No vision. No hammer prices. No BERT. No LangChain. Copart supplies ACV directly as the label. Monotone constraints on mileage and year enforce that more miles and older age always lower predicted value.

---

## Data sources

| Source | Provides |
|--------|----------|
| Copart | ACV (retail estimate), mileage, year, make, model |
| Your FB sales | Local calibration (manual entry, feedback loop — Phase 5) |

IAAI is a potential future addition but not needed — Copart alone provides sufficient training volume.

---

## Repo structure

```
lotsense/
├── data/
│   ├── copart.py               # Copart fetcher (httpx, no auth required)
│   ├── parser.py               # AuctionListing dataclass + field normalization
│   └── db.py                   # SQLite: listings, price_estimates
├── pricing/
│   ├── estimator.py            # XGBoost quantile regression (low/mid/high)
│   └── features.py             # log(mileage), monotone constraints
├── bot/
│   ├── adapter.py              # Playwright + mock adapters
│   ├── dispatcher.py           # message routing + poll loop
│   ├── classifier.py           # intent classifier (serious / lowball / scam)
│   ├── conversation.py         # response generation + negotiation logic
│   ├── handoff.py              # location request detection
│   └── listings_example.json   # listing config template
├── fb_scraper/                 # preserved for reuse, not active in LotSense
│   ├── collector.py
│   └── parser.py
├── data/training/
│   └── intent_seed.jsonl       # 120-example seed dataset
├── tests/
│   └── scripts/
│       └── negotiation.jsonl   # integration test script
├── .env.example
├── requirements.txt
└── main.py
```

---

## Build phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Data layer: Copart collector, parser, SQLite schema | done |
| 2 | Pricing model: XGBoost on ACV labels, monotone constraints | done |
| 3 | Conversation bot + intent classifier | in progress (90%) |
| 4 | Connect pricing context to negotiation logic | in progress (90%) |
| 5 | FB sales feedback loop | planned |

---

## Remaining gaps (Phase 3)

1. **Classifier training** — keyword fallback confidence (0.50–0.70) falls below auto-reply thresholds in the dispatcher. Needs ~200+ examples per class; currently 120. Fix: expand seed dataset or raise keyword confidence for clear-cut matches.
2. **Integration test** — `tests/scripts/negotiation.jsonl` uses `listing_id: "TEST-001"` but no `bot/listings.json` exists yet. Create from `listings_example.json` first.

---

## Stack

| Component | Library |
|-----------|---------|
| Price regression | `xgboost` |
| HTTP | `httpx` |
| Local data store | `SQLite` |
| Browser automation (bot) | `playwright` |

---

## How the two components connect

```
Price estimator output
        ↓
  Street value:  $8,400
  Your floor:    $7,800
  Listed at:     $9,000
        ↓
Conversation bot uses this context
  → Knows how much room exists to negotiate
  → Holds firm intelligently, not arbitrarily
  → Hands off to you when buyer asks to meet
```

---

## Usage

```bash
# collect Copart data
python main.py collect --make Toyota --model Camry --min-year 2010 --max-year 2016 --pages 10

# show DB summary
python main.py stats

# run the bot (real)
python main.py bot --config bot/listings.json

# run the bot (mock replay)
python main.py bot --mock --script tests/scripts/negotiation.jsonl
```
