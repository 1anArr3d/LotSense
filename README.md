# LotSense
 
Local car price estimator + automated Marketplace conversation bot. Part of the [SwiftLot](https://github.com/yourusername/swiftlot) ecosystem.
Collects Facebook Marketplace listings in your area, trains a vision + mileage model to estimate fair market value, and handles buyer conversations automatically until they ask for your location.
 
---
 
## What it does
 
### Price estimator
Pulls local Marketplace listings, extracts mileage and asking price from listing text, and trains an XGBoost regression model cross-referenced with a fine-tuned EfficientNet vision model that grades condition from photos. The result is a price range grounded in your actual local market — not national book value averages.
 
### Conversation bot
Classifies incoming buyer messages by intent (serious buyer, lowballer, scammer), answers listing questions from a per-car knowledge base, negotiates within your set floor price, and automatically hands off to you the moment a buyer asks to meet or requests your location.
 
### Photo ranker
Scores your listing photos by lighting quality, angle, and background clutter. Picks the best lead image automatically and flags weak shots worth re-taking.
 
### Feedback loop
Every completed sale feeds back into the price model. Over time it calibrates specifically to your local market, your categories, and your selling patterns in ways no generic tool can replicate.
 
---
 
## Repo structure
 
```
lotsense/
├── data/
│   ├── collector.py        # FB session token + listing fetcher
│   ├── parser.py           # mileage + price text extraction
│   └── db.py               # local SQLite handler
├── vision/
│   ├── condition.py        # EfficientNet fine-tune for condition grading
│   └── photo_ranker.py     # lead image picker
├── pricing/
│   ├── estimator.py        # XGBoost price regression model
│   └── features.py         # feature engineering
├── bot/
│   ├── classifier.py       # intent classifier (serious / lowball / scam)
│   ├── conversation.py     # response generation + negotiation logic
│   └── handoff.py          # location request detection
├── utils/
│   └── shap_explain.py     # model explainability via SHAP
├── .env.example
├── requirements.txt
└── main.py
```
## Build phases
 
| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Data collection + baseline price model | in progress |
| 2 | Vision condition grader (EfficientNet fine-tune) | planned |
| 3 | Conversation bot + intent classifier | planned |
| 4 | Connect pricing context to negotiation logic | planned |
 
---
 
## Stack
 
| Component | Library |
|-----------|---------|
| Price regression | `scikit-learn`, `XGBoost` |
| Condition grading | `EfficientNet` via `torchvision` |
| Intent classification | `BERT-base` fine-tuned |
| Conversation state | `LangChain` |
| Model explainability | `SHAP` |
| Local data store | `SQLite` + `pandas` |
| API requests | `httpx` |
 
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
 
Every conversation also becomes training data. If someone offered $7,000 and you sold at $8,200, that real outcome feeds back into the price model as a local data point — more accurate than any national pricing database.
 
---
 
## Data and usage note
 
This tool is built for personal use. Requests are scoped to your own listings and messages and kept at human-like rates. The intent is to automate your own normal Marketplace activity, not to bulk-collect other users' data.
