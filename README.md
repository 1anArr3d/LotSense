# LotSense

Auction car evaluator. Two independent ML models — price and mileage — that together tell you what a car is worth and whether the odometer reading is believable before you bid.

---

## Models

### pricing/ — Street price estimator
XGBoost quantile regression trained on Copart and SalvageBid auction comps. Returns a low/mid/high street price range. Training target is ACV × 0.70 (retail estimate discounted to street market). Monotone constraints enforce correct depreciation direction.

| Feature | Role |
|---------|------|
| `year` | input |
| `mileage` | input |
| `log(mileage)` | engineered input |
| `ACV × 0.70` | training target |

Confidence tiers: **high** (≥30 comps, XGBoost) / **low** (5–29, percentiles) / **none** (<5, error)

### mileage/ — Odometer estimator
Fetches Texas inspection history from mytxcar.org by VIN. Fits XGBoost quantile regression on (date, mileage) pairs to project current odometer. Detects suspect readings when a stated mileage deviates significantly from the projected range.


---

## Data sources

| Source | Provides |
|--------|----------|
| Copart | ACV, mileage, year, make, model |
| SalvageBid | ACV, mileage, year, make, model, VINs (Texas) |
| mytxcar.org | Texas inspection history by VIN (odometer readings over time) |

---

## Repo structure

```
lotsense/
├── data/
│   ├── copart.py           # Copart fetcher (httpx)
│   ├── salvagebid.py       # SalvageBid fetcher + VIN capture
│   ├── parser.py           # AuctionListing dataclass + field normalization
│   └── db.py               # SQLite: listings (90-day TTL), odometer_history
├── mileage/
│   ├── inspection_scrape.py  # mytxcar.org scraper
│   ├── mileage_model.py      # XGBoost quantile training + model save/load
│   └── predict_vin.py        # VIN → odometer estimate
├── pricing/
│   ├── estimator.py          # XGBoost quantile regression (low/mid/high)
│   └── features.py           # log(mileage), monotone constraints
├── collect.py              # standalone full pipeline (collect + scrape)
├── .env.example
├── requirements.txt
└── main.py
```

---

## Build phases

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Data layer: Copart collector, parser, SQLite schema | done |
| 2 | Pricing model: XGBoost on ACV labels | done |
| 3 | SalvageBid collector + VIN capture | done |
| 4 | Mileage model: inspection scrape + XGBoost odometer estimator | done |
| 5 | FB sales feedback loop | planned |

---

## Stack

| Component | Library |
|-----------|---------|
| Price + mileage regression | `xgboost` |
| HTTP | `httpx` |
| Local data store | `SQLite` |

---

## Usage

```bash
# collect Copart auction comps
python main.py collect --make Toyota --model Camry --year 2018

# collect SalvageBid comps + capture VINs
python main.py salvage

# run full pipeline (collect + scrape inspections)
python collect.py

# estimate street price
python main.py price Toyota Camry 2018 97000

# estimate odometer from inspection history
python main.py mileage 1HGBH41JXMN109186

# validate a stated mileage against inspection history
python main.py mileage 1HGBH41JXMN109186 --stated 43000

# retrain mileage model
python -m mileage.mileage_model

# show DB summary
python main.py stats
```
