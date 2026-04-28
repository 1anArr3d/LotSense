"""
pricing/estimator.py — XGBoost price estimator, scoped to a single make/model.

Trains three quantile models (10th / 50th / 90th percentile) so predict()
returns a price range rather than a single number.

Confidence tiers based on comp count:
    HIGH   (≥30 comps) — XGBoost quantile regression
    LOW    (5–29 comps) — numpy percentiles, mileage-sorted, no ML
    NONE   (<5 comps)  — raises ValueError

Usage:
    est = Estimator.from_db("Toyota", "Camry")
    result = est.predict(year=2016, mileage=130_000)
    # {"low": 6800, "mid": 9200, "high": 11500, "n_comps": 34, "confidence": "high"}
"""

import numpy as np
import xgboost as xgb

from data.db import get_listings, open_db
from data.parser import AuctionListing
from pricing.features import FEATURE_NAMES, build_features

MIN_SAMPLES = 5
MIN_SAMPLES_ML = 30
MAX_MILEAGE = 400_000
_QUANTILES = [0.10, 0.50, 0.90]

_XGB_PARAMS_BASE = {
    "objective": "reg:quantileerror",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "min_child_weight": 3,
    "seed": 42,
    # log_mileage↓  mileage↓  year↑
    "monotone_constraints": "(-1,-1,1)",
}


class Estimator:

    def __init__(self, make: str, model: str) -> None:
        self.make = make
        self.model = model
        self.n_samples: int = 0
        self.confidence: str = "none"
        self._models: dict[float, xgb.Booster] = {}
        self._fallback_prices: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, listings: list[AuctionListing], target_year: int | None = None) -> None:
        comps = [l for l in listings if l.retail_estimate]
        n = len(comps)
        if n < MIN_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_SAMPLES} comps for {self.make} {self.model}, "
                f"got {n}. Collect more listings and try again."
            )

        self.n_samples = n
        prices = np.array([l.retail_estimate / 100 for l in comps], dtype=np.float32)

        # Weight comps by proximity to target year — farther years count less.
        # Decay of 0.3 per year: ±3 years = ~40% weight, ±5 years = ~22%.
        weights = None
        if target_year is not None:
            years = np.array([l.year for l in comps], dtype=np.float32)
            weights = np.exp(-0.3 * np.abs(years - target_year)).astype(np.float32)

        if n >= MIN_SAMPLES_ML:
            X, y = build_features(comps)
            dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES, weight=weights)
            self._models = {}
            for q in _QUANTILES:
                params = {**_XGB_PARAMS_BASE, "quantile_alpha": q}
                self._models[q] = xgb.train(params, dtrain, num_boost_round=300)
            self.confidence = "high"
        else:
            self._fallback_prices = np.sort(prices)
            self.confidence = "low"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, year: int, mileage: int, market_discount: float = 0.75) -> dict:
        if self.confidence == "none":
            raise RuntimeError("Model not trained. Call train() or use from_db().")
        if not (1 <= mileage <= MAX_MILEAGE):
            raise ValueError(f"Mileage must be between 1 and {MAX_MILEAGE:,}. Got {mileage:,}.")

        if self.confidence == "high":
            X = np.array([[np.log(mileage), mileage, year]], dtype=np.float32)
            dtest = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
            raw = {q: float(m.predict(dtest)[0]) for q, m in self._models.items()}
            low = min(raw[0.10], raw[0.50])
            high = max(raw[0.50], raw[0.90])
            mid = raw[0.50]
        else:
            p = self._fallback_prices
            low = float(np.percentile(p, 10))
            mid = float(np.percentile(p, 50))
            high = float(np.percentile(p, 90))

        return {
            "low": max(0, round(low * market_discount)),
            "mid": max(0, round(mid * market_discount)),
            "high": max(0, round(high * market_discount)),
            "acv_mid": round(mid),
            "n_comps": self.n_samples,
            "confidence": self.confidence,
        }

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_db(
        cls,
        make: str,
        model: str,
        target_year: int | None = None,
    ) -> "Estimator":
        with open_db() as conn:
            listings = get_listings(conn, make=make, model=model)

        est = cls(make, model)
        est.train(listings, target_year=target_year)
        return est
