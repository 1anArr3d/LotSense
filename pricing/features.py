"""
pricing/features.py — Build the XGBoost feature matrix from ParsedListings.

Features per listing:
  [0] mileage  — integer miles (primary price driver within a model)
  [1] year     — model year (accounts for facelift/gen differences within a nameplate)

Price (y) is in dollars (not cents) for human-readable residuals.
"""

import numpy as np

from data.parser import ParsedListing

FEATURE_NAMES = ["mileage", "year"]


def build_features(listings: list[ParsedListing]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) ready for xgb.DMatrix. X shape: (n, 2), y shape: (n,)."""
    X = np.array(
        [[l.mileage, l.year] for l in listings],
        dtype=np.float32,
    )
    y = np.array(
        [l.price_cents / 100 for l in listings],
        dtype=np.float32,
    )
    return X, y
