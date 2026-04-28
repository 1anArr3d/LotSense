"""
pricing/features.py — Build the XGBoost feature matrix from AuctionListings.

Features:
  [0] log_mileage    log(miles) — compresses scale, captures depreciation curve
  [1] mileage        raw miles
  [2] year           model year
  [3] has_accidents  1 if collision damage (front/rear/side/rollover), else 0

Target (y): retail_estimate in dollars — ACV from Copart/IAAI appraiser.
Listings without retail_estimate are excluded.
"""

import numpy as np

from data.parser import AuctionListing

FEATURE_NAMES = ["log_mileage", "mileage", "year"]


def build_features(listings: list[AuctionListing]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) ready for xgb.DMatrix. Drops listings with no retail_estimate."""
    rows = [l for l in listings if l.retail_estimate]
    X = np.array(
        [[np.log(l.mileage), l.mileage, l.year] for l in rows],
        dtype=np.float32,
    )
    y = np.array([l.retail_estimate / 100 for l in rows], dtype=np.float32)
    return X, y
