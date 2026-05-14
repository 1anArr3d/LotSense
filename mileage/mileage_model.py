"""
mileage/mileage_model.py — XGBoost odometer estimator trained on inspection history.

Trains three quantile models (p10/p50/p90).

Usage:
    python -m mileage.mileage_model
    python -m mileage.mileage_model --txt path/to/inspection_results.txt
"""
from __future__ import annotations

import argparse
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

_MODEL_DIR = Path(__file__).parent
_QUANTILES = [0.10, 0.50, 0.90]
_XGB_PARAMS = {
    "objective":        "reg:quantileerror",
    "max_depth":        4,
    "eta":              0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "seed":             42,
}

_OUTLIER_THRESHOLD  = 45_000  # |stated - projected| above this → drop from training
_ROLLBACK_THRESHOLD = 40_000  # last_inspection > stated + this → suspected fraud
_DROP_THRESHOLD     = 1_000   # mileage dips > this in history → detected rollback

FEATURE_NAMES = [
    "last_inspection_mileage",
    "days_since_inspection",
    "lifetime_annual_rate",
    "recent_annual_rate",
    "rate_delta",
    "year",
    "projected_mileage",
    "age_at_last_inspection",
    "n_records",
    "inspection_span_days",
    "rate_stability",
    "rate_std",          # std dev of year-over-year interval rates
    "rate_trend",        # slope of interval rates over time (+ = accelerating)
    "recent_vs_early",   # avg of last 2 intervals / avg of first 2 (ratio)
    "max_interval_rate", # highest single-interval annual rate
    "min_interval_rate", # lowest single-interval annual rate
]


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _projection(last_mi: float, days: float, recent_rate: float) -> float:
    return last_mi + (days / 365.25) * recent_rate


def _detect_rollback(readings: list[tuple[date, int]]) -> bool:
    peak = 0
    for _, mi in readings:
        if mi < peak - _DROP_THRESHOLD:
            return True
        if mi > peak:
            peak = mi
    return False


def _interval_rates(readings: list[tuple[date, int]]) -> list[float]:
    rates = []
    for i in range(1, len(readings)):
        d1, m1 = readings[i - 1]
        d2, m2 = readings[i]
        yrs = (d2 - d1).days / 365.25
        if yrs > 0.05:
            rates.append((m2 - m1) / yrs)
    return rates


def features_from_readings(
    readings_raw: list[tuple[date, int]],
    model_year: int,
    today: date,
) -> dict | None:
    """
    Convert raw (date, mileage) pairs into the full feature dict.
    Returns None if the data is invalid (rollback detected, too few readings, etc).
    Deduplication and sorting are handled internally.
    """
    by_date: dict[date, int] = {}
    for d, mi in readings_raw:
        by_date[d] = max(by_date.get(d, 0), mi)
    readings = sorted(by_date.items())

    if len(readings) < 2:
        return None
    if _detect_rollback(readings):
        return None

    first_date, first_mi = readings[0]
    last_date,  last_mi  = readings[-1]

    total_yrs = (last_date - first_date).days / 365.25
    if total_yrs <= 0.1:
        return None
    lifetime_rate = round((last_mi - first_mi) / total_yrs)

    prev_date, prev_mi = readings[-2]
    recent_yrs  = (last_date - prev_date).days / 365.25
    recent_rate = round((last_mi - prev_mi) / recent_yrs) if recent_yrs > 0.05 else lifetime_rate
    rate_delta  = recent_rate - lifetime_rate

    days_since = (today - last_date).days
    span_days  = (last_date - first_date).days
    insp_year  = last_date.year

    projected      = _projection(last_mi, days_since, recent_rate)
    age_at_insp    = insp_year - model_year
    rate_stability = abs(rate_delta) / lifetime_rate if lifetime_rate > 0 else 0.0

    rates = _interval_rates(readings)

    rate_std   = float(np.std(rates)) if len(rates) >= 2 else 0.0
    rate_trend = float(np.polyfit(np.arange(len(rates), dtype=float), rates, 1)[0]) if len(rates) >= 2 else 0.0

    if len(rates) >= 4:
        early = np.mean(rates[:2])
        rv    = float(np.mean(rates[-2:]) / early) if early > 0 else 1.0
    elif len(rates) >= 2:
        rv = float(rates[-1] / rates[0]) if rates[0] > 0 else 1.0
    else:
        rv = 1.0

    return {
        "last_inspection_mileage": last_mi,
        "days_since_inspection":   days_since,
        "lifetime_annual_rate":    lifetime_rate,
        "recent_annual_rate":      recent_rate,
        "rate_delta":              rate_delta,
        "year":                    model_year,
        "projected_mileage":       projected,
        "age_at_last_inspection":  age_at_insp,
        "n_records":               len(readings),
        "inspection_span_days":    span_days,
        "rate_stability":          rate_stability,
        "rate_std":                rate_std,
        "rate_trend":              rate_trend,
        "recent_vs_early":         rv,
        "max_interval_rate":       float(max(rates)) if rates else float(lifetime_rate),
        "min_interval_rate":       float(min(rates)) if rates else float(lifetime_rate),
    }


# ── Inspection file parser ─────────────────────────────────────────────────────

_HEADER_RE  = re.compile(r"--- (\S+) \| (.+?) \| stated: ([\d,]+) mi ---")
_READING_RE = re.compile(r"\s+\S{17}\s+(\d{1,2}/\d{1,2}/\d{4})\s+([\d,]+)\s+mi")


def _parse_date(s: str) -> date | None:
    try:
        return datetime.strptime(s.strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def parse_inspection_txt(
    path: str,
    min_stated: int = 1_000,
    today: date | None = None,
) -> list[tuple[dict, int]]:
    """
    Parse inspection_results.txt into (feature_dict, stated_mileage) pairs.
    Drops: no records, fewer than 2 readings, detected rollbacks, bad stated mileage.
    """
    if today is None:
        today = date.today()

    blocks: list[dict] = []
    current: dict | None = None

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("---"):
                if current:
                    blocks.append(current)
                m = _HEADER_RE.match(line)
                if m:
                    parts = m.group(2).split(" ", 2)
                    current = {
                        "year":           int(parts[0]) if parts[0].isdigit() else None,
                        "stated_mileage": int(m.group(3).replace(",", "")),
                        "readings":       [],
                        "no_records":     False,
                    }
                else:
                    current = None
            elif current is None:
                continue
            elif "NO RECORDS FOUND" in line:
                current["no_records"] = True
            else:
                m = _READING_RE.match(line)
                if m:
                    d = _parse_date(m.group(1))
                    mi = int(m.group(2).replace(",", ""))
                    if d:
                        current["readings"].append((d, mi))

    if current:
        blocks.append(current)

    rows: list[tuple[dict, int]] = []
    n_no_records = n_sparse = n_rollback = n_bad_stated = 0

    for block in blocks:
        if block["no_records"]:
            n_no_records += 1
            continue
        if len(block["readings"]) < 2 or block["year"] is None:
            n_sparse += 1
            continue
        stated = block["stated_mileage"]
        if stated < min_stated:
            n_bad_stated += 1
            continue
        feats = features_from_readings(block["readings"], block["year"], today)
        if feats is None:
            n_rollback += 1
            continue
        rows.append((feats, stated))

    print(f"[parser] {len(blocks)} blocks  →  dropped: no_records={n_no_records} "
          f"sparse={n_sparse} rollback={n_rollback} bad_stated={n_bad_stated}  "
          f"→  {len(rows)} clean rows")
    return rows


# ── DB parser ─────────────────────────────────────────────────────────────────

def parse_inspection_db(
    min_stated: int = 1_000,
    today: date | None = None,
) -> list[tuple[dict, int]]:
    """
    Read inspection history from SQLite (listings + odometer_history).
    Returns same (feature_dict, stated_mileage) pairs as parse_inspection_txt.
    """
    from data.db import open_db

    if today is None:
        today = date.today()

    with open_db() as conn:
        vin_rows = conn.execute("""
            SELECT vin, year, mileage AS stated_mileage
            FROM listings
            WHERE vin IS NOT NULL AND source = 'salvagebid'
            GROUP BY vin
        """).fetchall()

        reading_rows = conn.execute("""
            SELECT vin, inspection_date, mileage
            FROM odometer_history
            WHERE vin IS NOT NULL
            ORDER BY vin, inspection_date
        """).fetchall()

    vin_meta: dict[str, dict] = {
        r["vin"]: {"year": r["year"], "stated": r["stated_mileage"]}
        for r in vin_rows
    }

    readings_by_vin: dict[str, list[tuple[date, int]]] = {}
    for r in reading_rows:
        d = _parse_date(r["inspection_date"])
        if d:
            readings_by_vin.setdefault(r["vin"], []).append((d, r["mileage"]))

    rows: list[tuple[dict, int]] = []
    n_sparse = n_rollback = n_bad_stated = n_no_meta = 0

    for vin, readings in readings_by_vin.items():
        meta = vin_meta.get(vin)
        if not meta or meta["year"] is None:
            n_no_meta += 1
            continue
        if len(readings) < 2:
            n_sparse += 1
            continue
        stated = meta["stated"]
        if stated < min_stated:
            n_bad_stated += 1
            continue
        feats = features_from_readings(readings, meta["year"], today)
        if feats is None:
            n_rollback += 1
            continue
        rows.append((feats, stated))

    print(f"[parser] {len(readings_by_vin)} VINs  →  dropped: no_meta={n_no_meta} "
          f"sparse={n_sparse} rollback={n_rollback} bad_stated={n_bad_stated}  "
          f"→  {len(rows)} clean rows")
    return rows


# ── Training ───────────────────────────────────────────────────────────────────

def _baseline_mae(X: np.ndarray, y: np.ndarray) -> float:
    li = FEATURE_NAMES.index("last_inspection_mileage")
    di = FEATURE_NAMES.index("days_since_inspection")
    ri = FEATURE_NAMES.index("recent_annual_rate")
    return float(np.mean(np.abs(X[:, li] + (X[:, di] / 365.25) * X[:, ri] - y)))


def train(txt_path: str | None = None) -> None:
    today = date.today()
    rows  = parse_inspection_txt(txt_path, today=today) if txt_path else parse_inspection_db(today=today)

    X = np.array([[f[k] for k in FEATURE_NAMES] for f, _ in rows], dtype=np.float32)
    y = np.array([s for _, s in rows], dtype=np.float32)

    proj_idx = FEATURE_NAMES.index("projected_mileage")
    clean    = np.abs(y - X[:, proj_idx]) <= _OUTLIER_THRESHOLD
    n_drop   = int((~clean).sum())
    X, y     = X[clean], y[clean]
    print(f"[mileage_model] dropped {n_drop} outliers  →  {len(y)} training rows")

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(len(y))
    split = int(len(y) * 0.8)
    tr, te = idx[:split], idx[split:]

    dtrain = xgb.DMatrix(X[tr], label=y[tr], feature_names=FEATURE_NAMES)
    dtest  = xgb.DMatrix(X[te], label=y[te], feature_names=FEATURE_NAMES)

    baseline = _baseline_mae(X[te], y[te])
    print(f"[mileage_model] baseline MAE (linear):    {baseline:>9,.0f} mi")

    models: dict[float, xgb.Booster] = {}
    for q in _QUANTILES:
        models[q] = xgb.train(
            {**_XGB_PARAMS, "quantile_alpha": q},
            dtrain,
            num_boost_round=400,
            evals=[(dtest, "test")],
            verbose_eval=False,
        )

    mae = float(np.mean(np.abs(models[0.50].predict(dtest) - y[te])))
    print(f"[mileage_model] XGBoost p50 MAE:          {mae:>9,.0f} mi")
    print(f"[mileage_model] improvement over baseline: {baseline - mae:>+9,.0f} mi")

    for q, model in models.items():
        fname = f"mileage_q{int(q * 100):02d}.json"
        model.save_model(_MODEL_DIR / fname)
        print(f"[mileage_model] saved {fname}")


# ── Inference ──────────────────────────────────────────────────────────────────

def load_models() -> dict[float, xgb.Booster]:
    models: dict[float, xgb.Booster] = {}
    for q in _QUANTILES:
        fname = f"mileage_q{int(q * 100):02d}.json"
        path  = _MODEL_DIR / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\nTrain first: python -m mileage.mileage_model"
            )
        b = xgb.Booster()
        b.load_model(path)
        models[q] = b
    return models


def predict(
    last_inspection_mileage: int,
    days_since_inspection:   int,
    lifetime_annual_rate:    int,
    year:                    int,
    recent_annual_rate:      int   | None = None,
    rate_delta:              int   | None = None,
    n_records:               int          = 4,
    inspection_span_days:    int          = 1095,
    stated_mileage:          int   | None = None,
    rate_std:                float        = 0.0,
    rate_trend:              float        = 0.0,
    recent_vs_early:         float        = 1.0,
    max_interval_rate:       float | None = None,
    min_interval_rate:       float | None = None,
) -> dict:
    """Return {"low", "mid", "high", "suspected_rollback"}."""
    models   = load_models()
    recent   = recent_annual_rate if recent_annual_rate is not None else lifetime_annual_rate
    delta    = rate_delta         if rate_delta         is not None else 0
    max_rate = max_interval_rate  if max_interval_rate  is not None else float(recent)
    min_rate = min_interval_rate  if min_interval_rate  is not None else float(recent)

    projected      = _projection(last_inspection_mileage, days_since_inspection, recent)
    insp_year      = date.today().year - round(days_since_inspection / 365.25)
    age_at_insp    = insp_year - year
    rate_stability = abs(delta) / lifetime_annual_rate if lifetime_annual_rate > 0 else 0.0

    X = np.array([[
        last_inspection_mileage, days_since_inspection,
        lifetime_annual_rate, recent, delta, year,
        projected, age_at_insp, n_records, inspection_span_days, rate_stability,
        rate_std, rate_trend, recent_vs_early, max_rate, min_rate,
    ]], dtype=np.float32)
    d = xgb.DMatrix(X, feature_names=FEATURE_NAMES)

    suspected_rollback = (
        stated_mileage is not None
        and last_inspection_mileage > stated_mileage + _ROLLBACK_THRESHOLD
    )

    floor = last_inspection_mileage
    low   = max(int(models[0.10].predict(d)[0]), floor)
    mid   = max(int(models[0.50].predict(d)[0]), floor)
    high  = max(int(models[0.90].predict(d)[0]), floor)

    # Out-of-distribution fallback: all quantiles clamped → use linear projection ±15k
    if low == mid == high == floor:
        proj = int(projected)
        low  = max(floor, proj - 15_000)
        mid  = max(floor, proj)
        high = proj + 15_000

    return {
        "low":                low,
        "mid":                mid,
        "high":               high,
        "suspected_rollback": suspected_rollback,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default=None, help="Train from text file instead of DB")
    train(ap.parse_args().txt)
