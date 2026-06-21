"""
ParkSight BLR -- AI-driven Parking Congestion Intelligence
Data pipeline: transforms raw Bengaluru Traffic Police parking-violation
records into analytics artifacts consumed by the dashboard.

Pipeline stages
---------------
1.  Load + clean the raw CSV (parse JSON violation arrays, timestamps, geo).
2.  Keep parking-relevant violations only and convert UTC -> IST.
3.  Grid-bin every violation to ~165 m cells -> the congestion heatmap.
4.  Score every cell with CIS/CIRS (0-100), using density, severity,
    peak-hour concentration, junction proximity and recurrence.
5.  Layer 4: estimate traffic-flow obstruction risk from violation type,
    peak-hour concentration and junction spillback risk.
6.  Merge hot cells into named enforcement zones with DBSCAN and rank them.
7.  Layer 1: train a supervised zone×day×hour ML forecast and backtest it
    so the dashboard predicts patrol priority instead of only mapping history.
8.  Layer 5: expose ParkSched, an MLFQ-inspired patrol scheduler that turns
    ML need scores into deployable patrol slots with aging fairness.
9.  Layer 6: train an IsolationForest early-warning detector to flag zone-hour
    surges that are abnormal relative to each zone's own baseline.
10. Build temporal/categorical analytics and emit compact JSON into web/data/.

Run:  python3 pipeline/build_data.py
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORKSPACE = os.path.dirname(ROOT)
RAW_CSV = os.path.join(WORKSPACE, "jan to may police violation_anonymized791b166.csv")
OUT_DIR = os.path.join(ROOT, "web", "data")

# Bengaluru bounding box (filters out bad / null coordinates).
BLR_BBOX = dict(lat_min=12.70, lat_max=13.25, lon_min=77.35, lon_max=77.85)

# ~165 m grid (0.0015 deg lat; longitude distortion is acceptable for this city-scale prototype).
GRID = 0.0015

# Rush-hour windows (IST) used to weight congestion impact.
PEAK_HOURS = set(range(8, 12)) | set(range(16, 22))

# Severity weight per violation type -- how badly it chokes a carriageway / intersection.
SEVERITY = {
    "PARKING NEAR ROAD CROSSING": 1.00,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 1.00,
    "PARKING IN A MAIN ROAD": 0.95,
    "DOUBLE PARKING": 0.90,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.80,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.75,
    "PARKING OTHER THAN BUS STOP": 0.70,
    "PARKING ON FOOTPATH": 0.65,
    "WRONG PARKING": 0.60,
    "NO PARKING": 0.55,
}

# Layer 4: transparent traffic-flow obstruction weights.
# This is not measured speed loss. It is a calibration-ready proxy for lane blockage,
# intersection spillback and flow obstruction until speed/CCTV queue data is connected.
FLOW_OBSTRUCTION = {
    "PARKING NEAR ROAD CROSSING": 1.00,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 1.00,
    "PARKING IN A MAIN ROAD": 0.96,
    "DOUBLE PARKING": 0.94,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.86,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.82,
    "PARKING OTHER THAN BUS STOP": 0.74,
    "WRONG PARKING": 0.63,
    "NO PARKING": 0.60,
    "PARKING ON FOOTPATH": 0.52,
}

# Validation status confidence. Unknown/null rows are retained but discounted.
CONFIDENCE = {
    "approved": 1.0,
    "processing": 0.7,
    "created1": 0.6,
    "unknown": 0.6,
    "": 0.6,
    "rejected": 0.0,
    "duplicate": 0.0,
}

# Any violation type that contains "PARK" is parking-relevant.
PARK_RE = re.compile(r"PARK", re.I)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_violations(raw: str) -> list[str]:
    """Parse the stringified JSON array of violation labels."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        vals = json.loads(raw)
        return [str(v).strip().upper() for v in vals] if isinstance(vals, list) else []
    except Exception:
        # Fallback for malformed rows.
        return [t.strip().upper() for t in re.findall(r'"([^"]+)"', raw)]


def short_name(addr: str) -> str:
    """Derive a readable locality label from a long address string."""
    if not isinstance(addr, str):
        return "Unknown"
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    parts = [p for p in parts if not re.search(r"(India|Karnataka|Pin-|\d{6})", p)]
    return ", ".join(parts[:2]) if parts else "Unknown"


def clean_junction(j: str) -> str:
    if not isinstance(j, str) or j.strip() in ("", "No Junction", "NULL"):
        return ""
    # "BTP051 - Safina Plaza Junction" -> "Safina Plaza Junction"
    return re.sub(r"^BTP\d+\s*-\s*", "", j).strip()


def row_max_weight(vs: list[str], table: dict[str, float], default: float) -> float:
    return max((table.get(v, default) for v in vs if PARK_RE.search(v)), default=default)


def assign_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable grid ids used by both cells and zone-row joins."""
    if "gy" not in df.columns or "gx" not in df.columns:
        df["gy"] = np.round(df.latitude / GRID).astype(int)
        df["gx"] = np.round(df.longitude / GRID).astype(int)
    return df


def impact_band(score: float) -> str:
    if score >= 82:
        return "Severe"
    if score >= 65:
        return "High"
    if score >= 45:
        return "Moderate"
    return "Low"


# --------------------------------------------------------------------------
# 1-2. Load + clean
# --------------------------------------------------------------------------
def load() -> pd.DataFrame:
    print("Loading raw CSV ...")
    required = [
        "id", "latitude", "longitude", "location", "vehicle_type",
        "violation_type", "created_datetime", "closed_datetime",
        "police_station", "junction_name",
    ]
    optional = ["validation_status", "data_sent_to_scita"]
    available = set(pd.read_csv(RAW_CSV, nrows=0).columns)
    cols = [c for c in required + optional if c in available]
    df = pd.read_csv(RAW_CSV, usecols=cols, low_memory=False)
    print(f"  raw rows: {len(df):,}")

    # Geo cleaning.
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    b = BLR_BBOX
    df = df[
        df.latitude.between(b["lat_min"], b["lat_max"])
        & df.longitude.between(b["lon_min"], b["lon_max"])
    ]

    # Timestamps -> IST.
    df["created"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["created"])
    df["created"] = df["created"] + timedelta(hours=5, minutes=30)
    df["hour"] = df["created"].dt.hour
    df["weekday"] = df["created"].dt.weekday  # Mon=0
    df["date"] = df["created"].dt.date

    # Validation-aware confidence. Rejected/duplicate rows stay in raw auditability,
    # but they receive 0 confidence and therefore do not drive scores.
    if "validation_status" in df.columns:
        df["validation_status"] = df["validation_status"].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        df["validation_status"] = "unknown"
    df["confidence"] = df["validation_status"].map(CONFIDENCE).fillna(0.6).astype(float)

    # Violation labels.
    df["violations"] = df["violation_type"].apply(parse_violations)
    df = df[df["violations"].map(len) > 0]

    # Parking-relevant subset: at least one violation contains "PARK".
    df["is_parking"] = df["violations"].apply(lambda vs: any(PARK_RE.search(v) for v in vs))
    df = df[df["is_parking"]].copy()

    # Per-row factors.
    df["severity"] = df["violations"].apply(lambda vs: row_max_weight(vs, SEVERITY, 0.55))
    df["obstruction"] = df["violations"].apply(lambda vs: row_max_weight(vs, FLOW_OBSTRUCTION, 0.60))
    df["junction_clean"] = df["junction_name"].apply(clean_junction)
    df["locality"] = df["location"].apply(short_name)
    assign_grid(df)

    print(f"  parking-violation rows (clean): {len(df):,}")
    print(f"  validation-discounted scoring rows: {df['confidence'].sum():,.0f} equivalent rows")
    return df


# --------------------------------------------------------------------------
# 3-4. Grid heatmap + Congestion Impact Risk Score
# --------------------------------------------------------------------------
def build_cells(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_grid(df.copy())

    # Precompute boolean factors once. This is much faster than groupby lambdas.
    df["is_peak"] = df["hour"].isin(PEAK_HOURS).astype(float)
    df["has_junction"] = (df["junction_clean"].str.len() > 0).astype(float)

    grp = df.groupby(["gy", "gx"], sort=False)
    cells = grp.agg(
        count=("id", "size"),
        weighted_count=("confidence", "sum"),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
        severity=("severity", "mean"),
        obstruction=("obstruction", "mean"),
        peak_share=("is_peak", "mean"),
        ndays=("date", "nunique"),
        njunc=("has_junction", "mean"),
    ).reset_index()

    span = int(df["date"].nunique()) or 1
    # Chronic recurrence: fraction of calendar days in the data window this cell saw a violation.
    cells["recurrence"] = (cells["ndays"] / span).clip(0, 1)

    # --- Congestion Impact Risk Score (CIRS/CIS, 0-100) -------------------
    # density: log-scaled confidence-weighted violation volume (saturating)
    dens = np.log1p(cells["weighted_count"].clip(lower=0)) / np.log1p(cells["weighted_count"].max())
    sev = cells["severity"]                       # road-criticality of violations
    peak = 0.5 + 0.5 * cells["peak_share"]        # rush-hour concentration multiplier
    junc = 1.0 + 0.25 * cells["njunc"]            # intersection proximity boost
    rec = 0.6 + 0.4 * cells["recurrence"]         # chronic vs one-off
    raw = dens * sev * peak * junc * rec
    cells["cis"] = (100 * raw / raw.max()).round(1)
    cells["density_n"] = (100 * dens).round(1)

    # --- Layer 4: traffic-flow obstruction / calibration proxy ------------
    # Until speed, queue length or lane occupancy feeds are attached, this is a transparent
    # obstruction-risk proxy from the fields available in the violation dump.
    obstruction_raw = (
        0.40 * cells["obstruction"] +
        0.25 * cells["severity"] +
        0.20 * cells["peak_share"] +
        0.15 * cells["njunc"].clip(0, 1)
    )
    cells["obstruction_risk"] = (100 * obstruction_raw.clip(0, 1)).round(1)
    cells["flow_impact_score"] = (0.60 * cells["cis"] + 0.40 * cells["obstruction_risk"]).round(1)
    return cells


# --------------------------------------------------------------------------
# 5. Enforcement zones (DBSCAN over hot cells)
# --------------------------------------------------------------------------
def build_zones(df: pd.DataFrame, cells: pd.DataFrame, top_cells: int = 600) -> tuple[list[dict], dict, dict, list[dict]]:
    hot = cells.sort_values("flow_impact_score", ascending=False).head(top_cells).copy()
    coords = np.radians(hot[["lat", "lon"]].to_numpy())
    # eps ~ 300 m (in radians on unit sphere; earth radius 6371 km)
    eps = 0.30 / 6371.0
    db = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit(coords)
    hot["zone"] = db.labels_

    # Fast row-zone join: avoid scanning all rows once per zone.
    df = assign_grid(df.copy())
    df_hot = df.merge(hot[["gy", "gx", "zone"]], on=["gy", "gx"], how="inner")

    dates = sorted(df["date"].unique())
    split_idx = max(1, int(len(dates) * 0.70))
    cutoff = dates[split_idx - 1]
    train_days = max(1, split_idx)
    valid_days = max(1, len(dates) - split_idx)

    zones: list[dict] = []
    for zid, g in hot.groupby("zone"):
        rows = df_hot[df_hot["zone"] == zid]
        if len(rows) == 0:
            continue
        total = int(len(rows))
        weighted_total = float(rows["confidence"].sum())

        # Weighted centroid (by cell count).
        lat = float(np.average(g.lat, weights=g["count"]))
        lon = float(np.average(g.lon, weights=g["count"]))

        # Zone scores: volume-weighted mean of member cells, lightly boosted by spatial extent.
        zcis = float(np.average(g.cis, weights=g["count"]))
        zcis = min(100.0, zcis * (1 + 0.04 * (len(g) - 1)))
        zflow = float(np.average(g.flow_impact_score, weights=g["count"]))
        zflow = min(100.0, zflow * (1 + 0.03 * (len(g) - 1)))
        zobstruction = float(np.average(g.obstruction_risk, weights=g["count"]))

        # Dominant descriptors.
        vio = Counter(v for vs in rows["violations"] for v in vs if PARK_RE.search(v))
        veh = Counter(rows["vehicle_type"].dropna())
        loc = Counter(rows["locality"])
        juncs = Counter(j for j in rows["junction_clean"] if j)
        station = Counter(rows["police_station"].dropna())

        # Hour-of-day profile (24) -> powers the predictive deployment slider.
        hour_counts = rows["hour"].value_counts()
        hourly = [int(hour_counts.get(h, 0)) for h in range(24)]
        top_hours = sorted(sorted(range(24), key=lambda h: hourly[h], reverse=True)[:3])
        peak_share = float(np.mean([h in PEAK_HOURS for h in rows["hour"]]))

        train_rows = rows[rows["date"] <= cutoff]
        valid_rows = rows[rows["date"] > cutoff]
        train_count = int(len(train_rows))
        valid_count = int(len(valid_rows))
        predicted_valid = (train_count / train_days) * valid_days

        name = ""
        if juncs:
            name = juncs.most_common(1)[0][0]
        if not name:
            name = loc.most_common(1)[0][0] if loc else "Unnamed zone"

        # Layer 4 approximate obstruction outputs.
        # Name is intentionally risk/estimated, not measured delay.
        lane_blockage_risk_pct = round(5 + 45 * (zflow / 100), 1)
        band = impact_band(zflow)
        evidence = []
        if peak_share >= 0.40:
            evidence.append("rush-hour concentration")
        if juncs:
            evidence.append("junction spillback risk")
        if vio and vio.most_common(1)[0][0] in {
            "PARKING NEAR ROAD CROSSING",
            "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS",
            "PARKING IN A MAIN ROAD",
            "DOUBLE PARKING",
        }:
            evidence.append("carriageway-blocking offence mix")
        if not evidence:
            evidence.append("repeat illegal-parking pressure")

        zones.append({
            "id": int(zid),
            "name": name,
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "cis": round(zcis, 1),
            "flow_impact_score": round(zflow, 1),
            "obstruction_risk": round(zobstruction, 1),
            "flow_impact_band": band,
            "lane_blockage_risk_pct": lane_blockage_risk_pct,
            "impact_evidence": evidence[:3],
            "violations": total,
            "weighted_violations": round(weighted_total, 1),
            "cells": int(len(g)),
            "peak_hours": top_hours,
            "hourly": hourly,
            "peak_share": round(peak_share, 3),
            "recurrence": round(float(g["recurrence"].max()), 3),
            "top_violation": vio.most_common(1)[0][0] if vio else "",
            "violation_mix": [{"label": k, "n": int(v)} for k, v in vio.most_common(4)],
            "top_vehicle": veh.most_common(1)[0][0] if veh else "",
            "station": station.most_common(1)[0][0] if station else "",
            "junctions": [k for k, _ in juncs.most_common(3)],
            "locality": loc.most_common(1)[0][0] if loc else "",
            "train_violations": train_count,
            "validation_violations": valid_count,
            "predicted_validation_violations": round(predicted_valid, 1),
        })

    span_days = int(df["date"].nunique()) or 1

    # Baseline forecast first, then supervised ML overwrites the dashboard-facing
    # forecast fields when training succeeds. This keeps the app robust.
    print(f"  hot rows assigned to zones: {len(df_hot):,}", flush=True)
    attach_hourly_forecasts(zones, span_days=span_days)
    print("  training supervised ML forecast ...", flush=True)
    ml_summary = attach_supervised_ml_forecasts(df_hot, zones, span_days=span_days)
    print(f"  ML enabled: {ml_summary.get('enabled', False)}", flush=True)
    print("  training AI early-warning detector ...", flush=True)
    ai_summary, ai_alerts = build_ai_early_warnings(df_hot, zones)
    print(f"  AI alerts: {len(ai_alerts)}", flush=True)

    # Sort by traffic-flow impact risk first, then CIRS. This directly supports Layer 4.
    zones.sort(key=lambda z: (z["flow_impact_score"], z["cis"], z["violations"]), reverse=True)
    for rank, z in enumerate(zones, 1):
        z["rank"] = rank
    return zones, ml_summary, ai_summary, ai_alerts


# --------------------------------------------------------------------------
# Layer 1. Hourly hotspot prediction + simple persistence backtest
# --------------------------------------------------------------------------
def attach_hourly_forecasts(zones: list[dict], span_days: int) -> None:
    """Attach per-hour expected demand and normalized deployment-risk forecast.

    This is deliberately simple and explainable: it uses each zone's historical
    hour-of-day pattern, then weights it by CIRS + flow-impact risk. It turns the
    dashboard from a static heatmap into an hour-by-hour enforcement forecast.
    """
    if not zones:
        return

    # First pass: expected violations per calendar day/hour and raw deployment pressure.
    by_hour_raw = [[] for _ in range(24)]
    for z in zones:
        expected = [round((cnt + 0.25) / span_days, 3) for cnt in z["hourly"]]
        # confidence rises with recurrence and sample size, capped at 100.
        confidence = min(100.0, 35 + 40 * z["recurrence"] + 25 * min(1.0, z["violations"] / 1500))
        z["expected_hourly"] = expected
        z["prediction_confidence"] = round(confidence, 1)
        z["forecast_model"] = "hour-of-day persistence × CIRS × flow-impact risk"
        raw = []
        for h, exp in enumerate(expected):
            # Hourly forecast score before normalization.
            # The expected count handles demand; CIRS + flow impact handle enforcement value.
            v = exp * (0.55 + z["cis"] / 100) * (0.55 + z["flow_impact_score"] / 100)
            # Rush-hour violations are operationally more valuable to prevent.
            if h in PEAK_HOURS:
                v *= 1.12
            raw.append(v)
            by_hour_raw[h].append(v)
        z["_forecast_raw"] = raw

    max_by_hour = [max(vals) if vals else 1 for vals in by_hour_raw]
    for z in zones:
        scores = []
        for h, raw in enumerate(z["_forecast_raw"]):
            scores.append(round(100 * raw / max(max_by_hour[h], 1e-9), 1))
        z["forecast_scores"] = scores
        z["forecast_peak_hours"] = sorted(sorted(range(24), key=lambda h: scores[h], reverse=True)[:3])
        del z["_forecast_raw"]


def build_prediction_summary(zones: list[dict]) -> dict:
    if not zones:
        return {}
    k = min(20, len(zones))
    pred_top = {z["id"] for z in sorted(zones, key=lambda z: z["predicted_validation_violations"], reverse=True)[:k]}
    actual_top = {z["id"] for z in sorted(zones, key=lambda z: z["validation_violations"], reverse=True)[:k] if z["validation_violations"] > 0}
    overlap = len(pred_top & actual_top) if actual_top else 0
    recall = round(100 * overlap / max(len(actual_top), 1), 1)
    total_valid = sum(z["validation_violations"] for z in zones)
    covered_valid = sum(z["validation_violations"] for z in zones if z["id"] in pred_top)
    coverage = round(100 * covered_valid / max(total_valid, 1), 1)
    return {
        "type": "temporal persistence backtest",
        "top_k": k,
        "recall_at_k_pct": recall,
        "validation_coverage_pct": coverage,
        "explanation": "Top zones learned from the first 70% of dates are checked against the last 30% to test whether hotspots persist.",
    }



# --------------------------------------------------------------------------
# Layer 1B. Supervised ML hotspot forecasting
# --------------------------------------------------------------------------
def _time_features(frame: pd.DataFrame, max_day_idx: int) -> pd.DataFrame:
    """Attach cyclic and rush-hour features for tree-based forecasting."""
    out = frame.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out["is_peak"] = out["hour"].isin(PEAK_HOURS).astype(int)
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    out["day_idx_norm"] = out["day_idx"] / max(max_day_idx, 1)
    return out


def _add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lag features by zone-hour time series, using only past values."""
    frame = frame.sort_values(["zone", "hour", "day_idx"]).copy()
    g = frame.groupby(["zone", "hour"], sort=False)["y"]
    frame["lag_1"] = g.shift(1).fillna(0)
    frame["lag_7"] = g.shift(7).fillna(0)
    frame["roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()).fillna(0)
    frame["roll28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean()).fillna(0)
    return frame


def attach_supervised_ml_forecasts(df_zone: pd.DataFrame, zones: list[dict], span_days: int) -> dict:
    """Train a supervised ML model to forecast zone-hour parking pressure.

    This version is intentionally fast enough for the hackathon demo. It builds
    supervised samples at zone × hour level:

    features from an earlier date window  ->  target demand in the next window.

    Backtest:
    - training sample: first 50% dates predict next 20% dates
    - validation sample: first 70% dates predict last 30% dates

    Dashboard forecast:
    - train final model on both historical windows
    - use all available data as features to predict next-period hourly pressure.
    """
    if not zones or df_zone.empty:
        return {"enabled": False, "reason": "no zone-level rows available"}

    try:
        zone_ids = sorted(int(z["id"]) for z in zones)
        zones_df = pd.DataFrame([{
            "zone": int(z["id"]),
            "lat": z["lat"],
            "lon": z["lon"],
            "cis": z["cis"],
            "flow_impact_score": z["flow_impact_score"],
            "obstruction_risk": z["obstruction_risk"],
            "recurrence": z["recurrence"],
            "peak_share_zone": z["peak_share"],
            "lane_blockage_risk_pct": z["lane_blockage_risk_pct"],
            "log_zone_violations": np.log1p(z["weighted_violations"]),
            "zone_cells": z["cells"],
        } for z in zones])

        dates = sorted(df_zone["date"].unique())
        n_dates = len(dates)
        if n_dates < 30:
            return {"enabled": False, "reason": "not enough dates for train/validation split"}

        cut50 = max(1, int(n_dates * 0.50))
        cut70 = max(cut50 + 1, int(n_dates * 0.70))

        d0_50 = set(dates[:cut50])
        d50_70 = set(dates[cut50:cut70])
        d0_70 = set(dates[:cut70])
        d70_100 = set(dates[cut70:])

        def build_window(feature_dates: set, target_dates: set, feature_days: int, target_days: int) -> pd.DataFrame:
            """One supervised table: each row is one zone-hour."""
            idx = pd.MultiIndex.from_product([zone_ids, range(24)], names=["zone", "hour"]).to_frame(index=False)

            feat_rows = df_zone[df_zone["date"].isin(feature_dates)]
            targ_rows = df_zone[df_zone["date"].isin(target_dates)]

            f = feat_rows.groupby(["zone", "hour"], as_index=False).agg(
                hist_weighted_count=("confidence", "sum"),
                hist_raw_count=("id", "size"),
                hist_days_active=("date", "nunique"),
                hist_avg_severity=("severity", "mean"),
                hist_avg_obstruction=("obstruction", "mean"),
                hist_junction_share=("junction_clean", lambda s: float((s.str.len() > 0).mean())),
            )
            y = targ_rows.groupby(["zone", "hour"], as_index=False).agg(
                target_weighted_count=("confidence", "sum"),
                target_raw_count=("id", "size"),
            )

            out = idx.merge(f, on=["zone", "hour"], how="left").merge(y, on=["zone", "hour"], how="left")
            fill_cols = [c for c in out.columns if c.startswith("hist_") or c.startswith("target_")]
            out[fill_cols] = out[fill_cols].fillna(0.0)
            out = out.merge(zones_df, on="zone", how="left")

            out["hist_weighted_per_day"] = out["hist_weighted_count"] / max(feature_days, 1)
            out["hist_raw_per_day"] = out["hist_raw_count"] / max(feature_days, 1)
            out["hist_active_ratio"] = out["hist_days_active"] / max(feature_days, 1)
            out["target_per_day"] = out["target_weighted_count"] / max(target_days, 1)

            out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
            out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
            out["is_peak"] = out["hour"].isin(PEAK_HOURS).astype(int)
            out["late_night"] = out["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

            return out

        train_tbl = build_window(d0_50, d50_70, len(d0_50), len(d50_70))
        valid_tbl = build_window(d0_70, d70_100, len(d0_70), len(d70_100))
        all_feature_tbl = build_window(set(dates), set(), n_dates, 1)

        features = [
            "hour_sin", "hour_cos", "is_peak", "late_night",
            "hist_weighted_per_day", "hist_raw_per_day", "hist_active_ratio",
            "hist_avg_severity", "hist_avg_obstruction", "hist_junction_share",
            "lat", "lon", "cis", "flow_impact_score", "obstruction_risk", "recurrence",
            "peak_share_zone", "lane_blockage_risk_pct", "log_zone_violations", "zone_cells",
        ]

        # Fill any all-zero history means for zone-hours that never appeared.
        for tbl in (train_tbl, valid_tbl, all_feature_tbl):
            tbl[features] = tbl[features].replace([np.inf, -np.inf], 0).fillna(0)

        model = RandomForestRegressor(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_tbl[features], train_tbl["target_per_day"])
        valid_pred = np.clip(model.predict(valid_tbl[features]), 0, None)

        mae = float(mean_absolute_error(valid_tbl["target_per_day"], valid_pred))
        rmse = float(mean_squared_error(valid_tbl["target_per_day"], valid_pred) ** 0.5)

        eval_tbl = valid_tbl[["zone", "target_per_day"]].copy()
        eval_tbl["pred"] = valid_pred
        actual_zone = eval_tbl.groupby("zone")["target_per_day"].sum()
        pred_zone = eval_tbl.groupby("zone")["pred"].sum()
        k = min(20, len(zone_ids))
        actual_top = set(actual_zone.sort_values(ascending=False).head(k).index.astype(int))
        pred_top = set(pred_zone.sort_values(ascending=False).head(k).index.astype(int))
        recall_at_k = round(100 * len(actual_top & pred_top) / max(len(actual_top), 1), 1)
        total_actual = float(actual_zone.sum())
        covered_actual = float(actual_zone.loc[list(pred_top & set(actual_zone.index))].sum()) if pred_top else 0.0
        coverage_at_k = round(100 * covered_actual / max(total_actual, 1e-9), 1)

        # Train final model on both supervised windows and forecast from all available history.
        final_train = pd.concat([train_tbl, valid_tbl], ignore_index=True)
        final_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        final_model.fit(final_train[features], final_train["target_per_day"])
        all_feature_tbl["ml_pred_per_day"] = np.clip(final_model.predict(all_feature_tbl[features]), 0, None)

        # Attach 24-hour arrays to every zone.
        pred_map = all_feature_tbl.pivot(index="zone", columns="hour", values="ml_pred_per_day").fillna(0)
        for z in zones:
            zid = int(z["id"])
            arr = pred_map.loc[zid].to_numpy() if zid in pred_map.index else np.zeros(24)
            z["ml_expected_hourly"] = [round(float(x), 3) for x in arr]
            z["ml_forecast_model"] = "RandomForestRegressor on zone×hour future demand"
            support = min(1.0, z["weighted_violations"] / 1500)
            z["ml_prediction_confidence"] = round(min(100, 45 + 0.30 * recall_at_k + 20 * support + 10 * z["recurrence"]), 1)

        raw_by_hour = [[] for _ in range(24)]
        for z in zones:
            raw = []
            for h, exp in enumerate(z["ml_expected_hourly"]):
                val = exp * (0.65 + z["flow_impact_score"] / 100)
                if h in PEAK_HOURS:
                    val *= 1.10
                raw.append(val)
                raw_by_hour[h].append(val)
            z["_ml_raw"] = raw

        hour_max = [max(vals) if vals else 1 for vals in raw_by_hour]
        for z in zones:
            z["ml_forecast_scores"] = [round(100 * v / max(hour_max[h], 1e-9), 1) for h, v in enumerate(z["_ml_raw"])]
            # Peak patrol window should reflect this zone's own expected demand, not
            # hour-normalized rank, otherwise a top zone can look equally "100" all day.
            z["ml_forecast_peak_hours"] = sorted(
                sorted(range(24), key=lambda h: z["ml_expected_hourly"][h], reverse=True)[:3]
            )

            # Make ML the default dashboard forecast while retaining baseline fields.
            z["expected_hourly"] = z["ml_expected_hourly"]
            z["forecast_scores"] = z["ml_forecast_scores"]
            z["forecast_peak_hours"] = z["ml_forecast_peak_hours"]
            z["prediction_confidence"] = z["ml_prediction_confidence"]
            z["forecast_model"] = z["ml_forecast_model"]
            del z["_ml_raw"]

        return {
            "enabled": True,
            "model": "RandomForestRegressor",
            "target": "future validation-weighted parking violations per zone-hour-day",
            "features": features,
            "train_window": "first 50% dates -> next 20% dates",
            "validation_window": "first 70% dates -> last 30% dates",
            "mae_per_zone_hour_day": round(mae, 4),
            "rmse_per_zone_hour_day": round(rmse, 4),
            "top_k": int(k),
            "recall_at_k_pct": recall_at_k,
            "validation_coverage_pct": coverage_at_k,
            "forecast_horizon": "next operational period, expressed as average zone-hour demand per day",
            "explanation": "Supervised random-forest model learns which zone-hour pairs remain active in a future holdout window using historical demand, time features and zone risk features.",
        }
    except Exception as exc:
        return {"enabled": False, "reason": f"ML forecast failed: {exc}"}


# --------------------------------------------------------------------------
# Layer 6. AI early-warning anomaly detector
# --------------------------------------------------------------------------
def build_ai_early_warnings(df_zone: pd.DataFrame, zones: list[dict]) -> tuple[dict, list[dict]]:
    """Detect emerging zone-hour surges with an unsupervised ML model.

    This is a real ML layer, not a hardcoded alert rule. We train an
    IsolationForest on each zone-hour's normal baseline profile and score the
    most recent period against that baseline.

    Interpretation:
    - baseline window: first 80% of dates
    - recent window: last 20% of dates
    - high anomaly score means a zone-hour is currently behaving unusually
      relative to its historical baseline, so it deserves early enforcement
      attention before the pattern hardens into a chronic choke.
    """
    if df_zone.empty or not zones:
        return {"enabled": False, "reason": "no zone-level rows available"}, []

    try:
        zone_ids = sorted(int(z["id"]) for z in zones)
        zone_meta = {int(z["id"]): z for z in zones}
        dates = sorted(df_zone["date"].unique())
        if len(dates) < 45:
            return {"enabled": False, "reason": "not enough dates for anomaly baseline"}, []

        cut = max(1, int(len(dates) * 0.80))
        baseline_dates = set(dates[:cut])
        recent_dates = set(dates[cut:])
        baseline_days = max(1, len(baseline_dates))
        recent_days = max(1, len(recent_dates))

        idx = pd.MultiIndex.from_product([zone_ids, range(24)], names=["zone", "hour"]).to_frame(index=False)

        def profile(window_dates: set, days: int, prefix: str) -> pd.DataFrame:
            rows = df_zone[df_zone["date"].isin(window_dates)]
            g = rows.groupby(["zone", "hour"], as_index=False).agg(
                weighted_count=("confidence", "sum"),
                raw_count=("id", "size"),
                active_days=("date", "nunique"),
                avg_severity=("severity", "mean"),
                avg_obstruction=("obstruction", "mean"),
                junction_share=("junction_clean", lambda s: float((s.str.len() > 0).mean())),
            )
            out = idx.merge(g, on=["zone", "hour"], how="left")
            for c in ["weighted_count", "raw_count", "active_days", "avg_severity", "avg_obstruction", "junction_share"]:
                out[c] = out[c].fillna(0.0)
            out[f"{prefix}_weighted_per_day"] = out["weighted_count"] / days
            out[f"{prefix}_raw_per_day"] = out["raw_count"] / days
            out[f"{prefix}_active_ratio"] = out["active_days"] / days
            out[f"{prefix}_avg_severity"] = out["avg_severity"]
            out[f"{prefix}_avg_obstruction"] = out["avg_obstruction"]
            out[f"{prefix}_junction_share"] = out["junction_share"]
            keep = ["zone", "hour"] + [c for c in out.columns if c.startswith(prefix + "_")]
            return out[keep]

        base = profile(baseline_dates, baseline_days, "base")
        recent = profile(recent_dates, recent_days, "recent")
        tbl = base.merge(recent, on=["zone", "hour"], how="left").fillna(0.0)

        for z in zones:
            zid = int(z["id"])
            tbl.loc[tbl.zone == zid, "flow_impact_score"] = z["flow_impact_score"]
            tbl.loc[tbl.zone == zid, "cis"] = z["cis"]
            tbl.loc[tbl.zone == zid, "recurrence"] = z["recurrence"]
            tbl.loc[tbl.zone == zid, "lane_blockage_risk_pct"] = z["lane_blockage_risk_pct"]
            tbl.loc[tbl.zone == zid, "ml_confidence"] = z.get("ml_prediction_confidence", z.get("prediction_confidence", 70))

        tbl["hour_sin"] = np.sin(2 * np.pi * tbl["hour"] / 24)
        tbl["hour_cos"] = np.cos(2 * np.pi * tbl["hour"] / 24)
        tbl["is_peak"] = tbl["hour"].isin(PEAK_HOURS).astype(int)
        tbl["surge_delta"] = tbl["recent_weighted_per_day"] - tbl["base_weighted_per_day"]
        tbl["surge_ratio"] = (tbl["recent_weighted_per_day"] + 0.10) / (tbl["base_weighted_per_day"] + 0.10)
        tbl["severity_shift"] = tbl["recent_avg_severity"] - tbl["base_avg_severity"]
        tbl["obstruction_shift"] = tbl["recent_avg_obstruction"] - tbl["base_avg_obstruction"]

        # The normal training profile represents "recent behaves like baseline".
        normal = tbl.copy()
        normal["recent_weighted_per_day"] = normal["base_weighted_per_day"]
        normal["recent_raw_per_day"] = normal["base_raw_per_day"]
        normal["recent_active_ratio"] = normal["base_active_ratio"]
        normal["surge_delta"] = 0.0
        normal["surge_ratio"] = 1.0
        normal["severity_shift"] = 0.0
        normal["obstruction_shift"] = 0.0

        features = [
            "base_weighted_per_day", "recent_weighted_per_day", "surge_delta", "surge_ratio",
            "base_active_ratio", "recent_active_ratio", "base_avg_severity", "recent_avg_severity",
            "base_avg_obstruction", "recent_avg_obstruction", "base_junction_share", "recent_junction_share",
            "severity_shift", "obstruction_shift", "flow_impact_score", "cis", "recurrence",
            "lane_blockage_risk_pct", "ml_confidence", "hour_sin", "hour_cos", "is_peak",
        ]
        for frame in (tbl, normal):
            frame[features] = frame[features].replace([np.inf, -np.inf], 0).fillna(0)

        detector = IsolationForest(
            n_estimators=160,
            contamination=0.08,
            random_state=42,
            n_jobs=-1,
        )
        detector.fit(normal[features])

        # Higher = more abnormal.
        raw_anom = -detector.decision_function(tbl[features])
        lo, hi = float(raw_anom.min()), float(raw_anom.max())
        tbl["anomaly_score"] = 100 * (raw_anom - lo) / max(hi - lo, 1e-9)
        tbl["early_warning_score"] = (
            0.42 * tbl["anomaly_score"] +
            0.24 * tbl["flow_impact_score"] +
            0.16 * np.minimum(100, 25 * np.log1p(tbl["surge_ratio"])) +
            0.10 * tbl["ml_confidence"] +
            0.08 * tbl["lane_blockage_risk_pct"]
        )

        # Keep only meaningful recent surges, not tiny statistical artifacts.
        candidates = tbl[
            (tbl["recent_weighted_per_day"] >= 0.15) &
            ((tbl["surge_delta"] > 0.05) | (tbl["surge_ratio"] >= 1.35))
        ].copy()
        if candidates.empty:
            candidates = tbl.sort_values("early_warning_score", ascending=False).head(12).copy()

        candidates = candidates.sort_values("early_warning_score", ascending=False).head(18)
        alerts: list[dict] = []
        for r in candidates.itertuples(index=False):
            z = zone_meta.get(int(r.zone))
            if not z:
                continue
            level = "Critical" if r.early_warning_score >= 82 else "High" if r.early_warning_score >= 68 else "Watch"
            reason_bits = []
            if r.surge_ratio >= 1.75:
                reason_bits.append(f"{r.surge_ratio:.1f}× recent surge")
            elif r.surge_delta > 0.1:
                reason_bits.append("recent demand above baseline")
            if r.hour in PEAK_HOURS:
                reason_bits.append("peak-hour window")
            if z.get("flow_impact_band") in ("Severe", "High"):
                reason_bits.append(f"{z.get('flow_impact_band')} flow-risk zone")
            if not reason_bits:
                reason_bits.append("abnormal recent zone-hour pattern")
            alerts.append({
                "zone_id": int(r.zone),
                "zone_name": z["name"],
                "hour": int(r.hour),
                "alert_level": level,
                "early_warning_score": round(float(r.early_warning_score), 1),
                "anomaly_score": round(float(r.anomaly_score), 1),
                "recent_per_day": round(float(r.recent_weighted_per_day), 3),
                "baseline_per_day": round(float(r.base_weighted_per_day), 3),
                "surge_ratio": round(float(r.surge_ratio), 2),
                "flow_impact_score": round(float(z["flow_impact_score"]), 1),
                "reason": " · ".join(reason_bits[:3]),
                "recommended_action": "Pre-position patrol/tow unit" if level != "Watch" else "Watchlist and rotate patrol",
            })

        # Attach the strongest alert back to each zone for modals/list badges.
        by_zone: dict[int, list[dict]] = {}
        for a in alerts:
            by_zone.setdefault(a["zone_id"], []).append(a)
        for z in zones:
            items = sorted(by_zone.get(int(z["id"]), []), key=lambda a: a["early_warning_score"], reverse=True)
            if items:
                z["ai_early_warning_score"] = items[0]["early_warning_score"]
                z["ai_alert_level"] = items[0]["alert_level"]
                z["ai_alert_hours"] = sorted({a["hour"] for a in items[:3]})
                z["ai_alert_reason"] = items[0]["reason"]
                z["ai_anomaly_score"] = items[0]["anomaly_score"]
                z["ai_recent_baseline_ratio"] = items[0]["surge_ratio"]
            else:
                z["ai_early_warning_score"] = 0
                z["ai_alert_level"] = "Normal"
                z["ai_alert_hours"] = []
                z["ai_alert_reason"] = "No abnormal recent surge detected"
                z["ai_anomaly_score"] = 0
                z["ai_recent_baseline_ratio"] = 1.0

        return {
            "enabled": True,
            "model": "IsolationForest",
            "purpose": "unsupervised early-warning detection for abnormal recent zone-hour surges",
            "baseline_window": f"first {baseline_days} unique dates",
            "recent_window": f"last {recent_days} unique dates",
            "features": features,
            "alerts_generated": len(alerts),
            "critical_or_high_alerts": sum(1 for a in alerts if a["alert_level"] in ("Critical", "High")),
            "explanation": "The detector learns normal zone-hour behavior from the historical baseline and flags recent surges that are abnormal relative to that baseline.",
        }, alerts
    except Exception as exc:
        return {"enabled": False, "reason": f"AI early-warning detector failed: {exc}"}, []


# --------------------------------------------------------------------------
# 6. Temporal demand surface
# --------------------------------------------------------------------------
def build_temporal(df: pd.DataFrame) -> dict:
    mat = np.zeros((7, 24), dtype=int)
    for wd, hr in zip(df.weekday, df.hour):
        mat[wd, hr] += 1
    hourly = df["hour"].value_counts().sort_index()
    hourly_arr = [int(hourly.get(h, 0)) for h in range(24)]
    weekday_tot = mat.sum(axis=1).tolist()
    peak_hour = int(np.argmax(hourly_arr))
    peak_share = float(np.mean([h in PEAK_HOURS for h in df.hour]))
    return {
        "matrix": mat.tolist(),          # [weekday][hour]
        "hourly": hourly_arr,
        "weekday_totals": [int(x) for x in weekday_tot],
        "peak_hour": peak_hour,
        "peak_share": round(peak_share, 3),
        "weekday_labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    }


# --------------------------------------------------------------------------
# 7. Categorical breakdowns
# --------------------------------------------------------------------------
def build_breakdowns(df: pd.DataFrame) -> dict:
    vio = Counter(v for vs in df["violations"] for v in vs if PARK_RE.search(v))
    veh = Counter(df["vehicle_type"].dropna())
    sta = Counter(df["police_station"].dropna())
    jun = Counter(j for j in df["junction_clean"] if j)

    def topn(c: Counter, n: int):
        return [{"label": k, "n": int(v)} for k, v in c.most_common(n)]

    return {
        "violations": topn(vio, 10),
        "vehicles": topn(veh, 8),
        "stations": topn(sta, 12),
        "junctions": topn(jun, 10),
    }


# --------------------------------------------------------------------------
# Orchestrate
# --------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()

    print("Building grid cells ...", flush=True)
    cells = build_cells(df)
    print(f"  cells: {len(cells):,}", flush=True)
    print("Building zones + ML forecasts ...", flush=True)
    zones, ml_prediction_summary, ai_early_warning_summary, ai_alerts = build_zones(df, cells)
    print(f"  zones: {len(zones):,}", flush=True)
    print("Building temporal/category analytics ...", flush=True)
    temporal = build_temporal(df)
    breakdowns = build_breakdowns(df)
    prediction_summary = build_prediction_summary(zones)

    # ---- Heatmap points (cap for browser performance) -------------------
    # Use Layer 4 flow impact so the heatmap represents enforcement value, not only volume.
    heat = cells.sort_values("flow_impact_score", ascending=False).head(4000)
    heat_points = [
        [round(r.lat, 5), round(r.lon, 5), round(float(r.flow_impact_score) / 100, 3)]
        for r in heat.itertuples()
    ]

    # ---- Summary KPIs ---------------------------------------------------
    span_days = int(df["date"].nunique()) or 1
    peak_share = temporal["peak_share"]

    total_zone_violations = sum(z["violations"] for z in zones)
    top20_zone_violations = sum(z["violations"] for z in zones[:20])
    concentration = round(100 * top20_zone_violations / max(total_zone_violations, 1), 1) if zones else 0

    total_weighted_impact = sum(z["violations"] * z["flow_impact_score"] for z in zones)
    top20_weighted_impact = sum(z["violations"] * z["flow_impact_score"] for z in zones[:20])
    impact_concentration = round(100 * top20_weighted_impact / max(total_weighted_impact, 1), 1) if zones else 0

    severe_zones = sum(1 for z in zones if z["flow_impact_band"] == "Severe")
    avg_flow = round(float(np.mean([z["flow_impact_score"] for z in zones])), 1) if zones else 0
    avg_pred_conf = round(float(np.mean([z["prediction_confidence"] for z in zones])), 1) if zones else 0

    # Impact-band breakdown from zones.
    band_order = ["Severe", "High", "Moderate", "Low"]
    band_counts = Counter(z["flow_impact_band"] for z in zones)
    breakdowns["impact_bands"] = [{"label": b, "n": int(band_counts.get(b, 0))} for b in band_order]

    scheduler_summary = {
        "enabled": True,
        "name": "ParkSched",
        "algorithm": "MLFQ-inspired priority scheduling with aging",
        "default_units": 5,
        "default_slots": 3,
        "quantum_minutes": 30,
        "need_score": "0.35*ML forecast + 0.25*flow impact + 0.15*recurrence + 0.15*ML confidence + 0.10*lane blockage",
        "queues": [
            {"id": 0, "name": "Q0 Critical", "rule": "need >= 85 or severe flow-impact zone", "action": "Tow + e-challan"},
            {"id": 1, "name": "Q1 High", "rule": "65 <= need < 85", "action": "Patrol + challan"},
            {"id": 2, "name": "Q2 Watch", "rule": "40 <= need < 65", "action": "Rotation patrol"},
            {"id": 3, "name": "Q3 Low", "rule": "need < 40", "action": "Passive monitoring"},
        ],
        "aging_rule": "Zones not assigned in a slot receive an aging boost before the next slot; repeated assignments receive a feedback penalty unless the zone is critical.",
    }

    summary = {
        "generated_from_rows": int(len(df)),
        "date_start": str(df["date"].min()),
        "date_end": str(df["date"].max()),
        "span_days": int(span_days),
        "n_zones": len(zones),
        "n_hot_cells": int(len(cells)),
        "daily_avg": round(len(df) / span_days, 0),
        "peak_hour": temporal["peak_hour"],
        "peak_share_pct": round(peak_share * 100, 1),
        "top_zone": zones[0]["name"] if zones else "",
        "top_zone_cis": zones[0]["cis"] if zones else 0,
        "top_zone_flow_impact": zones[0]["flow_impact_score"] if zones else 0,
        "top20_concentration_pct": concentration,
        "top20_impact_pct": impact_concentration,
        "avg_flow_impact_score": avg_flow,
        "severe_impact_zones": severe_zones,
        "avg_prediction_confidence": avg_pred_conf,
        "prediction_backtest": prediction_summary,
        "ml_prediction_backtest": ml_prediction_summary,
        "ml_model_enabled": bool(ml_prediction_summary.get("enabled", False)),
        "ai_early_warning": ai_early_warning_summary,
        "ai_alert_count": int(len(ai_alerts)),
        "scheduler": scheduler_summary,
        "city_center": [12.9716, 77.5946],
        "score_note": "CIRS/flow-impact scores are interpretable risk proxies; production calibration should use speed, queue-length or lane-occupancy feeds.",
    }

    def dump(name, obj):
        path = os.path.join(OUT_DIR, name)
        with open(path, "w") as f:
            json.dump(obj, f, separators=(",", ":"))
        print(f"  wrote {name}  ({os.path.getsize(path)/1024:.0f} KB)")

    print("Writing artifacts ...")
    dump("summary.json", summary)
    dump("hotspots.json", {"points": heat_points})
    dump("zones.json", {"zones": zones})
    dump("temporal.json", temporal)
    dump("breakdowns.json", breakdowns)
    dump("ai_alerts.json", {"alerts": ai_alerts, "summary": ai_early_warning_summary})
    print(f"\nDone. {len(zones)} enforcement zones, top: "
          f"{summary['top_zone']} (Flow impact {summary['top_zone_flow_impact']}, CIRS {summary['top_zone_cis']}).")


if __name__ == "__main__":
    main()
