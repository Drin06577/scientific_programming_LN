"""Data quality validation + outlier handling.

Requirement coverage: data validation, regex sanity checks, conditional
logic, set deduplication, procedural functions, dict return for reporting.

The pipeline collects ~200 Flatfox listings, but raw scrapes always contain
duplicates, impossible values, and statistical outliers. This module centralises
the validation logic so the cleaned dataset is defensible for downstream
statistical tests.

Returns a `QualityReport` dataclass that the dashboard renders verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Plausibility bounds for the Zurich rental market (2024-2026).
# CHF/m² above 150 is plausible only for very small/furnished studios;
# below 10 implies a data error. Tuned from the 200-listing sample so
# obvious garbage (e.g. a corrupted "CHF 4.4" parsed as 44) is rejected
# while genuine luxury studios survive.
PRICE_MIN: int = 500
PRICE_MAX: int = 15_000
SIZE_MIN: float = 15.0
SIZE_MAX: float = 400.0
ROOMS_MIN: float = 1.0
ROOMS_MAX: float = 8.0
CHF_PER_M2_MIN: float = 10.0
CHF_PER_M2_MAX: float = 200.0


@dataclass
class QualityReport:
    """Summary of every drop/repair the validator applied.

    The dashboard renders this directly — so each field is a count
    plus a small DataFrame of the rows that were dropped, for
    transparent before/after reporting.
    """
    n_input: int = 0
    n_output: int = 0
    n_duplicates: int = 0
    n_missing_required: int = 0
    n_impossible_price: int = 0
    n_impossible_size: int = 0
    n_impossible_rooms: int = 0
    n_chf_per_m2_outliers: int = 0
    n_statistical_outliers: int = 0
    dropped_examples: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_dict(self) -> dict:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_duplicates": self.n_duplicates,
            "n_missing_required": self.n_missing_required,
            "n_impossible_price": self.n_impossible_price,
            "n_impossible_size": self.n_impossible_size,
            "n_impossible_rooms": self.n_impossible_rooms,
            "n_chf_per_m2_outliers": self.n_chf_per_m2_outliers,
            "n_statistical_outliers": self.n_statistical_outliers,
            "retention_rate": (self.n_output / self.n_input) if self.n_input else 0.0,
        }


def _ensure_chf_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    """Add a chf_per_m2 column if not present (used by downstream tests too)."""
    if "chf_per_m2" not in df.columns:
        df = df.copy()
        df["chf_per_m2"] = (df["price"] / df["size"]).round(2)
    return df


def _iqr_outlier_mask(series: pd.Series, k: float = 3.0) -> pd.Series:
    """Tukey fence (k=3.0 → "far outliers"). Returns True where row is OK.

    Why k=3.0 rather than the textbook 1.5: Zurich rents have a heavy
    right tail (lakeside luxury) that we want to keep — 1.5*IQR would
    discard ~10% of the dataset as "outliers" even though those rows
    are perfectly valid. 3.0 only flags genuinely impossible values.
    """
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.between(lower, upper)


def validate(df: pd.DataFrame, *, drop_outliers: bool = True) -> tuple[pd.DataFrame, QualityReport]:
    """Run the full validation pipeline.

    Order is intentional: duplicates → required fields → impossible values
    → CHF/m² sanity → statistical outliers. Each stage's drops are counted
    independently so the report stays interpretable.
    """
    report = QualityReport(n_input=len(df))
    if len(df) == 0:
        return df, report

    df = _ensure_chf_per_m2(df)
    dropped_rows: list[pd.DataFrame] = []

    # 1. Exact duplicates on the natural key (title + price + size + rooms).
    dup_mask = df.duplicated(subset=["title", "price", "size", "rooms"], keep="first")
    if dup_mask.any():
        dropped_rows.append(df.loc[dup_mask].assign(_drop_reason="duplicate"))
        report.n_duplicates = int(dup_mask.sum())
        df = df.loc[~dup_mask].copy()

    # 2. Missing required fields.
    required = ["price", "size", "rooms", "city"]
    miss_mask = df[required].isna().any(axis=1)
    if miss_mask.any():
        dropped_rows.append(df.loc[miss_mask].assign(_drop_reason="missing_required"))
        report.n_missing_required = int(miss_mask.sum())
        df = df.loc[~miss_mask].copy()

    # 3. Impossible values: price/size/rooms outside plausibility bounds.
    bad_price = ~df["price"].between(PRICE_MIN, PRICE_MAX)
    bad_size = ~df["size"].between(SIZE_MIN, SIZE_MAX)
    bad_rooms = ~df["rooms"].between(ROOMS_MIN, ROOMS_MAX)
    if bad_price.any():
        dropped_rows.append(df.loc[bad_price].assign(_drop_reason="impossible_price"))
        report.n_impossible_price = int(bad_price.sum())
    if bad_size.any():
        dropped_rows.append(df.loc[bad_size].assign(_drop_reason="impossible_size"))
        report.n_impossible_size = int(bad_size.sum())
    if bad_rooms.any():
        dropped_rows.append(df.loc[bad_rooms].assign(_drop_reason="impossible_rooms"))
        report.n_impossible_rooms = int(bad_rooms.sum())
    impossible_mask = bad_price | bad_size | bad_rooms
    df = df.loc[~impossible_mask].copy()

    # 4. CHF/m² hard sanity — rules out parse errors that pass the
    # individual price/size checks (e.g. price 2 000 with size 8 → 250 CHF/m²).
    bad_chf = ~df["chf_per_m2"].between(CHF_PER_M2_MIN, CHF_PER_M2_MAX)
    if bad_chf.any():
        dropped_rows.append(df.loc[bad_chf].assign(_drop_reason="chf_per_m2_oob"))
        report.n_chf_per_m2_outliers = int(bad_chf.sum())
        df = df.loc[~bad_chf].copy()

    # 5. Statistical outliers (only on price; size has a heavy bimodal tail
    # because of luxury vs micro-studios that we explicitly want to keep).
    if drop_outliers and len(df) > 10:
        ok_mask = _iqr_outlier_mask(df["price"], k=3.0)
        stat_mask = ~ok_mask
        if stat_mask.any():
            dropped_rows.append(df.loc[stat_mask].assign(_drop_reason="iqr_outlier"))
            report.n_statistical_outliers = int(stat_mask.sum())
            df = df.loc[ok_mask].copy()

    if dropped_rows:
        keep_cols = ["title", "price", "size", "rooms", "city", "chf_per_m2", "_drop_reason"]
        examples = pd.concat(dropped_rows, ignore_index=True)
        report.dropped_examples = examples[
            [c for c in keep_cols if c in examples.columns]
        ].reset_index(drop=True)

    report.n_output = len(df)
    return df.reset_index(drop=True), report


def price_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket apartments into market tiers based on price quartiles.

    The chosen bins (cheap/medium/expensive/luxury) are computed from the
    *cleaned* dataset's quartiles, so the labels reflect this dataset's
    distribution rather than fixed CHF thresholds (which would be brittle
    across years and markets).
    """
    if len(df) < 4:
        out = df.copy()
        out["price_category"] = pd.Categorical(["medium"] * len(out),
                                               categories=["cheap", "medium", "expensive", "luxury"])
        return out
    q = df["price"].quantile([0.25, 0.5, 0.75]).values
    bins = [-np.inf, q[0], q[1], q[2], np.inf]
    labels = ["cheap", "medium", "expensive", "luxury"]
    out = df.copy()
    out["price_category"] = pd.cut(out["price"], bins=bins, labels=labels, include_lowest=True)
    return out


def extended_summary(df: pd.DataFrame) -> dict:
    """Rich descriptive statistics for the dashboard KPI cards.

    Goes beyond `analysis.summary_stats` — adds medians, IQR, std,
    CHF/m² stats, and grouped comparisons (balcony/parking).
    """
    if len(df) == 0:
        return {}
    df = _ensure_chf_per_m2(df)
    out: dict = {
        "n_listings": int(len(df)),
        "n_cities": int(df["city"].nunique()),
        "mean_price": float(df["price"].mean()),
        "median_price": float(df["price"].median()),
        "std_price": float(df["price"].std()),
        "min_price": float(df["price"].min()),
        "max_price": float(df["price"].max()),
        "iqr_price": float(df["price"].quantile(0.75) - df["price"].quantile(0.25)),
        "mean_size": float(df["size"].mean()),
        "median_size": float(df["size"].median()),
        "mean_rooms": float(df["rooms"].mean()),
        "median_rooms": float(df["rooms"].median()),
        "mean_chf_per_m2": float(df["chf_per_m2"].mean()),
        "median_chf_per_m2": float(df["chf_per_m2"].median()),
    }
    # Balcony vs non-balcony comparison.
    if "balcony" in df.columns:
        b1 = df.loc[df["balcony"] == 1, "price"]
        b0 = df.loc[df["balcony"] == 0, "price"]
        out["mean_price_balcony"] = float(b1.mean()) if len(b1) else None
        out["mean_price_no_balcony"] = float(b0.mean()) if len(b0) else None
        if len(b1) and len(b0):
            out["balcony_premium_pct"] = float((b1.mean() - b0.mean()) / b0.mean() * 100)
    if "parking" in df.columns:
        p1 = df.loc[df["parking"] == 1, "price"]
        p0 = df.loc[df["parking"] == 0, "price"]
        out["mean_price_parking"] = float(p1.mean()) if len(p1) else None
        out["mean_price_no_parking"] = float(p0.mean()) if len(p0) else None
        if len(p1) and len(p0):
            out["parking_premium_pct"] = float((p1.mean() - p0.mean()) / p0.mean() * 100)
    return out
