"""Pandas analysis utilities.

Requirement coverage: procedural functions, pandas groupby/sort/filter,
loops, dicts.
"""

from __future__ import annotations

import pandas as pd


def records_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert cleaned/enriched records to a DataFrame with stable columns."""
    df = pd.DataFrame(rows)
    # Requirement coverage: loop over a fixed list of expected columns
    for col in ["price", "size", "rooms", "balcony", "parking"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def average_price_by_rooms(df: pd.DataFrame) -> pd.Series:
    """Requirement coverage: groupby + mean."""
    return df.groupby("rooms")["price"].mean().sort_index()


def sort_by_price(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Requirement coverage: sort_values."""
    return df.sort_values("price", ascending=ascending)


def filter_expensive(df: pd.DataFrame, threshold: float = 2000.0) -> pd.DataFrame:
    """Requirement coverage: boolean indexing."""
    return df[df["price"] > threshold]


def summary_stats(df: pd.DataFrame) -> dict:
    """High-level summary as a dict. Requirement coverage: dict construction."""
    return {
        "n_listings": int(len(df)),
        "n_cities": int(df["city"].nunique()) if "city" in df.columns else 0,
        "mean_price": float(df["price"].mean()) if len(df) else 0.0,
        "median_price": float(df["price"].median()) if len(df) else 0.0,
        "mean_size": float(df["size"].mean()) if len(df) else 0.0,
        "mean_rooms": float(df["rooms"].mean()) if len(df) else 0.0,
    }
