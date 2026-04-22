"""Seaborn + matplotlib visualizations.

Requirement coverage: visualizations (scatter + boxplot + bar),
procedural functions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGDIR = Path(__file__).resolve().parents[1] / "figures"


def _ensure_figdir() -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    return FIGDIR


def scatter_size_price(df: pd.DataFrame, save: bool = False):
    """Scatter of size vs price."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="size", y="price", hue="rooms",
                    palette="viridis", ax=ax)
    ax.set_title("Swiss housing: size vs. price")
    ax.set_xlabel("Size (m²)")
    ax.set_ylabel("Price (CHF/month)")
    if save:
        fig.savefig(_ensure_figdir() / "scatter_size_price.png", bbox_inches="tight")
    return fig


def boxplot_rooms_price(df: pd.DataFrame, save: bool = False):
    """Boxplot of price by number of rooms."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="rooms", y="price", ax=ax)
    ax.set_title("Swiss housing: price distribution by number of rooms")
    ax.set_xlabel("Rooms")
    ax.set_ylabel("Price (CHF/month)")
    if save:
        fig.savefig(_ensure_figdir() / "boxplot_rooms_price.png", bbox_inches="tight")
    return fig


def barplot_avg_price_by_city(df: pd.DataFrame, save: bool = False,
                              top_n: int = 15, min_listings: int = 2):
    """Average price by city — top N cities with at least `min_listings`.

    Zurich-canton sampling yields ~50 cities, many with a single listing
    (which makes the "average" meaningless noise). We filter to cities
    with at least 2 listings, then show the top N by mean price.
    """
    grouped = df.groupby("city")["price"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= min_listings]
    agg = grouped.sort_values("mean", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=agg, x="city", y="mean", ax=ax, color="steelblue")
    ax.set_title(
        f"Top {len(agg)} cities by average price "
        f"(≥ {min_listings} listings each)"
    )
    ax.set_xlabel("City")
    ax.set_ylabel("Avg price (CHF/month)")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    fig.tight_layout()
    if save:
        fig.savefig(_ensure_figdir() / "barplot_avg_price_by_city.png", bbox_inches="tight")
    return fig
