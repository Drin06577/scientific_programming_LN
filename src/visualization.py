"""Visualizations — both static (matplotlib/seaborn) and interactive (Plotly).

Requirement coverage: visualizations, procedural functions, multiple chart
types (scatter with regression, horizontal bar, boxplot, violin, histogram,
correlation heatmap, geographic map, pairplot/scatter-matrix).

The static charts (scatter_size_price, boxplot_rooms_price, barplot_avg_price_by_city)
are preserved so the notebook keeps working unchanged. The Plotly variants
power the new Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

FIGDIR = Path(__file__).resolve().parents[1] / "figures"

# Consistent professional palette — used across all Plotly charts.
PALETTE_PRIMARY = "#2E5C8A"
PALETTE_ACCENT = "#D97706"
PALETTE_SUCCESS = "#16A34A"
PALETTE_DANGER = "#DC2626"
PALETTE_NEUTRAL = "#6B7280"
PLOTLY_TEMPLATE = "simple_white"

CHF_FMT = "CHF {:,.0f}".format


def _ensure_figdir() -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    return FIGDIR


def _ensure_chf_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    if "chf_per_m2" not in df.columns:
        df = df.copy()
        df["chf_per_m2"] = (df["price"] / df["size"]).round(2)
    return df


# ===========================================================================
# Static charts (kept for the notebook + figures/ exports)
# ===========================================================================

def scatter_size_price(df: pd.DataFrame, save: bool = False):
    """Static scatter (matplotlib) — kept for the notebook walkthrough."""
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
    grouped = df.groupby("city")["price"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= min_listings]
    agg = grouped.sort_values("mean", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=agg, x="city", y="mean", ax=ax, color="steelblue")
    ax.set_title(f"Top {len(agg)} cities by average price (≥ {min_listings} listings)")
    ax.set_xlabel("City")
    ax.set_ylabel("Avg price (CHF/month)")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    fig.tight_layout()
    if save:
        fig.savefig(_ensure_figdir() / "barplot_avg_price_by_city.png", bbox_inches="tight")
    return fig


# ===========================================================================
# Plotly — interactive dashboard charts
# ===========================================================================

def plotly_city_bar(df: pd.DataFrame, top_n: int = 15, min_listings: int = 3) -> go.Figure:
    """Horizontal bar of average rent per city, sorted, with listing-count tooltip."""
    agg = (df.groupby("city")["price"]
             .agg(avg_price="mean", n_listings="count")
             .reset_index())
    agg = agg[agg["n_listings"] >= min_listings]
    agg = agg.sort_values("avg_price", ascending=False).head(top_n)
    agg = agg.sort_values("avg_price", ascending=True)  # ascending → top-priced at top

    fig = go.Figure()
    fig.add_bar(
        x=agg["avg_price"],
        y=agg["city"],
        orientation="h",
        marker=dict(color=agg["avg_price"], colorscale="Blues",
                    colorbar=dict(title="CHF/month", thickness=12)),
        text=[CHF_FMT(v) for v in agg["avg_price"]],
        textposition="outside",
        customdata=agg[["n_listings"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Average rent: %{x:,.0f} CHF/month<br>"
            "Listings: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        title=f"Average rent by city — top {len(agg)} (≥ {min_listings} listings)",
        xaxis_title="Average rent (CHF/month)",
        yaxis_title="",
        template=PLOTLY_TEMPLATE,
        height=max(420, 28 * len(agg) + 100),
        margin=dict(l=20, r=120, t=60, b=40),
    )
    return fig


def plotly_city_boxplot_chf_per_m2(df: pd.DataFrame, top_n: int = 8,
                                   min_listings: int = 4) -> go.Figure:
    """Box plot of CHF/m² per city for the cities with the largest samples."""
    df = _ensure_chf_per_m2(df)
    counts = df.groupby("city").size().sort_values(ascending=False)
    cities = counts[counts >= min_listings].head(top_n).index.tolist()
    sub = df[df["city"].isin(cities)].copy()
    if sub.empty:
        return _empty_chart("Not enough listings per city for box plot")
    median_order = (sub.groupby("city")["chf_per_m2"].median()
                       .sort_values(ascending=False).index.tolist())
    fig = px.box(
        sub, x="city", y="chf_per_m2",
        category_orders={"city": median_order},
        points="outliers",
        color="city",
        title=f"CHF per m² distribution by city (top {len(cities)} by sample size)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="CHF / m²",
                      height=460)
    return fig


def plotly_scatter_size_price(df: pd.DataFrame) -> go.Figure:
    """Scatter of size vs price with OLS regression line + CHF/m² colour.

    Annotation in the corner pre-computes the Pearson r so the reader sees the
    headline number without leaving the chart.
    """
    df = _ensure_chf_per_m2(df)
    sub = df[["size", "price", "chf_per_m2", "rooms", "city", "title"]].dropna()
    if len(sub) < 3:
        return _empty_chart("Not enough rows for scatter")

    fig = px.scatter(
        sub,
        x="size", y="price",
        color="chf_per_m2",
        color_continuous_scale="Viridis",
        hover_data={"city": True, "rooms": True, "chf_per_m2": ":.1f",
                    "title": False, "size": ":.0f", "price": ":,.0f"},
        opacity=0.78,
        title="Size vs price — coloured by CHF/m²",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))

    # OLS regression line + 95% CI band.
    slope, intercept = np.polyfit(sub["size"], sub["price"], 1)
    xs = np.linspace(sub["size"].min(), sub["size"].max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=intercept + slope * xs,
        mode="lines", name="OLS fit",
        line=dict(color=PALETTE_DANGER, width=2.5, dash="dash"),
        hovertemplate=f"price = {intercept:,.0f} + {slope:.1f} × m²<extra></extra>",
    ))

    r = sub["size"].corr(sub["price"])
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98,
        text=(f"<b>Pearson r = {r:.3f}</b><br>R² = {r**2:.3f}<br>"
              f"slope = CHF {slope:.0f}/m²"),
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor=PALETTE_NEUTRAL, borderwidth=1, font=dict(size=12),
    )
    fig.update_layout(
        xaxis_title="Living space (m²)", yaxis_title="Price (CHF/month)",
        coloraxis_colorbar=dict(title="CHF/m²"), height=520,
    )
    return fig


def plotly_rooms_distribution(df: pd.DataFrame) -> go.Figure:
    """Side-by-side violin + box of price by room count, with sample sizes."""
    sub = df.dropna(subset=["rooms", "price"]).copy()
    sub["rooms"] = sub["rooms"].astype(str)
    if sub.empty:
        return _empty_chart("No room data available")

    counts = sub.groupby("rooms").size()
    order = sorted(sub["rooms"].unique(), key=lambda x: float(x))
    sub["rooms_label"] = sub["rooms"].apply(
        lambda r: f"{r} rooms<br>(n={counts.get(r, 0)})"
    )
    label_order = [f"{r} rooms<br>(n={counts.get(r, 0)})" for r in order]

    fig = px.violin(
        sub, x="rooms_label", y="price",
        box=True, points="all",
        category_orders={"rooms_label": label_order},
        color="rooms_label",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        title="Rent distribution by room count (violin + box + raw points)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False,
                      xaxis_title="", yaxis_title="Price (CHF/month)",
                      height=520)
    return fig


def plotly_chf_per_m2_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of CHF/m² with median + mean reference lines."""
    df = _ensure_chf_per_m2(df)
    sub = df["chf_per_m2"].dropna()
    if sub.empty:
        return _empty_chart("No CHF/m² data")
    median = float(sub.median())
    mean = float(sub.mean())
    fig = px.histogram(
        sub, nbins=30,
        title="Distribution of CHF per m² across the Zurich-canton sample",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[PALETTE_PRIMARY],
    )
    fig.add_vline(x=median, line_dash="dash", line_color=PALETTE_DANGER,
                  annotation_text=f"median = {median:.1f}", annotation_position="top right")
    fig.add_vline(x=mean, line_dash="dot", line_color=PALETTE_ACCENT,
                  annotation_text=f"mean = {mean:.1f}", annotation_position="bottom right")
    fig.update_layout(xaxis_title="CHF / m²", yaxis_title="Number of listings",
                      bargap=0.05, showlegend=False, height=440)
    return fig


def plotly_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Pearson correlation matrix over the numeric columns."""
    df = _ensure_chf_per_m2(df)
    cols = [c for c in ["price", "size", "rooms", "balcony", "parking",
                        "llm_furnished", "chf_per_m2"] if c in df.columns]
    corr = df[cols].corr(numeric_only=True).round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        hovertemplate="%{y} ↔ %{x}: r = %{z:.2f}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(
        title="Correlation matrix — numeric features",
        template=PLOTLY_TEMPLATE,
        height=520,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plotly_geographic_map(df: pd.DataFrame, city_centroids: dict | None = None) -> go.Figure:
    """Bubble map over Zurich canton — coordinates joined from `city_centroids`.

    The function tolerates a missing centroid mapping: rows without coordinates
    are silently skipped, and an explanatory annotation appears when nothing is
    plottable.
    """
    centroids = city_centroids or _DEFAULT_ZURICH_CENTROIDS
    df = _ensure_chf_per_m2(df)
    rows: list[dict] = []
    for city, group in df.groupby("city"):
        coords = centroids.get(city)
        if not coords:
            continue
        rows.append({
            "city": city,
            "lat": coords[0],
            "lon": coords[1],
            "avg_price": float(group["price"].mean()),
            "avg_chf_per_m2": float(group["chf_per_m2"].mean()),
            "n_listings": int(len(group)),
        })
    if not rows:
        return _empty_chart("No city centroids matched — map unavailable")

    map_df = pd.DataFrame(rows)
    fig = px.scatter_map(
        map_df,
        lat="lat", lon="lon",
        size="n_listings",
        color="avg_chf_per_m2",
        color_continuous_scale="Plasma",
        hover_name="city",
        hover_data={"avg_price": ":,.0f", "avg_chf_per_m2": ":.1f",
                    "n_listings": True, "lat": False, "lon": False},
        size_max=42, zoom=8.5,
        title="Zurich canton — average CHF/m² by city",
    )
    fig.update_layout(
        map_style="open-street-map",
        height=560, margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_colorbar=dict(title="CHF/m²"),
    )
    return fig


def plotly_scatter_matrix(df: pd.DataFrame) -> go.Figure:
    """Scatter matrix (pairplot) over the key numeric variables."""
    df = _ensure_chf_per_m2(df)
    cols = [c for c in ["price", "size", "rooms", "chf_per_m2"] if c in df.columns]
    fig = px.scatter_matrix(
        df.dropna(subset=cols),
        dimensions=cols,
        color="rooms" if "rooms" in df.columns else None,
        color_continuous_scale="Viridis",
        title="Scatter matrix — price, size, rooms, CHF/m²",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False,
                      marker=dict(size=4, opacity=0.6))
    fig.update_layout(height=620)
    return fig


def plotly_price_categories(df: pd.DataFrame) -> go.Figure:
    """Distribution of apartments across cheap/medium/expensive/luxury tiers."""
    from .data_quality import price_categories
    df = price_categories(df)
    if "price_category" not in df.columns:
        return _empty_chart("price_category not available")
    counts = (df["price_category"].value_counts()
                .reindex(["cheap", "medium", "expensive", "luxury"])
                .fillna(0).astype(int).reset_index())
    counts.columns = ["category", "n"]
    color_map = {
        "cheap": PALETTE_SUCCESS, "medium": PALETTE_PRIMARY,
        "expensive": PALETTE_ACCENT, "luxury": PALETTE_DANGER,
    }
    fig = go.Figure(go.Bar(
        x=counts["category"], y=counts["n"],
        marker_color=[color_map[c] for c in counts["category"]],
        text=counts["n"], textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y} listings<extra></extra>",
    ))
    fig.update_layout(
        title="Listings by price tier (quartile bucketing)",
        xaxis_title="", yaxis_title="Number of listings",
        template=PLOTLY_TEMPLATE, height=400, showlegend=False,
    )
    return fig


def plotly_feature_comparison(df: pd.DataFrame) -> go.Figure:
    """Grouped bar of mean price for each binary feature flag (with/without)."""
    features = [c for c in ["balcony", "parking", "llm_furnished"] if c in df.columns]
    if not features:
        return _empty_chart("No feature columns available")
    rows = []
    for f in features:
        means = df.groupby(f)["price"].mean()
        rows.append({"feature": f, "with": float(means.get(1, np.nan)),
                     "without": float(means.get(0, np.nan))})
    fig = go.Figure()
    for label, color in [("without", PALETTE_NEUTRAL), ("with", PALETTE_PRIMARY)]:
        fig.add_bar(
            x=[r["feature"] for r in rows],
            y=[r[label] for r in rows],
            name=label,
            marker_color=color,
            text=[f"CHF {r[label]:,.0f}" if not np.isnan(r[label]) else ""
                  for r in rows],
            textposition="outside",
        )
    fig.update_layout(
        title="Mean rent: feature present vs absent",
        barmode="group",
        yaxis_title="Average rent (CHF/month)",
        xaxis_title="", template=PLOTLY_TEMPLATE, height=420,
    )
    return fig


def _empty_chart(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color=PALETTE_NEUTRAL))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=320,
                      xaxis_visible=False, yaxis_visible=False)
    return fig


# Centroid lookup table for the geographic map. Coordinates are approximate
# town/district centres — accurate enough for a bubble overlay at zoom 8.5.
# Source: public OSM coordinates rounded to 4 decimals. We intentionally
# hard-code these rather than fetch them at runtime, to keep the dashboard
# offline-capable and avoid an additional API dependency.
_DEFAULT_ZURICH_CENTROIDS: dict[str, tuple[float, float]] = {
    "Zürich": (47.3769, 8.5417),
    "Zurich": (47.3769, 8.5417),
    "Winterthur": (47.5022, 8.7386),
    "Uster": (47.3478, 8.7193),
    "Dübendorf": (47.3978, 8.6181),
    "Dietikon": (47.4015, 8.4002),
    "Wetzikon": (47.3247, 8.7975),
    "Wädenswil": (47.2299, 8.6705),
    "Horgen": (47.2598, 8.5959),
    "Kloten": (47.4525, 8.5874),
    "Bülach": (47.5208, 8.5400),
    "Adliswil": (47.3110, 8.5247),
    "Opfikon": (47.4292, 8.5772),
    "Schlieren": (47.3958, 8.4475),
    "Regensdorf": (47.4344, 8.4710),
    "Volketswil": (47.3920, 8.6960),
    "Illnau-Effretikon": (47.4290, 8.7170),
    "Thalwil": (47.2917, 8.5645),
    "Meilen": (47.2702, 8.6471),
    "Männedorf": (47.2536, 8.6920),
    "Stäfa": (47.2421, 8.7240),
    "Küsnacht": (47.3170, 8.5854),
    "Zollikon": (47.3402, 8.5775),
    "Erlenbach": (47.3030, 8.5933),
    "Herrliberg": (47.2867, 8.6188),
    "Rüschlikon": (47.3056, 8.5523),
    "Kilchberg": (47.3242, 8.5410),
    "Oberrieden": (47.2737, 8.5810),
    "Au": (47.2353, 8.6440),
    "Richterswil": (47.2069, 8.6995),
    "Fehraltorf": (47.3850, 8.7530),
    "Langwiesen": (47.6900, 8.6635),
    "Glattfelden": (47.5598, 8.5005),
    "Pfäffikon": (47.3640, 8.7903),
    "Hinwil": (47.2972, 8.8443),
    "Bauma": (47.3658, 8.8794),
    "Embrach": (47.5078, 8.5947),
    "Rümlang": (47.4519, 8.5325),
    "Wallisellen": (47.4128, 8.5961),
    "Niederhasli": (47.4806, 8.4878),
    "Zumikon": (47.3306, 8.6190),
    "Stallikon": (47.3197, 8.4920),
    "Affoltern am Albis": (47.2780, 8.4517),
    "Hedingen": (47.2950, 8.4480),
    "Hausen am Albis": (47.2440, 8.5277),
    "Bonstetten": (47.3110, 8.4670),
    "Mettmenstetten": (47.2440, 8.4670),
    "Knonau": (47.2237, 8.4622),
    "Maur": (47.3473, 8.6695),
    "Egg": (47.3013, 8.6960),
    "Greifensee": (47.3673, 8.6760),
    "Schwerzenbach": (47.3845, 8.6582),
    "Fällanden": (47.3756, 8.6403),
    "Bassersdorf": (47.4429, 8.6266),
    "Nürensdorf": (47.4453, 8.6533),
    "Effretikon": (47.4226, 8.6939),
    "Brüttisellen": (47.4192, 8.6376),
    "Wangen-Brüttisellen": (47.4192, 8.6376),
    "Dietlikon": (47.4205, 8.6175),
    "Wilen": (47.4830, 8.7300),
    "Andelfingen": (47.5947, 8.6803),
    "Stein am Rhein": (47.6589, 8.8550),
    "Henggart": (47.5660, 8.6815),
    "Rafz": (47.6028, 8.5430),
    "Wallisellen ZH": (47.4128, 8.5961),
    "Birmensdorf": (47.3543, 8.4424),
    "Aesch": (47.3084, 8.4350),
    "Männedorf ZH": (47.2536, 8.6920),
    "Oberglatt": (47.4747, 8.5188),
    "Hochfelden": (47.5104, 8.5223),
    "Höri": (47.5012, 8.5095),
    "Buchs ZH": (47.4475, 8.4400),
    "Otelfingen": (47.4570, 8.3940),
    "Boppelsen": (47.4795, 8.4280),
    "Steinmaur": (47.4956, 8.4660),
    "Lufingen": (47.4895, 8.6079),
    "Pfungen": (47.5247, 8.6539),
    "Neftenbach": (47.5346, 8.6633),
    "Seuzach": (47.5410, 8.7400),
    "Elsau": (47.5234, 8.7918),
    "Wila": (47.4204, 8.8434),
    "Turbenthal": (47.4380, 8.8530),
    "Sternenberg": (47.3958, 8.8970),
    "Bachenbülach": (47.5093, 8.5470),
}
