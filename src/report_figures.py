"""Report-ready static figures for the university write-up.

Each function builds one plot, applies the shared report theme, saves a
high-resolution PNG into reports/figures/, and prints a short plain-language
interpretation to the console. Functions are tolerant of missing optional
columns so a stripped-down dataset still produces the headline plots.

Run from the repo root:
    python -m src.report_figures

The figures are generated from data/processed/apartments.csv if present,
otherwise the full pipeline is invoked once to produce it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

REPORT_FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
PROCESSED_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "apartments.csv"

# Shared visual settings — light theme, generous fonts for a printed A4 report.
REPORT_THEME = {
    "figure.figsize": (10.5, 6.5),
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#E5E7EB",
    "grid.linewidth": 0.7,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 14,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "font.family": "DejaVu Sans",
}

# Brand-consistent palette reused across all report plots.
PRIMARY = "#2E5C8A"
ACCENT = "#D97706"
SUCCESS = "#16A34A"
DANGER = "#DC2626"
NEUTRAL = "#6B7280"
WITH_COLOR = PRIMARY
WITHOUT_COLOR = NEUTRAL


def _ensure_dir() -> Path:
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_FIG_DIR


def _ensure_chf_per_m2(df: pd.DataFrame) -> pd.DataFrame:
    if "chf_per_m2" not in df.columns and {"price", "size"} <= set(df.columns):
        df = df.copy()
        df["chf_per_m2"] = (df["price"] / df["size"]).round(2)
    return df


def _titled(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    """Set a two-line title (big bold headline + small italic subtitle).

    Using ax.set_title with an explicit y keeps the title above the subtitle
    and avoids matplotlib's tight_layout silently collapsing the subtitle
    onto the plot area.
    """
    ax.set_title(title, y=1.08, pad=10)
    if subtitle:
        ax.text(
            0.5, 1.015, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=11, color="#555555", style="italic",
        )


# ---------------------------------------------------------------------------
# 01. Size vs price scatter with regression line + Pearson annotation
# ---------------------------------------------------------------------------

def plot_size_vs_price(df: pd.DataFrame, out_dir: Path) -> Path | None:
    sub = df.dropna(subset=["size", "price"]).copy()
    sub = _ensure_chf_per_m2(sub)
    if len(sub) < 5:
        print("[plot 01] skipped — too few rows with size + price")
        return None

    r, p = stats.pearsonr(sub["size"], sub["price"])
    slope, intercept = np.polyfit(sub["size"], sub["price"], 1)
    xs = np.linspace(sub["size"].min(), sub["size"].max(), 100)

    fig, ax = plt.subplots()
    sc = ax.scatter(
        sub["size"], sub["price"],
        c=sub["chf_per_m2"], cmap="viridis",
        s=55, alpha=0.82, edgecolor="white", linewidth=0.6,
    )
    ax.plot(xs, intercept + slope * xs, color=DANGER, linewidth=2.4,
            linestyle="--", label="OLS regression")
    _titled(ax, "Larger flats usually have higher monthly rent",
            f"Zurich-canton listings (n={len(sub)})")
    ax.set_xlabel("Living space (m²)")
    ax.set_ylabel("Monthly rent (CHF)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("CHF per m²", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    sig = "significant" if p < 0.05 else "not statistically significant"
    annotation = (
        f"Pearson r = {r:.2f}  ({sig}, p = {p:.1e})\n"
        f"Each extra m² adds ~CHF {slope:.0f}/month on average."
    )
    ax.text(
        0.03, 0.97, annotation, transform=ax.transAxes,
        ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=NEUTRAL, linewidth=0.8, alpha=0.95),
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    out = out_dir / "01_size_vs_price.png"
    fig.savefig(out)
    plt.close(fig)

    direction = "positive" if r > 0 else "negative"
    print(f"[plot 01] {out.name}: {direction} correlation r={r:.2f}; "
          f"each extra m² adds about CHF {slope:.0f}/month.")
    return out


# ---------------------------------------------------------------------------
# 02. Rent by number of rooms — boxplot + raw points, ≥5-listing categories
# ---------------------------------------------------------------------------

def plot_rent_by_rooms(df: pd.DataFrame, out_dir: Path, min_n: int = 5) -> Path | None:
    sub = df.dropna(subset=["rooms", "price"]).copy()
    if sub.empty:
        print("[plot 02] skipped — no room/price data")
        return None

    sub["rooms"] = sub["rooms"].astype(float).round(1)
    counts = sub["rooms"].value_counts().sort_index()
    keep = counts[counts >= min_n].index.tolist()
    sub = sub[sub["rooms"].isin(keep)]
    excluded = int((counts < min_n).sum())
    if sub.empty:
        print(f"[plot 02] skipped — no room category has ≥{min_n} listings")
        return None

    order = sorted(sub["rooms"].unique())
    groups = [sub.loc[sub["rooms"] == r, "price"].values for r in order]
    labels = [f"{r:g} rooms\n(n = {len(sub.loc[sub['rooms']==r])})" for r in order]

    fig, ax = plt.subplots()
    bp = ax.boxplot(
        groups, positions=range(len(order)), widths=0.55, patch_artist=True,
        medianprops=dict(color=DANGER, linewidth=2),
        boxprops=dict(facecolor="#DBEAFE", edgecolor=PRIMARY, linewidth=1.2),
        whiskerprops=dict(color=PRIMARY, linewidth=1.0),
        capprops=dict(color=PRIMARY, linewidth=1.0),
        flierprops=dict(marker="", markersize=0),  # outliers shown via swarm below
    )
    # Raw points with horizontal jitter so they don't overlap exactly.
    rng = np.random.default_rng(seed=0)
    for i, vals in enumerate(groups):
        if len(vals) == 0:
            continue
        x_jitter = rng.normal(loc=i, scale=0.06, size=len(vals))
        ax.scatter(x_jitter, vals, s=20, color=PRIMARY, alpha=0.45,
                   edgecolor="white", linewidth=0.4, zorder=3)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of rooms (sample size shown)")
    ax.set_ylabel("Monthly rent (CHF)")
    excl_note = f" — {excluded} small categories (< {min_n} listings) excluded" if excluded else ""
    _titled(ax, "Monthly rent by number of rooms",
            f"Boxplots show the middle 50% of rents{excl_note}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.tight_layout()

    out = out_dir / "02_rent_by_rooms.png"
    fig.savefig(out)
    plt.close(fig)

    medians = {r: float(np.median(g)) for r, g in zip(order, groups) if len(g)}
    cheapest = min(medians, key=medians.get)
    priciest = max(medians, key=medians.get)
    print(f"[plot 02] {out.name}: cheapest median is {cheapest:g}-room flats at "
          f"CHF {medians[cheapest]:,.0f}; priciest is {priciest:g}-room flats at "
          f"CHF {medians[priciest]:,.0f}. {excluded} small room categories excluded.")
    return out


# ---------------------------------------------------------------------------
# 03. Rent-range histogram with fixed bins
# ---------------------------------------------------------------------------

_RENT_BINS = [0, 2000, 3000, 4000, 5000, 6000, np.inf]
_RENT_LABELS = ["0–2 000", "2 000–3 000", "3 000–4 000",
                "4 000–5 000", "5 000–6 000", "6 000+"]


def plot_rent_ranges(df: pd.DataFrame, out_dir: Path) -> Path | None:
    sub = df.dropna(subset=["price"])
    if sub.empty:
        print("[plot 03] skipped — no price data")
        return None

    counts = pd.cut(sub["price"], bins=_RENT_BINS, labels=_RENT_LABELS,
                    include_lowest=True, right=False).value_counts()
    counts = counts.reindex(_RENT_LABELS).fillna(0).astype(int)

    fig, ax = plt.subplots()
    bars = ax.bar(
        counts.index, counts.values,
        color=PRIMARY, edgecolor="white", linewidth=1.2, width=0.78,
    )
    for bar, value in zip(bars, counts.values):
        if value == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(value)), ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#1F2937")

    _titled(ax, "How many listings fall into each rent range?",
            f"Zurich-canton listings (n = {int(counts.sum())})")
    ax.set_xlabel("Monthly rent range (CHF)")
    ax.set_ylabel("Number of listings")
    ax.set_ylim(0, counts.max() * 1.18 if counts.max() > 0 else 1)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    out = out_dir / "03_rent_ranges.png"
    fig.savefig(out)
    plt.close(fig)

    top_bucket = counts.idxmax()
    print(f"[plot 03] {out.name}: most common bucket is CHF {top_bucket} "
          f"({int(counts[top_bucket])} listings); full distribution = "
          f"{dict(counts)}.")
    return out


# ---------------------------------------------------------------------------
# 04. Feature impact — mean & median rent for with/without each feature
# ---------------------------------------------------------------------------

_FEATURE_CANDIDATES = [
    ("balcony", "Balcony"),
    ("parking", "Parking / garage"),
    ("llm_furnished", "Furnished"),
]


def _feature_is_plottable(series: pd.Series, label: str) -> bool:
    """Validate a feature column: at least 5 non-null, both 0/1 present,
    each side has at least 3 observations. Prints the diagnostic to console."""
    if series is None:
        return False
    s = pd.to_numeric(series, errors="coerce")
    s_nonnull = s.dropna()
    vc = s_nonnull.value_counts().to_dict()
    print(f"  [feature] {label}: value_counts={vc}, "
          f"non_null={len(s_nonnull)}/{len(s)}")
    if len(s_nonnull) < 5:
        print(f"    → skipped: too many missing values")
        return False
    if s_nonnull.nunique() < 2:
        print(f"    → skipped: only one unique value present")
        return False
    n1 = int((s_nonnull == 1).sum())
    n0 = int((s_nonnull == 0).sum())
    if n1 < 3 or n0 < 3:
        print(f"    → skipped: with={n1}, without={n0} — need ≥3 in each group")
        return False
    return True


def plot_feature_impact(df: pd.DataFrame, out_dir: Path) -> Path | None:
    print("[plot 04] validating optional feature columns…")
    valid: list[tuple[str, str]] = []
    for col, label in _FEATURE_CANDIDATES:
        if col not in df.columns:
            print(f"  [feature] {label}: column absent — skipped")
            continue
        if _feature_is_plottable(df[col], label):
            valid.append((col, label))

    if not valid:
        print("[plot 04] skipped — no usable feature columns")
        return None

    rows = []
    for col, label in valid:
        s = pd.to_numeric(df[col], errors="coerce")
        mask = s.notna()
        with_p = df.loc[mask & (s == 1), "price"].dropna()
        without_p = df.loc[mask & (s == 0), "price"].dropna()
        rows.append({
            "label": label,
            "with_mean": float(with_p.mean()),
            "with_median": float(with_p.median()),
            "with_n": int(len(with_p)),
            "without_mean": float(without_p.mean()),
            "without_median": float(without_p.median()),
            "without_n": int(len(without_p)),
        })

    fig, ax = plt.subplots()
    x = np.arange(len(rows))
    width = 0.36
    bars_without = ax.bar(x - width / 2,
                          [r["without_mean"] for r in rows],
                          width, color=WITHOUT_COLOR, label="Without",
                          edgecolor="white", linewidth=1.2)
    bars_with = ax.bar(x + width / 2,
                       [r["with_mean"] for r in rows],
                       width, color=WITH_COLOR, label="With",
                       edgecolor="white", linewidth=1.2)

    # Sample-size labels under the x-axis tick.
    labels = [f"{r['label']}\n(without n={r['without_n']} · with n={r['with_n']})"
              for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    # Bar-top: mean (bold) and median (smaller, beneath the mean).
    for r, b_w, b_o in zip(rows, bars_with, bars_without):
        ax.text(b_o.get_x() + b_o.get_width() / 2, b_o.get_height() + 50,
                f"CHF {r['without_mean']:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(b_o.get_x() + b_o.get_width() / 2, b_o.get_height() - 200,
                f"median {r['without_median']:,.0f}",
                ha="center", va="top", fontsize=9, color="white")
        ax.text(b_w.get_x() + b_w.get_width() / 2, b_w.get_height() + 50,
                f"CHF {r['with_mean']:,.0f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(b_w.get_x() + b_w.get_width() / 2, b_w.get_height() - 200,
                f"median {r['with_median']:,.0f}",
                ha="center", va="top", fontsize=9, color="white")

    _titled(ax, "Do specific features affect monthly rent?",
            "Mean rent per bar; median annotated inside each bar")
    ax.set_ylabel("Monthly rent (CHF)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(handles=[Patch(color=WITHOUT_COLOR, label="Without feature"),
                       Patch(color=WITH_COLOR, label="With feature")],
              loc="upper left", frameon=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()

    out = out_dir / "04_feature_impact.png"
    fig.savefig(out)
    plt.close(fig)

    diffs = [(r["label"], r["with_mean"] - r["without_mean"]) for r in rows]
    diffs_str = "; ".join(f"{lab} +CHF {d:,.0f}" if d >= 0 else f"{lab} −CHF {abs(d):,.0f}"
                          for lab, d in diffs)
    print(f"[plot 04] {out.name}: mean rent difference (with − without): {diffs_str}.")
    return out


# ---------------------------------------------------------------------------
# 05. Correlation matrix — numeric features only
# ---------------------------------------------------------------------------

def plot_correlation_matrix(df: pd.DataFrame, out_dir: Path) -> Path | None:
    df = _ensure_chf_per_m2(df)
    candidate_cols = ["price", "size", "rooms", "chf_per_m2"]
    cols: list[str] = []
    for c in candidate_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.dropna().empty or s.dropna().nunique() < 2:
            continue
        cols.append(c)

    if len(cols) < 2:
        print("[plot 05] skipped — fewer than 2 numeric columns")
        return None

    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr().round(2)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.55 else "#1F2937",
                    fontsize=13, fontweight="bold")
    _titled(ax, "Correlation between numeric housing features",
            "Pearson r — values close to ±1 mean a strong linear link")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    fig.text(0.5, 0.01,
             "Note: correlation does not imply causation.",
             ha="center", fontsize=11, color="#555555", style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = out_dir / "05_correlation_matrix.png"
    fig.savefig(out)
    plt.close(fig)

    if "price" in cols and "size" in cols:
        r_size = corr.loc["price", "size"]
        print(f"[plot 05] {out.name}: strongest correlation in this dataset is "
              f"price↔size (r = {r_size:.2f}).")
    else:
        print(f"[plot 05] {out.name}: matrix saved.")
    return out


# ---------------------------------------------------------------------------
# 06. CHF/m² distribution histogram with mean + median lines
# ---------------------------------------------------------------------------

def plot_chf_per_m2_distribution(df: pd.DataFrame, out_dir: Path,
                                  cap_pct: float = 99.0) -> Path | None:
    df = _ensure_chf_per_m2(df)
    if "chf_per_m2" not in df.columns:
        print("[plot 06] skipped — chf_per_m2 column missing")
        return None
    series = df["chf_per_m2"].dropna()
    if series.empty:
        print("[plot 06] skipped — chf_per_m2 column empty")
        return None

    # Cap visualization to the 99th percentile so the histogram is readable;
    # mention how many points were excluded purely from the *plot* (still in
    # the statistics).
    upper = float(np.percentile(series, cap_pct))
    excluded = int((series > upper).sum())
    plot_series = series[series <= upper]
    median = float(series.median())
    mean = float(series.mean())

    fig, ax = plt.subplots()
    ax.hist(plot_series, bins=28, color=PRIMARY, edgecolor="white", linewidth=1.0)
    ax.axvline(median, color=DANGER, linewidth=2.2, linestyle="--",
               label=f"Median = CHF {median:,.1f}/m²")
    ax.axvline(mean, color=ACCENT, linewidth=2.2, linestyle=":",
               label=f"Mean = CHF {mean:,.1f}/m²")
    note = (f"Plot capped at the {cap_pct:.0f}th percentile "
            f"({excluded} outlier{'s' if excluded != 1 else ''} excluded from view; "
            "statistics use the full sample)")
    _titled(ax, "Distribution of monthly rent per square metre", note)
    ax.set_xlabel("Monthly rent per m² (CHF)")
    ax.set_ylabel("Number of listings")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    out = out_dir / "06_chf_per_m2_distribution.png"
    fig.savefig(out)
    plt.close(fig)

    skew = "right-skewed" if mean > median else "left-skewed" if mean < median else "symmetric"
    print(f"[plot 06] {out.name}: median CHF {median:.1f}/m², "
          f"mean CHF {mean:.1f}/m² ({skew}); "
          f"{excluded} extreme points hidden in the cap.")
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(df: pd.DataFrame, out_dir: Path | None = None) -> list[Path]:
    """Build all six report figures from a cleaned dataframe."""
    plt.rcParams.update(REPORT_THEME)
    out = out_dir or _ensure_dir()
    print(f"\n=== Generating report figures into {out} ===")
    saved: list[Path] = []
    for fn in (
        plot_size_vs_price,
        plot_rent_by_rooms,
        plot_rent_ranges,
        plot_feature_impact,
        plot_correlation_matrix,
        plot_chf_per_m2_distribution,
    ):
        path = fn(df, out)
        if path is not None:
            saved.append(path)
    print(f"\n=== Done: {len(saved)} figures saved ===")
    for p in saved:
        print(f"  • {p}")
    return saved


def _load_dataset() -> pd.DataFrame:
    if PROCESSED_CSV.exists():
        print(f"[report] loading cached dataset {PROCESSED_CSV}")
        return pd.read_csv(PROCESSED_CSV)
    print("[report] no cached CSV — running full pipeline once to produce one")
    from .pipeline import run_pipeline
    result = run_pipeline(use_llm=False, save_figures=False, max_listings=200)
    return result["df"]


if __name__ == "__main__":
    df = _load_dataset()
    generate_all(df)
