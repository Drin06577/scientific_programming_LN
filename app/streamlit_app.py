"""Swiss Housing Analytics — Streamlit dashboard.

Run:  streamlit run app/streamlit_app.py

A multi-tab analytics platform built on the SciPro pipeline:
  • Scrape (Flatfox) → Clean (regex) → Enrich (OpenAI) → Validate → SQLite → Stats → Viz

Architecture choices:
  - The dashboard NEVER recomputes anything itself. It pulls from
    pipeline.run_pipeline() and renders. This keeps data logic in one place.
  - All charts are Plotly so the user can hover, pan, zoom.
  - Filters live in the sidebar and apply to a single shared DataFrame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline  # noqa: E402
from src.database import HousingDatabase  # noqa: E402
from src.visualization import (  # noqa: E402
    plotly_city_bar,
    plotly_city_boxplot_chf_per_m2,
    plotly_scatter_size_price,
    plotly_rooms_distribution,
    plotly_chf_per_m2_histogram,
    plotly_correlation_heatmap,
    plotly_geographic_map,
    plotly_scatter_matrix,
    plotly_price_categories,
    plotly_feature_comparison,
)


st.set_page_config(
    page_title="Swiss Housing Analytics",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded",
)

# Minimal global CSS — keeps the rest of the dashboard configurable via
# Streamlit theme. Just enough to tighten spacing + lift the KPI cards.
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: #F8FAFC;
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    div[data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1E293B !important;
    }
    h1, h2, h3 {
        color: #0F172A;
    }
    .insight-card {
        background: #F8FAFC;
        border-left: 4px solid #2E5C8A;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .insight-card.high {border-left-color: #16A34A;}
    .insight-card.medium {border-left-color: #2E5C8A;}
    .insight-card.low {border-left-color: #94A3B8;}
    .insight-headline {
        font-weight: 600;
        color: #0F172A;
        font-size: 1.05rem;
        margin-bottom: 6px;
    }
    .insight-detail {color: #475569; font-size: 0.92rem; line-height: 1.5;}
    .insight-metric {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 8px;
    }
    .insight-metric b {color: #0F172A;}
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .stat-card .label {
        font-weight: 600;
        color: #0F172A;
        font-size: 1.0rem;
    }
    .stat-card .meta {
        font-size: 0.85rem;
        color: #64748B;
        margin: 4px 0;
    }
    .stat-card .sig {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .stat-card .sig.yes {background:#DCFCE7; color:#15803D;}
    .stat-card .sig.no  {background:#F1F5F9; color:#475569;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Running pipeline (scrape → clean → LLM → validate → stats)…")
def load_data(use_llm: bool, fetch_html: bool):
    return run_pipeline(use_llm=use_llm, save_figures=False,
                        max_listings=200, fetch_html=fetch_html,
                        apply_quality=True)


def apply_filters(df: pd.DataFrame, *,
                  cities: list[str], rooms_range: tuple[float, float],
                  price_range: tuple[float, float],
                  balcony_only: bool, parking_only: bool,
                  search: str) -> pd.DataFrame:
    """Apply all sidebar filters in one pass — easier to reason about than
    Streamlit's stateful filters, and the filtered view feeds every tab."""
    mask = pd.Series(True, index=df.index)
    if cities:
        mask &= df["city"].isin(cities)
    if rooms_range:
        mask &= df["rooms"].between(rooms_range[0], rooms_range[1])
    if price_range:
        mask &= df["price"].between(price_range[0], price_range[1])
    if balcony_only:
        mask &= df["balcony"] == 1
    if parking_only:
        mask &= df["parking"] == 1
    if search:
        q = search.lower().strip()
        text_cols = ["title", "city", "description"]
        text_mask = pd.Series(False, index=df.index)
        for c in text_cols:
            if c in df.columns:
                text_mask |= df[c].fillna("").astype(str).str.lower().str.contains(q)
        mask &= text_mask
    return df.loc[mask].reset_index(drop=True)


def fmt_chf(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"CHF {value:,.0f}"


def render_kpi_row(summary: dict) -> None:
    cols = st.columns(6)
    cols[0].metric("Listings", summary.get("n_listings", 0))
    cols[1].metric("Cities", summary.get("n_cities", 0))
    cols[2].metric("Mean rent", fmt_chf(summary.get("mean_price")))
    cols[3].metric("Median rent", fmt_chf(summary.get("median_price")))
    cols[4].metric("Mean CHF/m²", f"{summary.get('mean_chf_per_m2', 0):.0f}")
    cols[5].metric("Mean size", f"{summary.get('mean_size', 0):.0f} m²")

    cols2 = st.columns(6)
    cols2[0].metric("Std (price)", fmt_chf(summary.get("std_price")))
    cols2[1].metric("IQR (price)", fmt_chf(summary.get("iqr_price")))
    cols2[2].metric("Min rent", fmt_chf(summary.get("min_price")))
    cols2[3].metric("Max rent", fmt_chf(summary.get("max_price")))
    cols2[4].metric("Mean rooms", f"{summary.get('mean_rooms', 0):.1f}")
    cols2[5].metric("Median rooms", f"{summary.get('median_rooms', 0):.1f}")


def render_llm_badge(llm: dict) -> None:
    mode = llm.get("mode", "n/a")
    co = llm.get("calls_openai", 0)
    cf = llm.get("calls_fallback", 0)
    cc = llm.get("calls_cache", 0)
    if mode == "openai":
        st.success(f"🤖 LLM enrichment: OpenAI (gpt-4o-mini) — "
                   f"{co} new calls, {cc} cache hits, {cf} regex fallback", icon="✅")
    elif mode == "regex":
        st.warning(f"🤖 LLM enrichment: regex fallback (no OpenAI key) — {cf} calls", icon="⚠️")
    else:
        st.info(f"🤖 LLM mode: {mode} (cache {cc} / openai {co} / regex {cf})")


def render_insight_card(ins: dict) -> None:
    conf = ins.get("confidence", "medium").lower()
    st.markdown(f"""
    <div class="insight-card {conf}">
        <div class="insight-headline">{ins['headline']}</div>
        <div class="insight-detail">{ins['detail']}</div>
        <div class="insight-metric"><b>{ins['metric']}</b>: {ins['metric_value']}
        &nbsp;·&nbsp; confidence: <b>{conf}</b></div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(t: dict) -> None:
    sig_class = "yes" if t.get("significant") else "no"
    sig_label = "✓ significant" if t.get("significant") else "not significant"
    es_label = t.get("effect_size_label") or ""
    st.markdown(f"""
    <div class="stat-card">
      <div>
        <span class="label">{t['label']}</span>
        &nbsp; <span class="sig {sig_class}">{sig_label}</span>
      </div>
      <div class="meta">
        statistic = <b>{t['statistic']:.3f}</b> &nbsp;·&nbsp;
        p-value = <b>{t['p_value']:.4f}</b> &nbsp;·&nbsp;
        n = <b>{t['n']}</b>
        {("&nbsp;·&nbsp; " + es_label) if es_label else ""}
      </div>
      <div class="meta"><b>Interpretation:</b> {t['interpretation']}</div>
      {f'<div class="meta"><b>Why this test:</b> {t["why"]}</div>' if t.get("why") else ""}
      {f'<div class="meta"><b>Assumptions:</b> {t["assumptions"]}</div>' if t.get("assumptions") else ""}
    </div>
    """, unsafe_allow_html=True)


def build_listings_table(df: pd.DataFrame) -> pd.DataFrame:
    """Shape the listings DataFrame for display — clickable links, ordered cols."""
    show = df.copy()
    show["price"] = show["price"].round(0).astype("Int64")
    show["size"] = show["size"].round(1)
    if "chf_per_m2" in show.columns:
        show["chf_per_m2"] = show["chf_per_m2"].round(1)
    cols_order = [c for c in ["title", "city", "zip_code", "price", "size", "rooms",
                              "chf_per_m2", "balcony", "parking", "llm_furnished",
                              "price_category", "listing_url"]
                  if c in show.columns]
    return show[cols_order]


def main() -> None:
    st.title("🏠 Swiss Housing Analytics")
    st.caption(
        "Flatfox scraping → regex cleaning → OpenAI enrichment → "
        "data-quality validation → SQLite → statistical tests → Plotly dashboard. "
        "Canton Zurich, full-flat rentals only."
    )

    # ----- Sidebar -----------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Pipeline settings")
        use_llm = st.toggle("Use OpenAI enrichment", value=True,
                            help="Off = regex fallback only.")
        fetch_html = st.toggle("Re-fetch detail pages", value=False,
                               help="On = re-scrape descriptions (slow). "
                                    "Off = reuse cached data.")
        if st.button("Re-run pipeline", width="stretch"):
            load_data.clear()

    result = load_data(use_llm=use_llm, fetch_html=fetch_html)
    df_full: pd.DataFrame = result["df"]
    quality = result["quality_report"]
    summary = result["extended_summary"]

    with st.sidebar:
        st.markdown("---")
        st.header("🔍 Filters")
        cities = st.multiselect("City",
            options=sorted(df_full["city"].dropna().unique()),
            default=[],
            help="Filter listings to specific Zurich-canton cities.")
        rooms_min, rooms_max = float(df_full["rooms"].min()), float(df_full["rooms"].max())
        rooms_range = st.slider("Rooms",
            min_value=rooms_min, max_value=rooms_max,
            value=(rooms_min, rooms_max), step=0.5)
        price_min, price_max = float(df_full["price"].min()), float(df_full["price"].max())
        price_range = st.slider("Rent (CHF/month)",
            min_value=price_min, max_value=price_max,
            value=(price_min, price_max), step=100.0)
        balcony_only = st.checkbox("Balcony only", value=False)
        parking_only = st.checkbox("Parking only", value=False)
        search = st.text_input("Search (title / city / description)", value="",
                               placeholder="e.g. Zürich, lakeside, attic …")

    df = apply_filters(df_full, cities=cities, rooms_range=rooms_range,
                       price_range=price_range, balcony_only=balcony_only,
                       parking_only=parking_only, search=search)

    if df.empty:
        st.warning("No listings match the current filters. Adjust the sidebar.")
        return

    # KPIs use the filtered subset so cards react to filters live.
    from src.data_quality import extended_summary as _es
    filtered_summary = _es(df)
    render_llm_badge(result["llm"])
    st.markdown("### 📊 Headline metrics")
    render_kpi_row(filtered_summary)
    st.caption(f"Showing **{len(df)}** of {len(df_full)} cleaned listings. "
               f"Pipeline drew {result['raw_count']} raw listings from Flatfox.")

    # ----- Tabs --------------------------------------------------------
    tabs = st.tabs([
        "📈 Overview",
        "🗺️ Geography",
        "🧪 Statistical tests",
        "💡 Key insights",
        "🗄️ SQL explorer",
        "📋 Listings",
        "🧹 Data quality",
    ])

    # === Overview ======================================================
    with tabs[0]:
        st.subheader("Price drivers at a glance")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plotly_scatter_size_price(df), width="stretch")
        with c2:
            st.plotly_chart(plotly_chf_per_m2_histogram(df), width="stretch")

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(plotly_rooms_distribution(df), width="stretch")
        with c4:
            st.plotly_chart(plotly_price_categories(df), width="stretch")

        st.markdown("#### Feature premia — does each amenity affect rent?")
        st.plotly_chart(plotly_feature_comparison(df), width="stretch")

        st.markdown("#### Numeric feature correlations")
        c5, c6 = st.columns([3, 4])
        with c5:
            st.plotly_chart(plotly_correlation_heatmap(df), width="stretch")
        with c6:
            st.plotly_chart(plotly_scatter_matrix(df), width="stretch")

    # === Geography =====================================================
    with tabs[1]:
        st.subheader("Where the money is in Canton Zurich")
        st.plotly_chart(plotly_geographic_map(df), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plotly_city_bar(df, top_n=15, min_listings=2),
                            width="stretch")
        with c2:
            st.plotly_chart(plotly_city_boxplot_chf_per_m2(df, top_n=8, min_listings=4),
                            width="stretch")

    # === Statistical tests ============================================
    with tabs[2]:
        st.subheader("Statistical tests with effect sizes")
        st.markdown(
            "Each card shows the test statistic, p-value, effect size, "
            "and a plain-language interpretation. We pair parametric tests "
            "(Pearson, Welch t-test) with non-parametric alternatives "
            "(Spearman, Mann-Whitney) so the results survive any violation "
            "of the normality assumption."
        )
        st.info(
            "**How to read the p-value.** It's the probability of seeing a result "
            "this extreme if the null hypothesis (no relationship) were true. "
            "A value below 0.05 is conventionally called 'statistically significant', "
            "but a tiny p-value with a tiny effect size is rarely practically meaningful. "
            "Always read the **effect size** alongside the p-value.",
            icon="📘",
        )
        # Run the tests on the FILTERED subset so this tab is interactive.
        from src.statistics import run_all_tests
        tests = [t.as_dict() for t in run_all_tests(df)]
        for t in tests:
            render_stat_card(t)

    # === Insights ======================================================
    with tabs[3]:
        st.subheader("Key insights")
        st.caption("Auto-generated from the cleaned dataset — re-runs when you change filters above.")
        from src.insights import generate
        insights = [i.as_dict() for i in generate(df)]
        # Two-column layout so the insights tile nicely.
        col_a, col_b = st.columns(2)
        for i, ins in enumerate(insights):
            with (col_a if i % 2 == 0 else col_b):
                render_insight_card(ins)

    # === SQL explorer =================================================
    with tabs[4]:
        st.subheader("Database schema")
        st.dataframe(result["schema"], width="stretch")

        st.markdown("### Canned SQL queries")
        st.caption("Each query is computed against the SQLite `apartments` table built by the pipeline.")
        sql = result["sql"]

        for label, hint in HousingDatabase.QUERY_CATALOG:
            with st.expander(f"📜 {label}", expanded=False):
                st.caption(hint)
                # Map labels to result keys (kept in sync with QUERY_CATALOG).
                key_map = {
                    "Average price by room count": "avg_price_by_rooms",
                    "Average price by city (top by mean rent)": "avg_price_by_location",
                    "Balcony vs no-balcony pricing": "avg_price_by_balcony",
                    "Top 10 most expensive listings": "top_expensive",
                    "Top 10 cheapest listings": "top_cheapest",
                    "Price tiers via NTILE window function": "price_distribution",
                    "City ranking + premium vs market average": "city_ranking",
                    "Side-by-side feature premium (balcony / parking / furnished)": "feature_premium",
                    "Rolling 10-listing average price by size": None,
                }
                key = key_map.get(label)
                if key and key in sql:
                    st.dataframe(sql[key], width="stretch")
                elif label.startswith("Rolling"):
                    with HousingDatabase(result["db_path"]) as db:
                        st.dataframe(db.running_avg_price_by_size().head(50), width="stretch")

    # === Listings ======================================================
    with tabs[5]:
        st.subheader(f"Apartments — {len(df)} listings")
        st.caption("Sorted by CHF/m² (highest first). Click the link in the last column to open the listing on Flatfox.")
        listings = build_listings_table(df.sort_values("chf_per_m2", ascending=False))
        st.dataframe(
            listings,
            width="stretch",
            hide_index=True,
            column_config={
                "title": st.column_config.TextColumn("Title", width="large"),
                "city": st.column_config.TextColumn("City"),
                "zip_code": st.column_config.TextColumn("ZIP"),
                "price": st.column_config.NumberColumn("Rent (CHF)", format="CHF %d"),
                "size": st.column_config.NumberColumn("Size (m²)", format="%.1f"),
                "rooms": st.column_config.NumberColumn("Rooms", format="%.1f"),
                "chf_per_m2": st.column_config.NumberColumn("CHF/m²", format="%.1f"),
                "balcony": st.column_config.CheckboxColumn("🌿 Balcony"),
                "parking": st.column_config.CheckboxColumn("🚗 Parking"),
                "llm_furnished": st.column_config.CheckboxColumn("🛋 Furnished"),
                "price_category": st.column_config.TextColumn("Tier"),
                "listing_url": st.column_config.LinkColumn(
                    "🔗 Open listing", display_text="View on Flatfox"
                ),
            },
        )
        st.download_button(
            "⬇ Download filtered CSV",
            data=listings.to_csv(index=False).encode("utf-8"),
            file_name="filtered_apartments.csv",
            mime="text/csv",
        )

    # === Data quality ==================================================
    with tabs[6]:
        st.subheader("Data quality report")
        st.caption(
            "The validator in `src/data_quality.py` runs before the statistical tests "
            "and the dashboard. It enforces sensible plausibility bounds (rent 500–15 000 CHF, "
            "size 15–400 m², CHF/m² 10–200) and removes duplicates."
        )
        q = quality
        c = st.columns(5)
        c[0].metric("Raw collected", q["n_input"])
        c[1].metric("After validation", q["n_output"])
        c[2].metric("Retention rate", f"{q['retention_rate']*100:.1f}%")
        c[3].metric("Duplicates removed", q["n_duplicates"])
        c[4].metric("Statistical outliers", q["n_statistical_outliers"])

        c2 = st.columns(4)
        c2[0].metric("Impossible price", q["n_impossible_price"])
        c2[1].metric("Impossible size", q["n_impossible_size"])
        c2[2].metric("Impossible rooms", q["n_impossible_rooms"])
        c2[3].metric("CHF/m² out of bounds", q["n_chf_per_m2_outliers"])

        dropped = result["dropped_rows"]
        if isinstance(dropped, pd.DataFrame) and not dropped.empty:
            st.markdown("#### Dropped rows (full audit trail)")
            st.dataframe(dropped, width="stretch")
        else:
            st.success("No rows were dropped — the raw collector already enforces strict filters.", icon="✅")

        st.markdown("#### Missing values per column")
        miss = (df_full.isna().sum() / len(df_full) * 100).round(2)
        miss_df = miss[miss > 0].sort_values(ascending=False).reset_index()
        miss_df.columns = ["column", "missing_pct"]
        if miss_df.empty:
            st.success("No missing values across the cleaned dataset.", icon="✅")
        else:
            st.dataframe(miss_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
