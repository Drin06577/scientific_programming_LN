"""Minimal Streamlit dashboard for the Swiss housing analysis.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `from src...` when launched from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline  # noqa: E402
from src.visualization import (  # noqa: E402
    scatter_size_price,
    boxplot_rooms_price,
    barplot_avg_price_by_city,
)


@st.cache_data(show_spinner="Running pipeline (scrape → clean → LLM → SQLite → stats)…")
def load_data(use_llm: bool):
    return run_pipeline(use_llm=use_llm, save_figures=False, max_listings=200)


def main() -> None:
    st.set_page_config(page_title="Swiss Housing Analysis", layout="wide")
    st.title("🏠 Swiss Housing Price Analysis")
    st.caption(
        "Scraping → regex cleaning → OpenAI enrichment → SQLite → pandas → p-values → charts."
    )

    with st.sidebar:
        st.header("Settings")
        use_llm = st.toggle("Use OpenAI enrichment", value=True,
                            help="Turn off to use the regex fallback only.")
        if st.button("Re-run pipeline"):
            load_data.clear()

    result = load_data(use_llm=use_llm)
    df = result["df"]

    # -- LLM status badge --------------------------------------------
    llm = result["llm"]
    if llm["mode"] == "openai":
        st.success(
            f"🤖 LLM: **OpenAI** — {llm['calls_openai']} calls to gpt-4o-mini.",
            icon="✅",
        )
    elif llm["mode"] == "regex":
        st.warning(
            "🤖 LLM: **regex fallback** (no OPENAI_API_KEY in .env, or call failed). "
            f"{llm['calls_fallback']} fallback calls.",
            icon="⚠️",
        )
    else:
        st.info(f"🤖 LLM: {llm['mode']}")

    # -- Summary ------------------------------------------------------
    st.subheader("Summary")
    cols = st.columns(4)
    summary = result["summary"]
    cols[0].metric("Listings", summary["n_listings"])
    cols[1].metric("Cities", summary["n_cities"])
    cols[2].metric("Mean price (CHF)", f"{summary['mean_price']:.0f}")
    cols[3].metric("Mean size (m²)", f"{summary['mean_size']:.1f}")

    # -- Data ---------------------------------------------------------
    st.subheader("Cleaned data")
    st.dataframe(df, width="stretch")

    # -- SQL ----------------------------------------------------------
    st.subheader("SQL query results")
    sql = result["sql"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Average price by rooms**")
        st.dataframe(sql["avg_price_by_rooms"], width="stretch")
        st.markdown("**Average price by balcony flag**")
        st.dataframe(sql["avg_price_by_balcony"], width="stretch")
    with c2:
        st.markdown("**Average price by city**")
        st.dataframe(sql["avg_price_by_location"], width="stretch")
        st.markdown("**Top 5 most expensive**")
        st.dataframe(sql["top_expensive"], width="stretch")

    # -- Stats --------------------------------------------------------
    st.subheader("Statistical tests (α = 0.05)")
    for t in result["tests"]:
        sig = "✅ significant" if t["significant"] else "—"
        st.markdown(
            f"**{t['name']}** — statistic = `{t['statistic']:.3f}`, "
            f"p-value = `{t['p_value']:.4f}` {sig}"
        )
        st.caption(t["interpretation"])

    # -- Charts -------------------------------------------------------
    st.subheader("Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Scatter", "Boxplot", "City bar", "Quick charts"])
    with tab1:
        st.pyplot(scatter_size_price(df))
    with tab2:
        st.pyplot(boxplot_rooms_price(df))
    with tab3:
        st.pyplot(barplot_avg_price_by_city(df))
    with tab4:
        # Requirement coverage: native st.bar_chart as shown in SETUP.md.
        st.bar_chart(df.groupby("rooms")["price"].mean())

    st.caption(f"SQLite file: `{result['db_path']}` — "
               f"{result['clean_count']} cleaned of {result['raw_count']} raw listings.")


if __name__ == "__main__":
    main()
