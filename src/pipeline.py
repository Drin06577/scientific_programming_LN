"""End-to-end pipeline — Zurich canton only.

Order:
    load_or_collect_zurich → scrape detail pages (requests + BeautifulSoup)
    → regex clean → LLM enrich → data-quality validate → SQLite → SQL queries →
    stats (correlation, regression, t-test, Mann-Whitney) → insights → optional viz.

Run directly:  python -m src.pipeline
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from . import collect_zurich
from .analysis import records_to_dataframe, summary_stats
from .cleaning import clean_records
from .data_quality import validate, extended_summary, price_categories
from .database import HousingDatabase
from .insights import generate as generate_insights
from .llm_helper import LLMProcessor, enrich_records
from .scraper import DEFAULT_HEADERS, SwissHousingScraper
from .statistics import run_all_tests

PROCESSED_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "apartments.csv"
ZURICH_JSON = Path(__file__).resolve().parents[1] / "data" / "raw" / "zurich_apartments.json"


# --------------------------------------------------------------------
# Step 1: acquire the raw Zurich dataset (API collector, with file cache)
# --------------------------------------------------------------------

def load_or_collect_zurich(target: int = 200, min_cached: int = 50) -> list[dict]:
    """Load cached Zurich listings from data/raw or run the API collector."""
    if ZURICH_JSON.exists():
        try:
            data = json.loads(ZURICH_JSON.read_text(encoding="utf-8"))
            listings = data.get("listings") if isinstance(data, dict) else data
            if listings and len(listings) >= min_cached:
                print(f"[pipeline] cached: {len(listings)} Zurich listings "
                      f"from {ZURICH_JSON.name}")
                return listings
        except Exception as exc:
            print(f"[pipeline] cache unreadable ({exc!r}); recollecting.")

    print(f"[pipeline] collecting Zurich listings (target={target})…")
    listings = collect_zurich.collect(target=target)
    stats_block = collect_zurich.compute_statistics(listings)
    ZURICH_JSON.parent.mkdir(parents=True, exist_ok=True)
    ZURICH_JSON.write_text(
        json.dumps({"statistics": stats_block, "listings": listings},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return listings


# --------------------------------------------------------------------
# Step 2: scrape each listing's detail page for description + features
# --------------------------------------------------------------------

def fetch_details(listings: list[dict], max_workers: int = 4,
                  timeout: int = 15) -> list[dict]:
    """Parallel-fetch detail-page HTML and parse with BeautifulSoup."""
    scraper = SwissHousingScraper()
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    def fetch_one(listing: dict) -> dict:
        try:
            resp = session.get(listing["listing_url"], timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            parsed = scraper.parse_listing(soup)
            return {**listing,
                    "description": parsed.get("description", ""),
                    "features": parsed.get("features", [])}
        except Exception as exc:
            return {**listing, "description": "", "features": [],
                    "_fetch_error": str(exc)[:120]}

    out: list[dict] = [None] * len(listings)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, l): i for i, l in enumerate(listings)}
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            out[idx] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(listings):
                print(f"[pipeline] detail fetch: {done}/{len(listings)}")
    return [x for x in out if x is not None]


# --------------------------------------------------------------------
# Step 3: reshape Zurich API records into the format cleaning.py expects
# --------------------------------------------------------------------

def _to_cleaner_shape(rec: dict) -> dict:
    """API + HTML record → cleaning-stage dict. Preserves listing_url
    and zip_code so the dashboard can render clickable links, plus the
    price-provenance fields (raw_price_text / title_price / rent_net /
    additional_costs / price_mismatch) added by the updated collector.
    Older cached JSON without those fields stays compatible via .get."""
    return {
        "title": rec.get("title", ""),
        "price": f"CHF {rec.get('rent_price', '')}",
        "size":  f"{rec.get('living_space_m2', '')} m²",
        "rooms": f"{rec.get('rooms', '')} rooms",
        "location": f"{rec.get('city', '').strip()}, ZH",
        "zip_code": rec.get("zip_code", ""),
        "listing_url": rec.get("listing_url", ""),
        "features": rec.get("features", []) or [],
        "description": rec.get("description", "") or "",
        # Price provenance — added by collect_zurich.to_output.
        "raw_price_text": rec.get("raw_price_text"),
        "title_price": rec.get("title_price"),
        "rent_net": rec.get("rent_net"),
        "additional_costs": rec.get("additional_costs"),
        "price_mismatch": rec.get("price_mismatch", False),
    }


# --------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------

def run_pipeline(use_llm: bool = True, save_figures: bool = False,
                 max_listings: int = 200, fetch_html: bool = True,
                 apply_quality: bool = True) -> dict:
    """Run the Zurich end-to-end pipeline.

    `max_listings` caps how many records flow through. `fetch_html=False`
    skips the detail-page scrape (faster, but no descriptions for the LLM).
    `apply_quality=True` runs the data-quality validator before stats.
    """
    load_dotenv()

    # 1. Raw Zurich listings (cached JSON or fresh API pull).
    base = load_or_collect_zurich(target=max(200, max_listings))
    base = base[:max_listings]
    raw_count = len(base)

    # 2. Scrape each detail page with requests + BeautifulSoup.
    if fetch_html:
        print(f"[pipeline] fetching detail pages for {len(base)} listings…")
        enriched = fetch_details(base)
    else:
        # Skip detail scrape — useful for fast dev cycles. The LLM then has
        # no descriptions to work on and falls back to the 0-vector.
        enriched = [{**rec, "description": "", "features": []} for rec in base]

    # 3. Reshape into cleaner's expected format.
    raw_rows: list[dict] = [_to_cleaner_shape(r) for r in enriched]

    # 4. Regex cleaning + type conversion + balcony/parking flags.
    cleaned_rows = clean_records(raw_rows)

    # 5. LLM enrichment (parallelised inside enrich_records).
    llm_stats = {"mode": "skipped", "calls_openai": 0,
                 "calls_fallback": 0, "calls_cache": 0}
    if use_llm:
        processor = LLMProcessor()
        cleaned_rows = enrich_records(cleaned_rows, processor)
        llm_stats = {
            "mode": processor.mode,
            "calls_openai": processor.calls_openai,
            "calls_fallback": processor.calls_fallback,
            "calls_cache": processor.calls_cache,
        }

    # 6. Load into pandas, add CHF/m², persist CSV.
    df: pd.DataFrame = records_to_dataframe(cleaned_rows)
    df["chf_per_m2"] = (df["price"] / df["size"]).round(2)

    # 7. Data quality validation.
    if apply_quality:
        df_clean, quality_report = validate(df)
        # Print the price-focused QC summary to the console so anyone running
        # the pipeline sees mismatch/missing/removed counts before the plots.
        quality_report.print_summary()
    else:
        from .data_quality import QualityReport
        df_clean = df.copy()
        quality_report = QualityReport(n_input=len(df), n_output=len(df))

    # Add price-category column for the dashboard.
    df_clean = price_categories(df_clean)

    # Persist the *cleaned* dataset for downstream readers.
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_csv = df_clean.copy()
    if "features" in df_csv.columns:
        df_csv["features"] = df_csv["features"].apply(
            lambda xs: ",".join(xs) if isinstance(xs, list) else xs
        )
    df_csv.to_csv(PROCESSED_CSV, index=False)

    # 8. SQLite persistence + canned aggregation queries.
    db_path = os.getenv("DB_PATH", "./housing.db")
    with HousingDatabase(db_path) as db:
        db.save(df_csv)
        sql_results = {
            "avg_price_by_rooms": db.avg_price_by_rooms(),
            "avg_price_by_location": db.avg_price_by_location(min_n=2),
            "avg_price_by_balcony": db.avg_price_by_balcony(),
            "top_expensive": db.top_expensive(10),
            "top_cheapest": db.top_cheapest(10),
            "price_distribution": db.price_distribution_by_category(),
            "city_ranking": db.city_ranking_with_premium(min_n=3),
            "feature_premium": db.feature_premium(),
        }
        schema = db.schema()

    # 9. Statistical tests with p-values, effect sizes, interpretation.
    tests = run_all_tests(df_clean)

    # 10. Insights.
    insights = generate_insights(df_clean)

    # 11. Optional saved figures.
    if save_figures:
        from .visualization import (
            scatter_size_price,
            boxplot_rooms_price,
            barplot_avg_price_by_city,
        )
        scatter_size_price(df_clean, save=True)
        boxplot_rooms_price(df_clean, save=True)
        barplot_avg_price_by_city(df_clean, save=True)

    return {
        "df": df_clean,
        "df_pre_validation": df,
        "summary": summary_stats(df_clean),
        "extended_summary": extended_summary(df_clean),
        "quality_report": quality_report.as_dict(),
        "dropped_rows": quality_report.dropped_examples,
        "schema": schema,
        "sql": sql_results,
        "tests": [t.as_dict() for t in tests],
        "insights": [i.as_dict() for i in insights],
        "db_path": db_path,
        "raw_count": raw_count,
        "clean_count": len(df_clean),
        "llm": llm_stats,
    }


if __name__ == "__main__":
    result = run_pipeline(use_llm=True, save_figures=True, max_listings=200)
    print("--- Summary ---")
    for k, v in result["summary"].items():
        print(f"{k}: {v}")
    print(f"\n--- Quality ---")
    for k, v in result["quality_report"].items():
        print(f"{k}: {v}")
    print(f"\n--- LLM ---  mode={result['llm']['mode']}  "
          f"openai_calls={result['llm']['calls_openai']}  "
          f"fallback_calls={result['llm']['calls_fallback']}  "
          f"cache_hits={result['llm']['calls_cache']}")
    print("\n--- Statistical tests ---")
    for t in result["tests"]:
        print(f"  [{t['name']}]  stat={t['statistic']:.3f}  "
              f"p={t['p_value']:.4f}  sig={t['significant']}")
        print(f"    → {t['interpretation']}")
    print("\n--- Insights ---")
    for ins in result["insights"]:
        print(f"  • {ins['headline']}")
