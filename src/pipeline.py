"""End-to-end pipeline — Zurich canton only.

Order:
    load_or_collect_zurich → scrape detail pages (requests + BeautifulSoup)
    → regex clean → LLM enrich → SQLite → SQL queries →
    stats (correlation + t-test) → optional visualizations.

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
from .database import HousingDatabase
from .llm_helper import LLMProcessor, enrich_records
from .scraper import DEFAULT_HEADERS, SwissHousingScraper
from .statistics import run_all_tests

PROCESSED_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "apartments.csv"
ZURICH_JSON = Path(__file__).resolve().parents[1] / "data" / "raw" / "zurich_apartments.json"


# --------------------------------------------------------------------
# Step 1: acquire the raw Zurich dataset (API collector, with file cache)
# --------------------------------------------------------------------

def load_or_collect_zurich(target: int = 200, min_cached: int = 50) -> list[dict]:
    """Load cached Zurich listings from data/raw or run the API collector.

    Cache is the JSON file `collect_zurich` writes. We accept it if it
    already has at least `min_cached` listings.
    """
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
    """Parallel-fetch detail-page HTML and parse with BeautifulSoup.

    Adds `description` and `features` to each listing. Thread-pooled so
    ~50 pages take ~15 s (instead of ~50 s sequential). Failures are
    tolerated — bad pages get empty description/features.
    """
    scraper = SwissHousingScraper()  # reuse `parse_listing`
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

    out: list[dict] = [None] * len(listings)  # preserve input order
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
    """API + HTML record → {title, price, size, rooms, location, features, description}.

    We keep the price/size/rooms as *strings* so cleaning.py's regex
    pipeline still runs (it is the required data-cleaning step in
    SETUP.md). The integers from the API are converted back to the
    string shape a scraped row would have.
    """
    return {
        "title": rec.get("title", ""),
        "price": f"CHF {rec.get('rent_price', '')}",
        "size":  f"{rec.get('living_space_m2', '')} m²",
        "rooms": f"{rec.get('rooms', '')} rooms",
        "location": f"{rec.get('city', '').strip()}, ZH",
        "features": rec.get("features", []) or [],
        "description": rec.get("description", "") or "",
    }


# --------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------

def run_pipeline(use_llm: bool = True, save_figures: bool = False,
                 max_listings: int = 50) -> dict:
    """Run the Zurich end-to-end pipeline.

    `max_listings` caps how many records flow through the LLM + SQLite +
    stats path. Defaults to 50 so the Streamlit demo loads fast; bump to
    200 for the full-dataset run.
    """
    load_dotenv()

    # 1. Raw Zurich listings (cached JSON or fresh API pull).
    base = load_or_collect_zurich(target=200)
    base = base[:max_listings]

    # 2. Scrape each detail page with requests + BeautifulSoup.
    print(f"[pipeline] fetching detail pages for {len(base)} listings…")
    enriched = fetch_details(base)

    # 3. Reshape into cleaner's expected format.
    raw_rows: list[dict] = [_to_cleaner_shape(r) for r in enriched]

    # 4. Regex cleaning + type conversion + balcony/parking flags.
    cleaned_rows = clean_records(raw_rows)

    # 5. LLM enrichment (parallelised inside enrich_records).
    llm_stats = {"mode": "skipped", "calls_openai": 0, "calls_fallback": 0}
    if use_llm:
        processor = LLMProcessor()
        cleaned_rows = enrich_records(cleaned_rows, processor)
        llm_stats = {
            "mode": processor.mode,
            "calls_openai": processor.calls_openai,
            "calls_fallback": processor.calls_fallback,
            "calls_cache": processor.calls_cache,
        }

    # 6. Load into pandas + write CSV.
    df: pd.DataFrame = records_to_dataframe(cleaned_rows)
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_csv = df.copy()
    if "features" in df_csv.columns:
        df_csv["features"] = df_csv["features"].apply(
            lambda xs: ",".join(xs) if isinstance(xs, list) else xs
        )
    df_csv.to_csv(PROCESSED_CSV, index=False)

    # 7. SQLite persistence + the three required aggregation queries.
    db_path = os.getenv("DB_PATH", "./housing.db")
    with HousingDatabase(db_path) as db:
        db.save(df_csv)
        sql_results = {
            "avg_price_by_rooms": db.avg_price_by_rooms(),
            "avg_price_by_location": db.avg_price_by_location(),
            "avg_price_by_balcony": db.avg_price_by_balcony(),
            "top_expensive": db.top_expensive(5),
        }

    # 8. Statistical tests with p-values.
    tests = run_all_tests(df)

    # 9. Optional saved figures.
    if save_figures:
        from .visualization import (
            scatter_size_price,
            boxplot_rooms_price,
            barplot_avg_price_by_city,
        )
        scatter_size_price(df, save=True)
        boxplot_rooms_price(df, save=True)
        barplot_avg_price_by_city(df, save=True)

    return {
        "df": df,
        "summary": summary_stats(df),
        "sql": sql_results,
        "tests": [t.as_dict() for t in tests],
        "db_path": db_path,
        "raw_count": len(base),
        "clean_count": len(cleaned_rows),
        "llm": llm_stats,
    }


if __name__ == "__main__":
    result = run_pipeline(use_llm=True, save_figures=True, max_listings=200)
    print("--- Summary ---")
    for k, v in result["summary"].items():
        print(f"{k}: {v}")
    print(f"\n--- LLM ---  mode={result['llm']['mode']}  "
          f"openai_calls={result['llm']['calls_openai']}  "
          f"fallback_calls={result['llm']['calls_fallback']}")
    print("\n--- SQL: avg price by rooms ---")
    print(result["sql"]["avg_price_by_rooms"])
    print("\n--- Statistical tests ---")
    for t in result["tests"]:
        print(f"{t['name']}: stat={t['statistic']:.3f} p={t['p_value']:.4f} "
              f"sig={t['significant']} — {t['interpretation']}")
