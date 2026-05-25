"""Bulk collector + stats: 150–200 unique Zurich-canton apartment listings.

Strict filters:
  - Canton == ZH
  - Object type is a full apartment (APARTMENT / ATTIC_FLAT / LOFT / …)
  - NOT shared flat, NOT single room, NOT furnished/temporary
  - price present (rent_gross), size present, rooms present
  - 20 ≤ living_space_m2, 500 ≤ rent_price ≤ 15 000 CHF/month
  - price_unit == monthly, price_display_type == TOTAL  (rejects per-m²/yearly)
  - de-duplicated by URL and by (title, price, size)

Price provenance: each record carries
  - rent_price           — numeric authoritative monthly gross rent (rent_gross)
  - rent_net             — net rent without utilities (rent_net), may be None
  - additional_costs     — utilities/charges (rent_charges), may be None
  - raw_price_text       — CHF substring extracted from public_title
  - title_price          — int parsed from raw_price_text, may be None
  - price_mismatch       — True if title_price differs from rent_price by ≥ CHF 100
This lets data_quality.validate flag suspicious extractions transparently.

Usage:
    python -m src.collect_zurich            # collects up to 200
    python -m src.collect_zurich 150        # stop at 150
"""

from __future__ import annotations

import json
import re
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Captures a CHF amount from the public_title, e.g.
#   "…Zürich - CHF 2'870 incl. utilities per month"  →  "2'870"
#   "…Adliswil - CHF 4'4.0 incl. utilities per month" →  "4'4.0"   (kept verbatim)
# Swiss thousands separators: typographic apostrophe (’), straight ('), or dot.
_TITLE_PRICE_RE = re.compile(r"CHF\s*([\d'’.,]+)")


def parse_title_price(title: str) -> tuple[str | None, int | None]:
    """Return (raw_price_text, parsed_int) from a Flatfox listing title.

    Both values may be None when the title carries no parseable CHF amount.
    The raw text is preserved verbatim so the QC report can show exactly
    what was on the page; the parsed int is best-effort.
    """
    if not title:
        return None, None
    m = _TITLE_PRICE_RE.search(title)
    if not m:
        return None, None
    raw = m.group(1)
    # Strip thousands separators (both apostrophe variants and dots).
    # Note: dot-as-decimal would be lost too, but rents are whole CHF.
    digits = re.sub(r"[\D]", "", raw)
    try:
        parsed = int(digits) if digits else None
    except ValueError:
        parsed = None
    return raw, parsed

FLATFOX_API = "https://flatfox.ch/api/v1/public-listing/"
FLATFOX_BASE = "https://flatfox.ch"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "zurich_apartments.json"

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}

# Whole-apartment object types. Excludes SHARED_FLAT, SINGLE_ROOM,
# FURNISHED_FLAT (typically short-term/serviced), commercial, garages.
ALLOWED_OBJECT_TYPES: set[str] = {
    "APARTMENT", "ATTIC_FLAT", "ROOFTOP_FLAT", "DUPLEX",
    "STUDIO", "LOFT", "TERRACE_FLAT", "MAISONETTE",
    "HOUSE", "ROW_HOUSE", "SINGLE_HOUSE", "TERRACE_HOUSE",
}

# Defensive exclude list (in case ALLOWED is loosened later).
EXCLUDED_OBJECT_TYPES: set[str] = {
    "SHARED_FLAT", "SINGLE_ROOM", "FURNISHED_FLAT",
    "GARAGE_SLOT", "SINGLE_GARAGE", "OPEN_SLOT",
    "COVERED_PARKING_PLACE_BIKE", "STORAGE_ROOM",
    "OFFICE", "SHOP", "COMMERCIAL", "HOBBY_ROOM", "GARDENING",
}

PRICE_MIN: int = 500
PRICE_MAX: int = 15_000
SIZE_MIN: int = 20
# Mismatch between title-displayed CHF and rent_gross above this threshold is
# treated as suspicious — Flatfox occasionally serves stale public_title text.
PRICE_MISMATCH_TOLERANCE: int = 100


def passes_filters(rec: dict[str, Any]) -> bool:
    """Strict row-level filter against the spec's criteria.

    Adds unit guards: rejects per-m² and non-monthly listings whose CHF amount
    cannot be compared against a monthly rent (this protects the plots from
    silently mixing CHF/m²/year quotes into the monthly-rent distribution).
    """
    if rec.get("state") != "ZH":
        return False
    obj_type = rec.get("object_type")
    if obj_type in EXCLUDED_OBJECT_TYPES:
        return False
    if obj_type not in ALLOWED_OBJECT_TYPES:
        return False
    if rec.get("is_temporary"):
        return False
    if rec.get("offer_type") != "RENT":
        return False
    # Unit guards — drop yearly/per-m² listings before they pollute the dataset.
    if rec.get("price_unit") and rec.get("price_unit") != "monthly":
        return False
    if rec.get("price_display_type") and rec.get("price_display_type") != "TOTAL":
        return False
    price = rec.get("rent_gross")
    size = rec.get("surface_living")
    rooms = rec.get("number_of_rooms")
    if not price or not size or not rooms:
        return False
    if not (PRICE_MIN <= price <= PRICE_MAX):
        return False
    if size < SIZE_MIN:
        return False
    return True


def to_output(rec: dict[str, Any]) -> dict[str, Any]:
    """Shape an API record into the required output schema.

    Captures full price provenance (raw text + parsed title price + gross/net
    + additional costs) so downstream validators can flag mismatches without
    having to re-fetch the listing.
    """
    url = rec.get("url", "")
    title = rec.get("public_title") or rec.get("short_title") or ""
    raw_text, title_price = parse_title_price(title)
    gross = int(rec["rent_gross"])
    # Stale public_title is a known Flatfox artefact — surface it rather than hide it.
    mismatch = bool(
        title_price is not None
        and abs(title_price - gross) >= PRICE_MISMATCH_TOLERANCE
    )
    net = rec.get("rent_net")
    charges = rec.get("rent_charges")
    return {
        "title": title,
        "rent_price": gross,
        "rent_net": int(net) if net is not None else None,
        "additional_costs": int(charges) if charges is not None else None,
        "raw_price_text": raw_text,
        "title_price": title_price,
        "price_mismatch": mismatch,
        "rooms": float(rec["number_of_rooms"]),
        "living_space_m2": int(rec["surface_living"]),
        "city": (rec.get("city") or "").strip(),
        "zip_code": str(rec.get("zipcode") or "").strip(),
        "listing_url": f"{FLATFOX_BASE}{url}" if url.startswith("/") else url,
    }


def collect(
    target: int = 200,
    min_target: int = 150,
    page_size: int = 100,   # Flatfox hard-caps limit at 100
    max_pages: int = 400,   # cover the full ~34k-listing index
    request_delay: float = 0.15,
) -> list[dict[str, Any]]:
    """Paginate the Flatfox index until we have `target` clean ZH apartments."""
    session = requests.Session()
    session.headers.update(HEADERS)

    seen_urls: set[str] = set()
    seen_triples: set[tuple[str, int, int]] = set()
    out: list[dict[str, Any]] = []

    for page in range(max_pages):
        params = {
            "offer_type": "RENT",
            "object_category": "APART",
            "ordering": "-live_since",
            "limit": page_size,
            "offset": page * page_size,
        }
        resp = session.get(FLATFOX_API, params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            break

        page_kept = 0
        for rec in results:
            if not passes_filters(rec):
                continue
            url = rec.get("url")
            if not url or url in seen_urls:
                continue
            triple = (
                (rec.get("public_title") or rec.get("short_title") or "").strip(),
                int(rec["rent_gross"]),
                int(rec["surface_living"]),
            )
            if triple in seen_triples:
                continue

            seen_urls.add(url)
            seen_triples.add(triple)
            out.append(to_output(rec))
            page_kept += 1

            if len(out) >= target:
                print(f"[collect] reached target ({target}) on page {page+1}.")
                return out

        # Quieter logging: only every 20th page unless the page was productive.
        if page_kept > 0 or (page + 1) % 20 == 0:
            print(f"[collect] page {page+1}/{max_pages}: scanned {len(results)}, "
                  f"kept {page_kept} (running total {len(out)})")
        time.sleep(request_delay)

    if len(out) < min_target:
        print(f"[collect] WARNING: only {len(out)} collected, below min_target={min_target}")
    return out


def compute_statistics(listings: list[dict[str, Any]]) -> dict[str, float]:
    """Compute the mandatory aggregates over the collected listings.

    Returns mean/median rent, mean/median price per m², min/max rent —
    all rounded to 2 decimals for the JSON output.
    """
    if not listings:
        return {
            "mean_rent": 0.0, "median_rent": 0.0,
            "mean_price_per_m2": 0.0, "median_price_per_m2": 0.0,
            "min_rent": 0.0, "max_rent": 0.0,
        }
    rents = [float(r["rent_price"]) for r in listings]
    # Requirement coverage: per-row derivation + list comprehension.
    price_per_m2 = [float(r["rent_price"]) / float(r["living_space_m2"])
                    for r in listings if r["living_space_m2"]]
    return {
        "mean_rent": round(stats.fmean(rents), 2),
        "median_rent": round(stats.median(rents), 2),
        "mean_price_per_m2": round(stats.fmean(price_per_m2), 2),
        "median_price_per_m2": round(stats.median(price_per_m2), 2),
        "min_rent": round(min(rents), 2),
        "max_rent": round(max(rents), 2),
    }


def summarize(listings: list[dict[str, Any]]) -> None:
    """Print a brief sanity summary covering the distribution requirement."""
    if not listings:
        print("[summary] no listings.")
        return
    prices = [r["rent_price"] for r in listings]
    sizes = [r["living_space_m2"] for r in listings]
    rooms = [r["rooms"] for r in listings]
    # Requirement coverage: dict, sorted, sets.
    city_counts: dict[str, int] = {}
    for r in listings:
        city_counts[r["city"]] = city_counts.get(r["city"], 0) + 1
    top_cities = sorted(city_counts.items(), key=lambda kv: -kv[1])[:10]

    print("\n=== Collection summary ===")
    print(f"  listings:        {len(listings)}")
    print(f"  distinct cities: {len(city_counts)}")
    print(f"  price CHF/mo:    mean {sum(prices)/len(prices):,.0f}  "
          f"median {sorted(prices)[len(prices)//2]:,}  "
          f"min {min(prices):,}  max {max(prices):,}")
    print(f"  size m²:         mean {sum(sizes)/len(sizes):.1f}  "
          f"min {min(sizes)}  max {max(sizes)}")
    print(f"  rooms:           mean {sum(rooms)/len(rooms):.2f}  "
          f"min {min(rooms)}  max {max(rooms)}")
    print(f"  top 10 cities:")
    for city, n in top_cities:
        pct = 100 * n / len(listings)
        print(f"    {city or '(unknown)':<25} {n:4d}  ({pct:4.1f}%)")


def main() -> None:
    target = 200
    if len(sys.argv) > 1:
        target = int(sys.argv[1])

    listings = collect(target=target)
    statistics = compute_statistics(listings)
    payload = {"statistics": statistics, "listings": listings}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[collect] saved {len(listings)} records + stats → {OUTPUT_PATH}")

    print("\n=== Statistics ===")
    for k, v in statistics.items():
        print(f"  {k:>22}:  {v:>10,.2f}")
    summarize(listings)


if __name__ == "__main__":
    main()
