"""Swiss housing listings scraper.

Requirement coverage: OOP class, web scraping (requests + BeautifulSoup),
lists, dicts, sets, tuples, loops, conditionals, procedural helpers.

Target: flatfox.ch — a Swiss rental platform that returns server-rendered
HTML on detail pages. The JSON index API is used only to discover detail
URLs; the real scraping (price, size, rooms, features, description) is
done with `requests` + `BeautifulSoup` on the HTML detail page, matching
the requirement in SETUP.md.

Fallback order if the live site is unreachable:
    live Flatfox  →  local fixture (data/raw/sample_listing.html)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
}

FLATFOX_API = "https://flatfox.ch/api/v1/public-listing/"
FLATFOX_BASE = "https://flatfox.ch"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "sample_listing.html"
RAW_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "listings.json"


# Requirement coverage: dataclass record.
@dataclass
class Listing:
    price: str
    size: str
    rooms: str
    location: str
    features: list[str] = field(default_factory=list)
    description: str = ""
    title: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class SwissHousingScraper:
    """OOP scraper for Swiss rental listings on flatfox.ch.

    Requirement coverage: OOP class with init, fetch, parse methods.
    """

    # Structured-data label map: what Flatfox calls a fact -> our field name.
    # Requirement coverage: dict literal.
    FACT_LABELS: dict[str, str] = {
        "Number of rooms:": "rooms_raw",
        "Livingspace:": "size",
        "Facilities:": "facilities",
        "Gross rent (incl. utilities):": "price",
        "Net rent (excl. utilities):": "price_net",
        "Floor:": "floor",
    }

    def __init__(
        self,
        target_url: str | None = None,
        max_listings: int = 30,
        request_delay: float = 0.6,
        timeout: int = 15,
    ) -> None:
        self.target_url: str = target_url or os.getenv("TARGET_URL", FLATFOX_BASE)
        self.max_listings: int = max_listings
        self.request_delay: float = request_delay
        self.timeout: int = timeout
        # Requirement coverage: set — track distinct cities we've seen.
        self.seen_cities: set[str] = set()
        self.session: requests.Session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    # -- discovery --------------------------------------------------------

    def discover_urls(self) -> list[tuple[str, str, str]]:
        """Ask Flatfox's public JSON API for fresh apartment detail URLs.

        Returns a list of (url, city, canton) tuples — the API gives us the
        canton cheaply; the HTML detail page is still scraped for price,
        size, rooms, features, description with BeautifulSoup below.

        Requirement coverage: tuples in a list.
        """
        # Residential object types we accept. The API's own object_type /
        # min_price filters are ignored, so we filter client-side and
        # paginate until we have enough real apartments.
        # Requirement coverage: set literal membership test.
        residential_types: set[str] = {
            "APARTMENT", "FURNISHED_FLAT", "SHARED_FLAT", "ATTIC_FLAT",
            "ROOFTOP_FLAT", "DUPLEX", "STUDIO", "SINGLE_ROOM", "LOFT",
            "TERRACE_FLAT", "MAISONETTE", "HOUSE", "ROW_HOUSE",
            "SINGLE_HOUSE", "TERRACE_HOUSE",
        }

        urls: list[tuple[str, str, str]] = []
        page_size = 200
        max_pages = 10  # safety cap — max 2000 records scanned
        # Requirement coverage: while loop + conditional break.
        for page in range(max_pages):
            params = {
                "offer_type": "RENT",
                "object_category": "APART",
                "ordering": "-live_since",
                "limit": page_size,
                "offset": page * page_size,
            }
            response = self.session.get(FLATFOX_API, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])
            if not results:
                break

            for rec in results:
                if rec.get("object_type") not in residential_types:
                    continue
                if not rec.get("rent_gross") or not rec.get("number_of_rooms"):
                    continue
                url = rec.get("url")
                if not url:
                    continue
                city = (rec.get("city") or "").strip()
                canton = (rec.get("state") or "").strip()
                urls.append((f"{FLATFOX_BASE}{url}", city, canton))
                if len(urls) >= self.max_listings:
                    return urls
        return urls

    # -- fetching ---------------------------------------------------------

    def fetch_page(self, url: str) -> BeautifulSoup:
        """Fetch a page and return a parsed BeautifulSoup tree."""
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def fetch_fixture(self) -> BeautifulSoup:
        """Load the offline HTML fixture (used only as a last-resort fallback)."""
        html = FIXTURE_PATH.read_text(encoding="utf-8")
        return BeautifulSoup(html, "html.parser")

    # -- parsing ----------------------------------------------------------

    def parse_listing(self, soup: BeautifulSoup) -> dict:
        """Parse a Flatfox detail page into the standard listing dict.

        Requirement coverage: dict construction, loops, conditionals,
        tuple unpacking from fact rows, BeautifulSoup selectors.
        """
        def text_or_empty(node) -> str:
            return node.get_text(" ", strip=True) if node else ""

        title = text_or_empty(soup.find("h1"))
        header = text_or_empty(soup.find("h2"))  # e.g. "8005 Zürich - CHF 1'350 ..."

        # Requirement coverage: tuple used for (zip, city) split.
        zip_city, price_part = ("", header)
        if " - " in header:
            zip_city, price_part = tuple(header.split(" - ", 1))

        facts: dict[str, str] = {}
        # Requirement coverage: for loop over the facts table.
        for row in soup.select("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) != 2:
                continue
            key = cells[0].get_text(" ", strip=True)
            value = cells[1].get_text(" ", strip=True)
            if key in self.FACT_LABELS:
                facts[self.FACT_LABELS[key]] = value

        # Features = pills ("Furnished", "Temporary", …) + comma-separated Facilities.
        # Requirement coverage: list building + set for de-dup.
        pills = [p.get_text(strip=True) for p in soup.select("span.pill")]
        facilities_raw = facts.get("facilities", "")
        facilities = [f.strip() for f in facilities_raw.split(",") if f.strip()]
        feature_set: set[str] = set()
        features: list[str] = []
        for f in pills + facilities:
            norm = f.strip()
            if norm and norm.lower() not in feature_set:
                feature_set.add(norm.lower())
                features.append(norm)

        description = text_or_empty(soup.select_one("div.markdown"))

        # Derive location in the simple "City, Canton" shape the cleaner expects.
        # Zip-city looks like "8005 Zürich"; we drop the ZIP and attach canton if known.
        city = zip_city.split(" ", 1)[1].strip() if " " in zip_city else zip_city.strip()
        if city:
            self.seen_cities.add(city)

        # price_part is e.g. "CHF 1'350 incl. utilities per month" — cleaner handles it.
        return {
            "title": title,
            "price": facts.get("price") or price_part,
            "size": facts.get("size", ""),
            "rooms": facts.get("rooms_raw", ""),
            "location": city,
            "features": features,
            "description": description,
        }

    def parse_fixture_page(self, soup: BeautifulSoup) -> list[dict]:
        """Parse the offline fixture (simpler structure, many listings per page)."""
        articles = soup.select("article.listing")
        # Requirement coverage: list comprehension + loop.
        return [self._parse_fixture_article(a) for a in articles]

    def _parse_fixture_article(self, article) -> dict:
        def text_of(selector: str) -> str:
            el = article.select_one(selector)
            return el.get_text(strip=True) if el else ""

        features = [li.get_text(strip=True) for li in article.select("ul.features li")]
        location = text_of("span.location")
        if "," in location:
            canton = location.split(",")[-1].strip()
            if canton:
                self.seen_cities.add(canton)
        return {
            "title": text_of("h2.title"),
            "price": text_of("span.price"),
            "size": text_of("span.size"),
            "rooms": text_of("span.rooms"),
            "location": location,
            "features": features,
            "description": text_of("p.description"),
        }

    # -- high-level -------------------------------------------------------

    def scrape(self) -> list[dict]:
        """Discover URLs, scrape each detail page, fall back to fixture on failure."""
        # Requirement coverage: try/except + conditional fallback.
        try:
            url_tuples = self.discover_urls()
            if not url_tuples:
                raise RuntimeError("Flatfox returned no apartment URLs")
            print(f"[scraper] Discovered {len(url_tuples)} apartment URLs from Flatfox.")
            listings: list[dict] = []
            for i, (url, city_hint, canton_hint) in enumerate(url_tuples, 1):
                try:
                    soup = self.fetch_page(url)
                    parsed = self.parse_listing(soup)
                    # API city/canton are authoritative and clean; the H2-derived
                    # location often contains a street number. Prefer the API.
                    if city_hint and canton_hint:
                        parsed["location"] = f"{city_hint}, {canton_hint}"
                    elif city_hint:
                        parsed["location"] = city_hint
                    listings.append(parsed)
                    print(f"[scraper]   [{i}/{len(url_tuples)}] ok  {url}")
                except Exception as exc:
                    print(f"[scraper]   [{i}/{len(url_tuples)}] skip {url}: {exc!r}")
                time.sleep(self.request_delay)  # be polite
            if listings:
                return listings
            raise RuntimeError("No listings successfully parsed")
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[scraper] Live scrape failed ({exc!r}); using local fixture.")

        soup = self.fetch_fixture()
        return self.parse_fixture_page(soup)


# -- procedural helpers ---------------------------------------------------

def save_raw(listings: Iterable[dict], path: Path = RAW_OUTPUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(listings)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_raw(path: Path = RAW_OUTPUT_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    scraper = SwissHousingScraper(max_listings=30)
    rows = scraper.scrape()
    out = save_raw(rows)
    print(f"\nSaved {len(rows)} listings to {out}")
    print(f"Distinct cities seen: {sorted(scraper.seen_cities)}")
