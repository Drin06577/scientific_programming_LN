# 🏠 AI-Enhanced Swiss Housing Price Analysis

Scientific Programming module project — Switzerland housing prices via **web
scraping + regex cleaning + OpenAI enrichment + SQLite + pandas + statistical
testing + Streamlit**.

---

## Research question

> Which factors (size, number of rooms, location, features) significantly
> influence housing prices in Switzerland, and are these relationships
> statistically significant?

---

## Architecture

```
ZURICH COLLECTOR  (Flatfox JSON API, canton ZH only)
    ↓
RAW JSON  (data/raw/zurich_apartments.json  —  statistics + listings)
    ↓
DETAIL-PAGE SCRAPER  (requests + BeautifulSoup per URL, OOP)
    ↓
DATA CLEANING   (Regex + LLM, OOP)
    ↓
SQLITE DATABASE (housing.db)
    ↓
PANDAS ANALYSIS
    ↓
STATISTICAL TESTING (p-values)
    ↓
VISUALIZATION
    ↓
STREAMLIT DASHBOARD (minimal)
```

## Project structure

```
SciPro/
├── PHASES.md               # Phase tracker
├── SETUP.md                # Binding spec
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── sample_listing.html      # offline demo fixture
│   │   └── zurich_apartments.json   # generated (Zurich dataset + stats)
│   └── processed/
│       └── apartments.csv           # generated
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── scraper.py          # OOP scraper (requests + BeautifulSoup)
│   ├── cleaning.py         # OOP + regex
│   ├── database.py         # SQLite + SQL queries
│   ├── analysis.py         # pandas (procedural)
│   ├── statistics.py       # correlation + t-test + p-values
│   ├── visualization.py    # seaborn charts
│   ├── llm_helper.py       # OpenAI (with regex fallback)
│   └── pipeline.py         # glue: scrape → ... → stats
├── app/
│   └── streamlit_app.py
├── docs/
│   └── presentation_appendix.md
└── housing.db              # generated
```

---

## Setup

### 1. Clone and install

```bash
git clone <this repo>
cd SciPro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

| Variable          | Default                    | Purpose                               |
|-------------------|----------------------------|---------------------------------------|
| `OPENAI_API_KEY`  | —                          | Enables LLM feature extraction.       |
| `MODEL`           | `gpt-4o-mini`              | OpenAI chat model.                    |
| `DB_PATH`         | `./housing.db`             | SQLite location.                      |
| `TARGET_URL`      | `https://flatfox.ch`       | Live scrape target.                   |

The pipeline runs **without** an OpenAI key — the LLM step degrades to a
regex fallback.

---

## Zurich-canton bulk collection

Separate one-shot task: collect 150–200 clean ZH apartment listings and
compute summary statistics. Driven by `src/collect_zurich.py`.

```bash
python -m src.collect_zurich          # target 200
python -m src.collect_zurich 150      # smaller run
```

Filters: canton `ZH` only, full apartments only (no WG / single room /
furnished / temporary), price ≤ 10 000 CHF, size ≥ 20 m², deduplicated
by URL and by (title, price, size).

Output → `data/raw/zurich_apartments.json`:

```json
{
  "statistics": {
    "mean_rent": 4094.7,
    "median_rent": 3840.0,
    "mean_price_per_m2": 54.02,
    "median_price_per_m2": 46.1,
    "min_rent": 1500.0,
    "max_rent": 9820.0
  },
  "listings": [
    {
      "title": "…",
      "rent_price": 2870,
      "rooms": 4.5,
      "living_space_m2": 69,
      "city": "Fehraltorf",
      "zip_code": "8320",
      "listing_url": "https://flatfox.ch/en/flat/…"
    }
  ]
}
```

Runtime is ~4–5 min (Flatfox's API ignores location filters, so the
collector paginates the whole ~34 k-listing index and filters
client-side).

---

## How to run

### Full pipeline end-to-end

```bash
python -m src.pipeline
```

The pipeline is **Zurich-canton only**: it loads the cached Zurich
dataset (`data/raw/zurich_apartments.json`, produced by the collector
described below), scrapes each listing's detail page for description +
features, and feeds the result through cleaning → LLM → SQLite → stats.
Default cap is 50 listings; bump `max_listings` in the module's
`__main__` block for the full 200-listing run.

Produces:
- `data/raw/zurich_apartments.json` (if missing, runs the collector)
- `data/processed/apartments.csv`
- `housing.db`
- `figures/*.png` (if saved)

### Notebook walkthrough

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Opens a dashboard with summary metrics, cleaned data table, SQL query
results, p-value interpretations, and visualizations.

---

## Scraper notes

- **Target:** `flatfox.ch` — a Swiss rental platform. It returns
  server-rendered HTML on detail pages, so `requests` + `BeautifulSoup`
  do the actual listing-field extraction (price, size, rooms, features,
  description) from the HTML, satisfying the `SETUP.md` requirement.
- **Flow:**
  1. Ask Flatfox's public JSON index (`/api/v1/public-listing/`) for
     recent apartment URLs. We paginate and client-side filter to only
     keep residential object types (APARTMENT, FURNISHED_FLAT, …) with
     both a rent amount and a room count.
  2. For each discovered URL, fetch the HTML detail page and parse it
     with BeautifulSoup — title, `<h2>` price line, the `<tr>`-based
     facts table (rooms / living space / facilities), pill badges, and
     the description markdown block.
- **Fallback:** if the live site is unreachable, the scraper falls back
  to the bundled fixture `data/raw/sample_listing.html`. The same
  parsing code path is exercised in both cases — this guarantees the
  demo is reproducible offline.
- **Why not Homegate/ImmoScout/newhome?** All three sit behind Cloudflare
  and return 403 to a plain `requests.get`. Flatfox is the only major
  Swiss rental site currently scrapable with the stdlib-ish toolkit the
  module mandates.
- To switch sources, change `TARGET_URL` in `.env` and adapt the
  `discover_urls` / `parse_listing` selectors.

## Database notes

- `housing.db` is a standard SQLite file. Inspect with:
  ```bash
  sqlite3 housing.db "SELECT rooms, AVG(price) FROM apartments GROUP BY rooms;"
  ```
- Table: `apartments` (columns: title, price, size, rooms, city, canton,
  features, balcony, parking, description, llm_balcony, llm_parking,
  llm_furnished).

---

## Grading coverage

Each module below includes `# Requirement coverage: ...` comments pointing
at the feature it satisfies.

| Requirement                   | Where                                                        |
|-------------------------------|--------------------------------------------------------------|
| Real-world web scraping       | `src/scraper.py` (`SwissHousingScraper`, `requests`, `bs4`)  |
| OOP classes                   | `Scraper`, `DataCleaner`, `HousingDatabase`, `LLMProcessor`  |
| Procedural functions          | `analysis.py`, `statistics.py`, `cleaning.clean_records`     |
| Lists                         | `parse_page`, `clean_features`, `run_all_tests`              |
| Dictionaries                  | `parse_listing`, `summary_stats`, `enrich_records`           |
| Sets                          | `SwissHousingScraper.seen_cantons`, `clean_features`         |
| Tuples                        | `DataCleaner.clean_location`, `scipy.stats` returns          |
| Loops                         | `parse_page`, `enrich_records`, `run_all_tests`              |
| Conditionals                  | all modules (fallback chains, threshold checks)              |
| Regex                         | `src/cleaning.py`, `src/llm_helper.py` (fallback)            |
| SQL queries                   | `src/database.py` (avg-by-rooms/location/balcony + top-N)    |
| p-values + interpretation     | `src/statistics.py` (Pearson + t-test)                       |
| LLM integration               | `src/llm_helper.py` (OpenAI `gpt-4o-mini`)                   |
| Visualization                 | `src/visualization.py` (scatter, box, bar)                   |
| Web app                       | `app/streamlit_app.py`                                       |
| GitHub-ready structure        | `data/`, `notebooks/`, `src/`, `app/`, `README.md`, …        |
