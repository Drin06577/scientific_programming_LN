# 🏠 AI-Enhanced Swiss Housing Price Analysis

Scientific Programming module — Switzerland housing rents via **web
scraping + regex cleaning + OpenAI enrichment + data-quality validation +
SQLite (incl. window functions) + pandas + statistical testing with effect
sizes + interactive Plotly dashboard**.

---

## Research question

> Which factors (size, number of rooms, location, features) significantly
> influence Swiss apartment rents, and are these relationships statistically
> significant?

We answer this on a sample of ~200 listings from **Canton Zurich** (Flatfox),
running a full Pearson / Spearman / OLS / Welch t-test / Mann-Whitney
battery with effect sizes, and surfacing the findings in a multi-tab
analytics dashboard.

---

## Architecture

```
ZURICH COLLECTOR  (Flatfox JSON API, canton ZH only — strict object-type filter)
        ↓
RAW JSON         (data/raw/zurich_apartments.json — listings + summary stats)
        ↓
DETAIL-PAGE SCRAPER  (requests + BeautifulSoup, threaded, OOP)
        ↓
REGEX CLEANING       (typed fields + balcony/parking flags)
        ↓
LLM ENRICHMENT       (OpenAI gpt-4o-mini, SQLite-cached, regex fallback)
        ↓
DATA-QUALITY VALIDATOR  (duplicates, impossibles, CHF/m² band, IQR outliers)
        ↓
SQLITE DATABASE      (apartments table + v_apartments view)
        ↓
PANDAS + STATS       (Pearson, Spearman, OLS, Welch, Mann-Whitney, Shapiro)
        ↓
INSIGHTS GENERATOR   (auto-derived business findings, ranked by confidence)
        ↓
STREAMLIT DASHBOARD  (Plotly, multi-tab, filters, geographic map, SQL explorer)
```

## Project structure

```
SciPro/
├── README.md
├── CLAUDE.md                # WHAT the system is
├── SKILL.md                 # HOW to work in this repo
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/
│   │   ├── sample_listing.html      # offline demo fixture
│   │   └── zurich_apartments.json   # generated (Zurich dataset + stats)
│   ├── processed/
│   │   └── apartments.csv           # generated, post-validation
│   └── llm_cache.sqlite             # persistent LLM extraction cache
├── notebooks/
│   └── analysis.ipynb               # full walkthrough
├── src/
│   ├── scraper.py          # OOP scraper (requests + BeautifulSoup)
│   ├── collect_zurich.py   # Zurich-canton bulk collector
│   ├── cleaning.py         # OOP + regex
│   ├── data_quality.py     # validation + outlier removal + reporting
│   ├── database.py         # SQLite + SQL queries (incl. window functions)
│   ├── analysis.py         # pandas (procedural)
│   ├── statistics.py       # tests + effect sizes + plain-language output
│   ├── insights.py         # auto-generated business findings
│   ├── visualization.py    # matplotlib + Plotly charts
│   ├── llm_helper.py       # OpenAI (with regex fallback + caching)
│   └── pipeline.py         # glue: collect → ... → insights
├── app/
│   └── streamlit_app.py    # multi-tab analytics dashboard
├── docs/
│   └── presentation_appendix.md
└── housing.db              # generated SQLite database
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

```bash
cp .env.example .env
```

| Variable          | Default                  | Purpose                            |
|-------------------|--------------------------|------------------------------------|
| `OPENAI_API_KEY`  | —                        | Enables LLM feature extraction.    |
| `MODEL`           | `gpt-4o-mini`            | OpenAI chat model.                 |
| `DB_PATH`         | `./housing.db`           | SQLite location.                   |
| `LLM_CACHE_PATH`  | `data/llm_cache.sqlite`  | Persistent LLM-result cache.       |
| `TARGET_URL`      | `https://flatfox.ch`     | Live scrape target.                |

The pipeline runs **without** an OpenAI key — the LLM step degrades to a
regex fallback. Subsequent runs hit the cache and pay zero OpenAI cost.

---

## How to run

### Full pipeline end-to-end

```bash
python -m src.pipeline
```

Prints quality report, statistical tests, and the top auto-generated
insights to stdout. Produces:
- `data/raw/zurich_apartments.json` (if missing, runs the collector)
- `data/processed/apartments.csv`
- `housing.db`
- `figures/*.png` (when `save_figures=True`)

### Notebook walkthrough

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens a 7-tab analytics dashboard:

| Tab                  | Contains                                                              |
|----------------------|-----------------------------------------------------------------------|
| 📈 Overview          | KPI cards + scatter, histogram, violin, correlation heatmap, pairplot |
| 🗺️ Geography         | Plotly OpenStreetMap bubble map + city bar + CHF/m² boxplot           |
| 🧪 Statistical tests | Each test with p-value, effect size, interpretation, assumptions      |
| 💡 Key insights      | Auto-generated business-style findings, ranked by confidence          |
| 🗄️ SQL explorer      | Schema + 9 canned queries (incl. NTILE & RANK window functions)       |
| 📋 Listings          | Filterable table with **clickable Flatfox links**, downloadable CSV   |
| 🧹 Data quality      | Validator report + audit trail of dropped rows                        |

The sidebar exposes filters (city, rooms, price range, balcony, parking,
free-text search) that propagate to **every** tab — including the
statistical tests, which re-run on the filtered subset.

---

## Zurich-canton bulk collection

Separate one-shot collector for 150–200 clean ZH apartment listings:

```bash
python -m src.collect_zurich          # target 200
python -m src.collect_zurich 150      # smaller run
```

Filters: canton `ZH` only, full apartments only (no WG / single room /
furnished / temporary), price ≤ 10 000 CHF, size ≥ 20 m², deduplicated
by URL and by (title, price, size). Runtime ~4–5 min — Flatfox's API
ignores location filters, so we paginate the whole ~34 k-listing index
and filter client-side.

---

## Database notes

- `housing.db` is a standard SQLite file. Inspect with:
  ```bash
  sqlite3 housing.db "SELECT rooms, AVG(price) FROM apartments GROUP BY rooms;"
  ```
- Tables: `apartments` (raw) + view `v_apartments` (adds `chf_per_m2`).
- Advanced query examples (all rendered in the SQL Explorer tab):
  - `NTILE(4) OVER (ORDER BY price)` → cheap / medium / expensive / luxury
  - `RANK() OVER (ORDER BY avg_chf_per_m2 DESC)` → city ranking
  - `AVG(...) OVER (ORDER BY size ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)`
    → rolling average price by size

---

## Statistical methodology

| Test                            | Why this test                                                        |
|---------------------------------|----------------------------------------------------------------------|
| Shapiro-Wilk on price           | Documents whether parametric assumptions hold                        |
| Pearson r (size ↔ price)        | Both continuous, roughly linear                                      |
| Spearman ρ (rooms ↔ price)      | Rooms is ordinal-discrete; no normality assumed                      |
| Pearson r (size ↔ CHF/m²)       | Verifies the "small flats charge more per m²" rule                   |
| OLS regression (price ~ size)   | Yields an interpretable CHF-per-m² coefficient with 95% CI           |
| Welch t-test (balcony groups)   | Two means, unequal variances                                         |
| Mann-Whitney U (parking groups) | Non-parametric alternative; robust to bimodal price distribution     |

Each test reports its **statistic, p-value, effect size, plain-language
interpretation, assumptions, and rationale** — all rendered as cards in
the dashboard.

---

## Grading-criteria coverage

| Requirement                          | Where                                                            |
|--------------------------------------|------------------------------------------------------------------|
| Real-world web scraping              | `src/scraper.py`, `src/collect_zurich.py`                        |
| OOP classes                          | `SwissHousingScraper`, `DataCleaner`, `HousingDatabase`, `LLMProcessor`, `QualityReport`, `TestResult`, `Insight` |
| Procedural functions                 | `analysis.py`, `statistics.run_all_tests`, `cleaning.clean_records`, `insights.generate` |
| Lists / dicts / sets / tuples        | Used throughout — see `# Requirement coverage:` tags             |
| Loops + conditionals                 | All modules (fallback chains, validators, batch runs)            |
| Regex                                | `src/cleaning.py`, `src/llm_helper.py` (fallback patterns)       |
| SQL — basic                          | `database.avg_price_by_rooms`, `avg_price_by_location`, `top_expensive` |
| SQL — advanced (window functions)    | `database.price_distribution_by_category` (NTILE), `city_ranking_with_premium` (RANK + CTE + subquery), `running_avg_price_by_size` (windowed AVG) |
| Statistical tests + p-values         | `src/statistics.py` (7 tests, all with effect sizes)             |
| Effect size + plain-language interp. | Every `TestResult` has `effect_size_label` + `interpretation`    |
| LLM integration                      | `src/llm_helper.py` (OpenAI gpt-4o-mini, SQLite-cached)          |
| Data validation                      | `src/data_quality.py` (`validate` + `QualityReport`)             |
| Visualization — static               | `src/visualization.py` matplotlib/seaborn (notebook)             |
| Visualization — interactive          | `src/visualization.py` Plotly (dashboard, geographic map, heatmap, pairplot) |
| Web application                      | `app/streamlit_app.py` (7-tab Plotly dashboard with filters)     |
| Clickable hyperlinks in tables       | Listings tab uses `st.column_config.LinkColumn`                  |
| Storytelling / insights              | `src/insights.py` (auto-generated findings)                      |
| Notebook walkthrough                 | `notebooks/analysis.ipynb`                                       |
| Cached / persistent intermediate state | `data/llm_cache.sqlite`, `data/raw/zurich_apartments.json`       |

---

## Known tradeoffs

- **Flatfox-only source** — Homegate, ImmoScout, newhome all return 403
  behind Cloudflare; Flatfox is the only major Swiss rental site
  scrapable with the stdlib-ish toolkit the module mandates.
- **Flatfox API ignores location filters** — collector paginates the
  full ~34 k-listing index and filters client-side.
- **`gpt-4o-mini` cost vs accuracy** — chosen for the three boolean
  features we extract; accuracy is cross-checked by the regex fallback.
- **Cache is content-keyed, not listing-keyed** — two listings with
  identical descriptions share a cache entry. Invalidate by bumping
  `LLMProcessor.PROMPT_VERSION`.
- **Geographic map uses bundled centroids** — accurate enough for a
  bubble overlay at zoom 8.5; we don't fetch coordinates at runtime,
  to keep the dashboard offline-capable.
