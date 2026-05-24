# CLAUDE.md — SciPro (WHAT the system is)

Canonical project context. Reflects the **current** state of the system.
No workflows, no debug logs, no experiments — those belong in SKILL.md.

## Project purpose

AI-enhanced Swiss housing price analysis. Scrapes Flatfox apartment
listings (Zurich canton), cleans them with regex, enriches with OpenAI
feature extraction, validates with a dedicated data-quality module,
persists to SQLite, runs a statistical battery (Pearson, Spearman, OLS,
Welch, Mann-Whitney, Shapiro) with effect sizes and plain-language
interpretation, auto-generates business-style insights, and surfaces
everything through a multi-tab Plotly Streamlit dashboard.

**Research question:** which factors (size, rooms, location, features)
significantly influence Swiss housing prices?

## Commands

| Task                    | Command                                         |
|-------------------------|-------------------------------------------------|
| Install                 | `pip install -r requirements.txt`               |
| Collect Zurich dataset  | `python -m src.collect_zurich [target]`         |
| Run full pipeline       | `python -m src.pipeline`                        |
| Launch UI               | `streamlit run app/streamlit_app.py`            |
| Notebook walkthrough    | `jupyter notebook notebooks/analysis.ipynb`     |
| Inspect DB              | `sqlite3 housing.db "SELECT * FROM apartments"` |

## Architecture overview

```
collect_zurich  →  data/raw/zurich_apartments.json
        ↓
pipeline.fetch_details  (BeautifulSoup detail pages, threaded)
        ↓
cleaning.clean_records  (regex; preserves listing_url + zip_code)
        ↓
llm_helper.enrich_records  (OpenAI gpt-4o-mini, threaded, SQLite-cached)
        ↓
data_quality.validate  (duplicates, impossibles, CHF/m² band, IQR outliers)
        ↓
database.HousingDatabase  (housing.db + v_apartments view + window-function queries)
        ↓
statistics.run_all_tests  (Pearson / Spearman / OLS / Welch / MWU / Shapiro)
        ↓
insights.generate  (auto-derived business findings, confidence-tagged)
        ↓
app/streamlit_app.py  (Plotly 7-tab dashboard: Overview, Geography, Stats,
                      Insights, SQL explorer, Listings, Data quality)
```

### Modules (`src/`)

- `collect_zurich.py` — Flatfox JSON API collector, Zurich-canton only,
  strict object-type filter.
- `scraper.py` — `SwissHousingScraper` (OOP) for BeautifulSoup parsing of
  detail pages, with offline fixture fallback.
- `cleaning.py` — regex-based normalization into typed fields. Preserves
  `listing_url` and `zip_code` for downstream linkout rendering.
- `llm_helper.py` — `LLMProcessor` + `enrich_records`; OpenAI call with
  regex fallback and a persistent SQLite cache.
- `data_quality.py` — `validate(df) → (df_clean, QualityReport)`. Drops
  duplicates, impossible values (price/size/rooms), CHF/m² outside [10, 200],
  and Tukey k=3 statistical outliers. Also exposes `extended_summary` and
  `price_categories`.
- `database.py` — `HousingDatabase` wrapping `sqlite3`; table `apartments`
  + view `v_apartments`. Canned queries include NTILE quartile bucketing,
  RANK-based city ranking with market-average premium, and a windowed
  rolling-average query.
- `statistics.py` — 7 tests with effect sizes and rationale:
  Shapiro-Wilk, Pearson (size↔price, size↔CHF/m²), Spearman (rooms↔price),
  OLS regression, Welch t-test (balcony), Mann-Whitney (parking). Every
  `TestResult` carries `interpretation`, `why`, and `assumptions` strings
  that the dashboard renders verbatim.
- `insights.py` — auto-generates ~10 business-style findings from the
  cleaned dataset. Each `Insight` is tagged `low` / `medium` / `high`
  confidence based on p-values + sample size.
- `visualization.py` — static matplotlib/seaborn (kept for the notebook)
  plus interactive Plotly chart functions: city bar, scatter+OLS,
  violin+box, CHF/m² histogram, correlation heatmap, geographic
  OpenStreetMap bubble map, scatter matrix, price categories, feature
  comparison.
- `pipeline.py` — end-to-end orchestration. Returns a single dict with
  `df`, `summary`, `extended_summary`, `quality_report`, `dropped_rows`,
  `schema`, `sql`, `tests`, `insights`, `llm`, `db_path`.

### Request/response flows

- **UI startup flow** — `streamlit_app.py` calls `pipeline.run_pipeline`
  which executes the full chain above. Sidebar filters then narrow the
  returned DataFrame; statistical tests + insights re-run on the filtered
  subset so they remain interactive.
- **LLM enrichment flow** — for each cleaned row, `LLMProcessor.extract_features`
  hashes `(PROMPT_VERSION | model | description[:600])` → `data/llm_cache.sqlite`
  lookup → return cached result **or** call OpenAI (`response_format=json_object`,
  `max_tokens=30`) → store in cache → return. A 24-worker thread pool runs
  these in parallel.
- **Scraper flow** — detail-page fetch uses a shared `requests.Session`
  across a 4-worker pool; parse failures produce empty description/features
  rather than aborting the batch.
- **Validation flow** — `data_quality.validate` runs in a fixed order
  (dedup → required fields → impossibles → CHF/m² → IQR) so the audit
  trail in the dashboard's Data Quality tab is interpretable.

## Conventions

- **Source of truth for input data**: `data/raw/zurich_apartments.json`.
  The pipeline reuses it if it already contains ≥ 50 listings.
- **Source of truth for cleaned data**: `data/processed/apartments.csv`,
  written after validation. Includes `chf_per_m2` and `price_category`.
- **Graceful degradation**: missing `OPENAI_API_KEY` falls back to regex;
  hard-fatal errors (401, insufficient_quota) disable OpenAI for the
  session.
- **Prompt versioning**: bump `LLMProcessor.PROMPT_VERSION` whenever the
  prompt or schema changes — cache keys include it.
- **Requirement tags**: modules include `# Requirement coverage: ...`
  comments pointing at grading rubric items. Do not remove.
- **Zurich-canton scope**: pipeline is explicitly ZH-only. Expanding
  scope requires changes to `collect_zurich` filters and the cleaner.
- **Plotly for interactive viz, matplotlib for static exports**: Plotly
  charts power the dashboard; matplotlib functions are kept so the
  notebook works headlessly and so PNGs can be exported to `figures/`.

## Configuration

Environment variables (via `.env`):

| Variable          | Default                | Purpose                          |
|-------------------|------------------------|----------------------------------|
| `OPENAI_API_KEY`  | —                      | Enables LLM feature extraction.  |
| `MODEL`           | `gpt-4o-mini`          | OpenAI chat model.               |
| `DB_PATH`         | `./housing.db`         | SQLite location for apartments.  |
| `LLM_CACHE_PATH`  | `data/llm_cache.sqlite`| Persistent LLM-result cache.     |
| `TARGET_URL`      | `https://flatfox.ch`   | Live scrape target.              |

## Roadmap / next steps

- [x] Add data-quality validation module with full audit trail.
- [x] Expand statistical battery with effect sizes + plain-language interpretation.
- [x] Add interactive Plotly dashboard with filters, geographic map, SQL explorer.
- [x] Add auto-generated business insights with confidence tags.
- [x] Add window-function SQL queries (NTILE, RANK, rolling AVG).
- [x] Add clickable Flatfox links in the listings table.
- [ ] Short-circuit LLM when regex finds a confident positive with no
      negation (skip API call entirely for obvious listings).
- [ ] Persist geographic centroid map externally (CSV) so it can be
      updated without code changes.

## Known tradeoffs

- **Flatfox-only source**: Homegate, ImmoScout, and newhome all return
  403 behind Cloudflare. Flatfox is the only scrapable option given the
  stdlib-ish toolkit the module mandates.
- **Flatfox API ignores location filters**: collector paginates the full
  ~34k-listing index and filters client-side — ~4–5 min runtime for a
  full collection, acceptable because results are cached to JSON.
- **`gpt-4o-mini` cost vs accuracy**: chosen over larger models because
  only three boolean features are extracted; accuracy is verified by a
  regex fallback that agrees on the unambiguous cases.
- **Cache is content-keyed, not listing-keyed**: two listings with
  identical descriptions share a cache entry. Desirable, but invalidates
  per-listing invalidation strategies — invalidate via `PROMPT_VERSION`.
- **IQR k=3 instead of textbook 1.5**: Zurich rents have a heavy right
  tail (lakeside luxury) that's perfectly valid. k=1.5 would discard
  ~10% of the dataset; k=3 only catches genuine data errors.
- **Geographic centroids are hard-coded**: keeps the dashboard offline-
  capable. The trade is that newly-listed cities outside the dictionary
  silently drop off the map (still appear in every other tab).
