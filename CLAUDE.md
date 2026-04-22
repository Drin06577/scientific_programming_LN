# CLAUDE.md — SciPro (WHAT the system is)

Canonical project context. Reflects the **current** state of the system.
No workflows, no debug logs, no experiments — those belong in SKILL.md.

## Project purpose

AI-enhanced Swiss housing price analysis. Scrapes Flatfox apartment
listings (Zurich canton), cleans them, enriches with OpenAI feature
extraction, persists to SQLite, runs statistical tests, and surfaces
everything through a Streamlit dashboard.

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
cleaning.clean_records  (regex)
        ↓
llm_helper.enrich_records  (OpenAI gpt-4o-mini, threaded, SQLite-cached)
        ↓
database.HousingDatabase  (housing.db)
        ↓
statistics.run_all_tests  +  visualization
        ↓
app/streamlit_app.py  (dashboard)
```

### Modules (`src/`)

- `collect_zurich.py` — Flatfox JSON API collector, Zurich-canton only.
- `scraper.py` — `SwissHousingScraper` (OOP) for BeautifulSoup parsing of
  detail pages, with offline fixture fallback.
- `cleaning.py` — regex-based normalization into typed fields.
- `llm_helper.py` — `LLMProcessor` + `enrich_records`; OpenAI call with
  regex fallback and a persistent SQLite cache.
- `database.py` — `HousingDatabase` wrapping `sqlite3`; table
  `apartments`; canned aggregation queries.
- `statistics.py` — Pearson correlation + t-test with p-values.
- `visualization.py` — seaborn charts.
- `pipeline.py` — end-to-end orchestration.

### Request/response flows

- **UI startup flow** — `streamlit_app.py` invokes `pipeline.run_pipeline`
  which executes the full chain above and returns a dict containing
  `df`, `sql_results`, `tests`, and `llm_stats` (mode, calls_openai,
  calls_fallback, calls_cache).
- **LLM enrichment flow** — for each cleaned row, `LLMProcessor.extract_features`
  hashes `(PROMPT_VERSION | model | description[:600])` → `data/llm_cache.sqlite`
  lookup → return cached result **or** call OpenAI (`response_format=json_object`,
  `max_tokens=30`) → store in cache → return. A 24-worker thread pool runs
  these in parallel.
- **Scraper flow** — detail-page fetch uses a shared `requests.Session`
  across a 4-worker pool; parse failures produce empty description/features
  rather than aborting the batch.

## Conventions

- **Source of truth for input data**: `data/raw/zurich_apartments.json`.
  The pipeline reuses it if it already contains ≥ 50 listings.
- **Graceful degradation**: missing `OPENAI_API_KEY` falls back to regex;
  hard-fatal errors (401, insufficient_quota) disable OpenAI for the
  session.
- **Prompt versioning**: bump `LLMProcessor.PROMPT_VERSION` whenever the
  prompt or schema changes — cache keys include it, so old entries
  expire automatically.
- **Requirement tags**: modules include `# Requirement coverage: ...`
  comments pointing at grading rubric items. Do not remove.
- **Zurich-canton scope**: pipeline is explicitly ZH-only. Expanding
  scope requires changes to `collect_zurich` filters and the cleaner.

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

- [ ] Short-circuit LLM when regex finds a confident positive with no
      negation (skip API call entirely for obvious listings).
- [ ] End-to-end verification after the latest speedup: first UI run
      populates `data/llm_cache.sqlite`; second run reports
      `calls_cache == N` with near-zero `calls_openai`.
- [ ] Lift the 50-listing default cap to 200 for the full-dataset run
      once caching is validated.

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
