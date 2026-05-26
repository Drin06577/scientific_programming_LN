# How to Run the Project

Step-by-step instructions. Run each block from the project root
(`/Users/drinmuslija/git/SciPro`).

---

## 1. First-time setup

```bash
# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify install:

```bash
python -c "import pandas, scipy, bs4, openai, streamlit; print('ok')"
# → ok
```

---

## 2. Configure environment variables

Copy the example file and fill in your OpenAI key:

```bash
cp .env.example .env
# then open .env in your editor
```

Minimum you need to change in `.env`:

```
OPENAI_API_KEY=sk-...your-real-key...
```

The other defaults work as-is (`MODEL=gpt-4o-mini`,
`DB_PATH=./housing.db`, `TARGET_URL=https://flatfox.ch`).

> **If you skip this step**, everything still runs — the LLM step
> automatically drops back to a regex fallback and warns you about it
> (see §6).

---

## 3. Run the full pipeline from the terminal

This is the fastest way to check everything works end-to-end.

```bash
python -m src.pipeline
```

What you should see (with an OpenAI key set):

```
[pipeline] cached: 200 Zurich listings from zurich_apartments.json
[pipeline] fetching detail pages for 200 listings…
[pipeline] detail fetch: 20/200
…
[pipeline] detail fetch: 200/200
[llm] initialised: OpenAI client ready (model=gpt-4o-mini)
[llm] first OpenAI call OK → {'balcony': 0, 'parking': 1, 'furnished': 1}

=== Price quality-control report ===
  scraped (input rows)   : 200
  valid prices kept      : 197
  missing prices         : 0
  suspicious (kept)      : 0
  removed by validator   : 3
    duplicates           : 0
    impossible_price     : 0
    ...
    statistical_outliers : 3

--- Summary ---
n_listings: 197
mean_price: 3032.87
…
--- LLM ---  mode=openai  openai_calls=66  fallback_calls=0  cache_hits=133
--- Statistical tests ---
pearson_size_price: stat=0.394 p=0.0000 sig=True
…
```

The pipeline is **Zurich-canton only**: it reads the cached dataset at
`data/raw/zurich_apartments.json` (produced by `python -m src.collect_zurich`)
and scrapes each listing's detail page for description + features before
cleaning. Default cap is 200 listings (set in `src/pipeline.py`'s
`__main__` block).

The **Price quality-control report** prints automatically after the
validator and tells you, in order, how many listings were scraped, how
many had valid prices, how many were missing, how many are flagged as
suspicious (kept but with a mismatch between the title CHF text and the
structured `rent_gross`), and how many were dropped. The same numbers
are also rendered in the dashboard's "Data Quality" tab.

Artifacts written on disk:
- `data/raw/zurich_apartments.json` — raw Zurich dataset + summary
  stats; now also carries `raw_price_text`, `title_price`, `rent_net`,
  `additional_costs`, and `price_mismatch` per listing.
- `data/processed/apartments.csv` — cleaned + enriched data; same
  provenance columns are passed through.
- `housing.db` — SQLite database
- `figures/*.png` — three legacy notebook plots (see §4 below for the
  new report-ready figures).

---

## 4. Generate the report-ready figures

These are the six static PNGs intended for the printed write-up /
presentation deck. They are saved at 220 dpi with a light theme, large
fonts, sample sizes, and a one-line console interpretation per plot.

```bash
python -m src.report_figures
```

Output (each line is one plot's interpretation):

```
[plot 01] 01_size_vs_price.png: positive correlation r=0.39; each extra m² adds ~CHF 12/month.
[plot 02] 02_rent_by_rooms.png: cheapest median is 3-room flats at CHF 2,158; priciest is 2-room at CHF 3,800. 3 small categories excluded.
[plot 03] 03_rent_ranges.png: most common bucket is CHF 2 000–3 000 (80 listings); …
[plot 04] validating optional feature columns…
[plot 04] 04_feature_impact.png: mean rent difference (with − without): Balcony +CHF 156; …
[plot 05] 05_correlation_matrix.png: strongest correlation in this dataset is price↔size (r = 0.39).
[plot 06] 06_chf_per_m2_distribution.png: median CHF 33.2/m², mean CHF 39.1/m² (right-skewed); …
```

Artifacts:
- `reports/figures/01_size_vs_price.png`
- `reports/figures/02_rent_by_rooms.png`
- `reports/figures/03_rent_ranges.png`
- `reports/figures/04_feature_impact.png`
- `reports/figures/05_correlation_matrix.png`
- `reports/figures/06_chf_per_m2_distribution.png`

The module re-uses `data/processed/apartments.csv` if it exists, and
otherwise runs the pipeline once to produce it — so it is safe to run
on a fresh checkout.

---

## 5. Run the Streamlit dashboard

If you just want to look at the data, this single command is enough —
it triggers the full pipeline internally (collect → scrape → clean →
LLM enrich → validate → SQL → stats → insights) and renders the result.
You do **not** need to run §3 or §4 first.

```bash
streamlit run app/streamlit_app.py
```

This opens a browser tab at `http://localhost:8501` with:
- summary metrics (listings, mean price, mean size)
- a full data table
- four SQL query results (by rooms / city / balcony / top 5)
- the three statistical tests with p-values
- four chart tabs (scatter, boxplot, city bar, Streamlit native)
- the **Data Quality** tab — same QC numbers printed in §3, plus a
  table of suspicious / dropped listings with clickable Flatfox links
- **an LLM status badge at the top** (green = OpenAI, orange = regex fallback)

The same QC report from §3 is also printed to the terminal you launched
Streamlit from. Stop the app with `Ctrl+C` in the terminal.

---

## 6. How to check that the LLM is actually working

There are **four places** to verify this. The first two are the most
useful.

### 6a. Terminal output of `python -m src.pipeline`

Look for these two lines:

| Message | What it means |
|---|---|
| `[llm] initialised: OpenAI client ready (model=gpt-4o-mini)` | ✅ Your API key was loaded successfully. |
| `[llm] first OpenAI call OK → {'balcony': 1, 'parking': 1, 'furnished': 1}` | ✅ At least one real OpenAI call succeeded. |
| `[llm] no OPENAI_API_KEY set → using regex fallback` | ⚠️ No key → not using the LLM. Add it to `.env`. |
| `[llm] OpenAI call failed (...); using regex fallback.` | ⚠️ Key is present but the call itself failed (rate limit, bad key, network). Check the error. |

And near the end of the run:

```
--- LLM ---  mode=openai  openai_calls=25  fallback_calls=0
```

- `mode=openai` + `openai_calls > 0` → LLM is the real thing. ✅
- `mode=regex` → fallback was used. ⚠️

### 6b. Streamlit badge at the top of the dashboard

Right under the page title you'll see one of:

- 🟢 `LLM: OpenAI — 25 calls to gpt-4o-mini.` → working.
- 🟠 `LLM: regex fallback (no OPENAI_API_KEY in .env, or call failed).` → not working.

### 6c. The CSV has LLM-extracted columns

Open `data/processed/apartments.csv`. You should see three new columns:

- `llm_balcony`
- `llm_parking`
- `llm_furnished`

They're `0` or `1`. Compare them to the regex-derived `balcony` /
`parking` columns — the LLM ones often catch things the regex misses
(e.g. "south-facing terrace" → balcony=1).

### 6d. Force-test with a single listing

```bash
python -c "
from src.llm_helper import LLMProcessor
p = LLMProcessor()
print(p.extract_features('Lovely 3-room apartment with south-facing terrace, no parking. Fully furnished kitchen.'))
print('mode after call:', p.mode)
"
```

Expected with key set:
```
[llm] initialised: OpenAI client ready (model=gpt-4o-mini)
[llm] first OpenAI call OK → {'balcony': 1, 'parking': 0, 'furnished': 1}
{'balcony': 1, 'parking': 0, 'furnished': 1}
mode after call: openai
```

Expected without key:
```
[llm] no OPENAI_API_KEY set → using regex fallback
{'balcony': 0, 'parking': 0, 'furnished': 1}
mode after call: regex
```
(Note: regex fallback won't catch "terrace" — only the LLM will.)

---

## 7. Run the Jupyter notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook walks through the pipeline cell by cell — use it for the
presentation appendix screenshots.

---

## 8. Inspect the SQLite database directly

```bash
sqlite3 housing.db "SELECT rooms, ROUND(AVG(price),0) AS avg_price, COUNT(*) FROM apartments GROUP BY rooms;"
```

Or interactively:

```bash
sqlite3 housing.db
sqlite> .tables
sqlite> .schema apartments
sqlite> SELECT city, price FROM apartments ORDER BY price DESC LIMIT 5;
sqlite> .quit
```

---

## 9. Common commands quick reference

| What you want | Command |
|---|---|
| **Just open the dashboard (does everything)** | `streamlit run app/streamlit_app.py` |
| Fresh scrape + full pipeline (with QC report) | `python -m src.pipeline` |
| Build the six report figures (PNG) | `python -m src.report_figures` |
| Just the scraper | `python -m src.scraper` |
| Bulk ZH collection (200 listings + stats) | `python -m src.collect_zurich` |
| Jupyter notebook | `jupyter notebook notebooks/analysis.ipynb` |
| Inspect DB | `sqlite3 housing.db` |
| Test one LLM call | see §6d |
| Wipe generated data | `rm housing.db data/processed/*.csv data/raw/zurich_apartments.json figures/*.png reports/figures/*.png` |

---

## 10. Troubleshooting

- **`[pipeline] cache unreadable … recollecting.`** — the cached
  `data/raw/zurich_apartments.json` is missing/corrupt, so the pipeline
  will run the Zurich collector from scratch (~4–5 min). For a fast
  demo, run `python -m src.collect_zurich` once first.
- **`Price quality-control report` shows many `suspicious (kept)`** —
  the title CHF text in those listings disagrees with the structured
  `rent_gross` by ≥ CHF 100. The pipeline keeps them (gross is
  authoritative) but lists them in the QC report and the dashboard's
  Data Quality tab so you can audit them via the Flatfox URL. Refresh
  the dataset with `python -m src.collect_zurich` to drop stale titles.
- **`plot 04 skipped — no usable feature columns`** in
  `python -m src.report_figures` — `balcony` / `parking` /
  `llm_furnished` are missing, all-null, or have fewer than three
  listings on one side. Run the LLM-enabled pipeline first to populate
  them.
- **Detail-page fetch errors during `[pipeline] detail fetch`** — individual
  listings whose HTML fetch fails get empty `description`/`features`; the
  rest of the pipeline still runs. If all 200 fail, you're offline.
- **`mode=regex` even though I set my key** — check the `.env` file sits
  in the project root, not inside `app/` or `src/`. Also verify:
  `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(repr(os.getenv('OPENAI_API_KEY'))[:12])"`
- **`ttest_balcony: stat=nan`** — too few listings have balconies in the
  current sample. Increase `max_listings` in `src/pipeline.py`'s
  `__main__` block and re-run.
- **`ModuleNotFoundError: No module named 'src'`** — run commands from
  the project root, not from inside `src/`.
