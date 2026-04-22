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
> (see §5).

---

## 3. Run the full pipeline from the terminal

This is the fastest way to check everything works end-to-end.

```bash
python -m src.pipeline
```

What you should see (with an OpenAI key set):

```
[pipeline] cached: 200 Zurich listings from zurich_apartments.json
[pipeline] fetching detail pages for 50 listings…
[pipeline] detail fetch: 20/50
[pipeline] detail fetch: 40/50
[pipeline] detail fetch: 50/50
[llm] initialised: OpenAI client ready (model=gpt-4o-mini)
[llm] first OpenAI call OK → {'balcony': 0, 'parking': 1, 'furnished': 1}
--- Summary ---
n_listings: 50
mean_price: 4165.0
…
--- LLM ---  mode=openai  openai_calls=50  fallback_calls=0
--- Statistical tests ---
pearson_size_price: stat=0.540 p=0.0001 sig=True
pearson_rooms_price: stat=0.179 p=0.2126 sig=False
ttest_balcony: …
```

The pipeline is **Zurich-canton only**: it reads the cached dataset at
`data/raw/zurich_apartments.json` (produced by `python -m src.collect_zurich`)
and scrapes each listing's detail page for description + features before
cleaning. Default cap is 50 listings; bump `max_listings=200` in
`src/pipeline.py`'s `__main__` block for the full set.

Artifacts written on disk:
- `data/raw/zurich_apartments.json` — raw Zurich dataset + summary stats
- `data/processed/apartments.csv` — cleaned + enriched data
- `housing.db` — SQLite database
- `figures/*.png` — three saved plots

---

## 4. Run the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

This opens a browser tab at `http://localhost:8501` with:
- summary metrics (listings, mean price, mean size)
- a full data table
- four SQL query results (by rooms / city / balcony / top 5)
- the three statistical tests with p-values
- four chart tabs (scatter, boxplot, city bar, Streamlit native)
- **an LLM status badge at the top** (green = OpenAI, orange = regex fallback)

Stop the app with `Ctrl+C` in the terminal.

---

## 5. How to check that the LLM is actually working

There are **four places** to verify this. The first two are the most
useful.

### 5a. Terminal output of `python -m src.pipeline`

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

### 5b. Streamlit badge at the top of the dashboard

Right under the page title you'll see one of:

- 🟢 `LLM: OpenAI — 25 calls to gpt-4o-mini.` → working.
- 🟠 `LLM: regex fallback (no OPENAI_API_KEY in .env, or call failed).` → not working.

### 5c. The CSV has LLM-extracted columns

Open `data/processed/apartments.csv`. You should see three new columns:

- `llm_balcony`
- `llm_parking`
- `llm_furnished`

They're `0` or `1`. Compare them to the regex-derived `balcony` /
`parking` columns — the LLM ones often catch things the regex misses
(e.g. "south-facing terrace" → balcony=1).

### 5d. Force-test with a single listing

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

## 6. Run the Jupyter notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook walks through the pipeline cell by cell — use it for the
presentation appendix screenshots.

---

## 7. Inspect the SQLite database directly

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

## 8. Common commands quick reference

| What you want | Command |
|---|---|
| Fresh scrape + full pipeline | `python -m src.pipeline` |
| Just the scraper | `python -m src.scraper` |
| Bulk ZH collection (200 listings + stats) | `python -m src.collect_zurich` |
| Open the dashboard | `streamlit run app/streamlit_app.py` |
| Jupyter notebook | `jupyter notebook notebooks/analysis.ipynb` |
| Inspect DB | `sqlite3 housing.db` |
| Test one LLM call | see §5d |
| Wipe generated data | `rm housing.db data/processed/*.csv data/raw/zurich_apartments.json figures/*.png` |

---

## 9. Troubleshooting

- **`[pipeline] cache unreadable … recollecting.`** — the cached
  `data/raw/zurich_apartments.json` is missing/corrupt, so the pipeline
  will run the Zurich collector from scratch (~4–5 min). For a fast
  demo, run `python -m src.collect_zurich` once first.
- **Detail-page fetch errors during `[pipeline] detail fetch`** — individual
  listings whose HTML fetch fails get empty `description`/`features`; the
  rest of the pipeline still runs. If all 50 fail, you're offline.
- **`mode=regex` even though I set my key** — check the `.env` file sits
  in the project root, not inside `app/` or `src/`. Also verify:
  `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(repr(os.getenv('OPENAI_API_KEY'))[:12])"`
- **`ttest_balcony: stat=nan`** — too few listings have balconies in the
  current sample. Bump `max_listings` in `src/scraper.py`'s `__main__`
  block to 50 and re-run.
- **`ModuleNotFoundError: No module named 'src'`** — run commands from
  the project root, not from inside `src/`.
