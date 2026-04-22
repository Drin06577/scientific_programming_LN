# Presentation Appendix — Screenshot & Snippet Checklist

The module expects an appendix of screenshots and code evidence. Gather
these before the presentation so the defence covers every requirement.

## Required screenshots (one per bullet)

- [ ] **Scraper running** — terminal output of `python -m src.scraper`
      showing "Saved N listings to data/raw/listings.json" and the set of
      cantons encountered. (Shows OOP + `requests` + `bs4` in action.)
- [ ] **Regex cleaning** — a terminal or notebook cell showing
      `DataCleaner().clean_price("CHF 2'850.—")` → `2850` and
      `clean_rooms("3.5 rooms")` → `3.5`.
- [ ] **LLM call** — notebook cell with `LLMProcessor().extract_features(...)`
      showing `{'balcony': 1, 'parking': 1, 'furnished': 1}`. Include a
      second screenshot with the regex fallback (unset key) to demonstrate
      graceful degradation.
- [ ] **SQLite inspection** — terminal:
      `sqlite3 housing.db "SELECT rooms, AVG(price) FROM apartments GROUP BY rooms;"`
- [ ] **SQL in Python** — notebook output of
      `db.avg_price_by_location()`.
- [ ] **Statistical tests** — terminal output from `python -m src.pipeline`
      showing Pearson correlations and the balcony t-test with p-values
      and the plain-English interpretation.
- [ ] **Visualizations** — `figures/scatter_size_price.png`,
      `figures/boxplot_rooms_price.png`,
      `figures/barplot_avg_price_by_city.png`.
- [ ] **Streamlit dashboard** — full-page screenshot with summary metrics,
      SQL tables, p-value block and at least one chart tab open.
- [ ] **GitHub structure** — the rendered repo README on GitHub.

## Required code snippets (paste in appendix slides)

1. `SwissHousingScraper.parse_listing` — shows OOP + dict + list + conditional.
2. `DataCleaner._PRICE_RE` + `clean_price` — shows regex + type conversion.
3. `LLMProcessor._extract_via_openai` — shows OpenAI call with `gpt-4o-mini`.
4. `HousingDatabase.avg_price_by_rooms` — shows SQL + GROUP BY.
5. `statistics.correlation_price_size` and `ttest_balcony` — shows scipy
   + tuple unpack + p-value interpretation.
6. `app/streamlit_app.py` `main()` — shows minimal web app wiring.

## Mapping to grading rubric

| Slide | Requirement                              | Evidence                          |
|-------|------------------------------------------|-----------------------------------|
| 3     | Web scraping (real-world data)           | Scraper screenshot + snippet 1    |
| 4     | Regex + LLM + DB (cleaning/methods)      | Screenshots + snippets 2, 3, 4    |
| 5     | Pandas + visualization                   | Viz PNGs + notebook cells         |
| 6     | Statistical testing with p-values        | Stats terminal + snippet 5        |
| 7     | Streamlit web app                        | Dashboard screenshot + snippet 6  |
| 7     | Paradigms (OOP + procedural + types)     | Coverage table in README          |
