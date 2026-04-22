# SKILL.md — SciPro (HOW to work)

Reusable workflows and procedures for working on this repo. No
architecture, no business logic, no project state — those live in
CLAUDE.md.

## 1. Starting the app (golden path)

1. Ensure `.env` exists with `OPENAI_API_KEY` set (optional — regex
   fallback kicks in if missing).
2. `source .venv/bin/activate`
3. `streamlit run app/streamlit_app.py`
4. First run: expect the LLM step to populate `data/llm_cache.sqlite`.
5. Re-run the UI: expect `llm_stats.calls_cache` to dominate and
   `calls_openai` to be near zero.

## 2. Adding or changing an LLM feature

1. Edit the prompt or `SCHEMA_KEYS` in `src/llm_helper.py`.
2. **Bump `LLMProcessor.PROMPT_VERSION`** in the same edit — otherwise
   stale cache entries will mask the new prompt.
3. Run one pipeline end-to-end; confirm `calls_openai > 0` on first run
   (cache miss forced by version bump).
4. Update the Roadmap section in CLAUDE.md if scope changed.

## 3. Adjusting LLM throughput

Decision framework:

- **If calls are slow but under RPM limit** → raise `enrich_records(max_workers=…)`.
  `gpt-4o-mini` tier 1 is 500 RPM; 24 is safe, 40 is the practical ceiling.
- **If you see `429` / `RateLimitError`** → lower workers, do not retry
  blindly; the soft-retry in `_extract_via_openai` already handles
  transient bursts.
- **If repeat runs are slow** → the cache isn't hitting. Check
  `LLM_CACHE_PATH`, confirm `PROMPT_VERSION` hasn't changed, inspect
  `data/llm_cache.sqlite` row count.

## 4. Diagnosing a slow UI start

Checklist, in order:

1. Inspect `llm_stats` printed by the pipeline — is `calls_cache` low on
   a repeat run? → cache miss; see §3.
2. Is `calls_fallback` high? → OpenAI was disabled mid-session (check
   logs for 401 / insufficient_quota).
3. Is detail-page scraping the bottleneck? → increase `fetch_details`
   `max_workers` (default 4), but watch for Flatfox throttling.
4. Is `collect_zurich` running from scratch? → it should hit the
   `data/raw/zurich_apartments.json` cache; if not, the file is missing
   or < 50 listings.

## 5. Safe edits to `src/llm_helper.py`

Checklist before committing:

- [ ] `PROMPT_VERSION` bumped if prompt/schema changed.
- [ ] `MAX_DESCRIPTION_CHARS` unchanged, or cache invalidation planned.
- [ ] `max_tokens` still >= the JSON payload size (currently 30 for 3
      boolean keys; raise if you add keys).
- [ ] `response_format={"type":"json_object"}` preserved — removing it
      re-introduces the ```json``` stripping regex path.
- [ ] `calls_cache` still exposed via `processor.calls_cache` and
      surfaced in `pipeline.run_pipeline`'s `llm_stats`.

## 6. Prompt construction strategy (for feature extraction)

- Keep the prompt terse: Claude/GPT obey brevity when the schema is
  rigid.
- State the output shape explicitly ("Return ONLY a compact JSON
  object with integer 0/1 values for keys: …").
- Truncate long descriptions to `MAX_DESCRIPTION_CHARS` — cheap, and
  the signal for balcony / parking / furnished is always early.
- `temperature=0` + `response_format=json_object` → deterministic JSON
  suitable for caching.

## 7. Evaluation checklist after a pipeline change

Run `python -m src.pipeline` and verify:

- [ ] No exceptions.
- [ ] `data/processed/apartments.csv` written, row count > 0.
- [ ] `housing.db` has non-empty `apartments` table (`sqlite3 housing.db
      "SELECT COUNT(*) FROM apartments"`).
- [ ] `llm_stats.mode` is either `openai` or `regex` (not
      `uninitialised`).
- [ ] Streamlit UI loads without errors against the new DB.

## 8. Updating CLAUDE.md alongside code

Trigger rules (strictly follow global memory rules):

- Architecture / flow / command / convention / roadmap change → update
  CLAUDE.md **in the same turn** as the code change.
- Never commit code that invalidates CLAUDE.md without updating it.
- Move any workflow-shaped content you're tempted to add to CLAUDE.md
  into **this** file instead.

## 9. When a new reusable workflow emerges

1. Confirm it's genuinely reusable (≥ 2 plausible future applications).
2. Add a numbered section to SKILL.md with steps, decision points, and
   a checklist.
3. Keep it step-based — prose belongs in CLAUDE.md or comments, not
   here.
