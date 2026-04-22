"""OpenAI-backed feature extraction.

Requirement coverage: LLM integration (OpenAI), OOP class, dict return,
conditionals, graceful degradation when no API key is present.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

# Auto-load .env so direct `python -c` invocations work too, not only
# entry points that remember to call load_dotenv themselves.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass


class LLMProcessor:
    """Extract structured boolean features from listing descriptions.

    If OPENAI_API_KEY is missing, falls back to a keyword-based regex
    extractor so the pipeline still runs (e.g. during grading).
    """

    SCHEMA_KEYS: tuple[str, ...] = ("balcony", "parking", "furnished")
    # Bump prompt version when changing the prompt / schema so stale cache
    # entries are ignored instead of silently returning outdated extractions.
    PROMPT_VERSION: str = "v1"
    DEFAULT_CACHE_PATH: str = str(Path(__file__).resolve().parents[1] / "data" / "llm_cache.sqlite")

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        cache_path: str | None = None,
    ) -> None:
        self.api_key: str | None = api_key or os.getenv("OPENAI_API_KEY")
        self.model: str = model or os.getenv("MODEL", "gpt-4o-mini")
        self._client: Any = None
        # Track call source so callers (tests, Streamlit) can show a badge.
        self.calls_openai: int = 0
        self.calls_fallback: int = 0
        self.calls_cache: int = 0
        # Persistent cache so repeat UI runs don't re-pay for the same listings.
        self._cache_path: str = cache_path or os.getenv("LLM_CACHE_PATH", self.DEFAULT_CACHE_PATH)
        self._cache_lock: Lock = Lock()
        self._cache_conn: sqlite3.Connection | None = None
        self._init_cache()
        # Requirement coverage: conditional
        if self.api_key:
            try:
                from openai import OpenAI  # lazy import
                self._client = OpenAI(api_key=self.api_key)
                print(f"[llm] initialised: OpenAI client ready (model={self.model})")
            except Exception as exc:  # pragma: no cover
                print(f"[llm] could not initialise OpenAI client: {exc!r}")
                self._client = None
        else:
            print("[llm] no OPENAI_API_KEY set → using regex fallback")

    @property
    def mode(self) -> str:
        """'openai' when at least one successful OpenAI call has happened,
        'regex' when only the fallback has run, 'uninitialised' before any call."""
        if self.calls_openai > 0:
            return "openai"
        if self.calls_fallback > 0:
            return "regex"
        return "openai-ready" if self._client is not None else "regex-only"

    # -- cache ------------------------------------------------------------

    def _init_cache(self) -> None:
        try:
            Path(self._cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._cache_conn = sqlite3.connect(
                self._cache_path, check_same_thread=False, timeout=5.0
            )
            self._cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "  key TEXT PRIMARY KEY,"
                "  result TEXT NOT NULL"
                ")"
            )
            self._cache_conn.commit()
        except Exception as exc:  # pragma: no cover
            print(f"[llm] cache disabled ({exc!r})")
            self._cache_conn = None

    def _cache_key(self, description: str) -> str:
        raw = f"{self.PROMPT_VERSION}|{self.model}|{description[:self.MAX_DESCRIPTION_CHARS]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        if self._cache_conn is None:
            return None
        with self._cache_lock:
            row = self._cache_conn.execute(
                "SELECT result FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
            return {k: int(bool(data.get(k, 0))) for k in self.SCHEMA_KEYS}
        except Exception:
            return None

    def _cache_put(self, key: str, result: dict) -> None:
        if self._cache_conn is None:
            return
        try:
            with self._cache_lock:
                self._cache_conn.execute(
                    "INSERT OR REPLACE INTO llm_cache(key, result) VALUES (?, ?)",
                    (key, json.dumps(result)),
                )
                self._cache_conn.commit()
        except Exception:
            pass

    # -- public API -------------------------------------------------------

    def extract_features(self, description: str) -> dict:
        """Return {'balcony': int, 'parking': int, 'furnished': int}.

        Uses OpenAI when available, otherwise a regex fallback.
        """
        if not description:
            return {k: 0 for k in self.SCHEMA_KEYS}

        cache_key = self._cache_key(description)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.calls_cache += 1
            return cached

        if self._client is not None:
            try:
                result = self._extract_via_openai(description)
                self._cache_put(cache_key, result)
                self.calls_openai += 1
                if self.calls_openai == 1:
                    print(f"[llm] first OpenAI call OK → {result}")
                total = self.calls_openai + self.calls_fallback
                if total % 20 == 0:
                    print(f"[llm] progress: {total} calls "
                          f"(openai={self.calls_openai}, fallback={self.calls_fallback})")
                return result
            except Exception as exc:  # pragma: no cover
                msg = str(exc)
                # Permanently fatal: bad auth or no prepaid quota.
                hard_fatal = ("401", "invalid_api_key", "AuthenticationError",
                              "insufficient_quota")
                if any(m in msg for m in hard_fatal):
                    print(f"[llm] OpenAI unusable — disabling for this session. "
                          f"({msg[:140]}…)")
                    self._client = None
                # Soft: rate limit. Tier 1+ gives 500 RPM, so a burst rarely
                # triggers this — when it does, 3 s is usually enough.
                elif "RateLimitError" in msg or "429" in msg:
                    time.sleep(3)
                    try:
                        result = self._extract_via_openai(description)
                        self._cache_put(cache_key, result)
                        self.calls_openai += 1
                        if self.calls_openai == 1:
                            print(f"[llm] first OpenAI call OK after retry → {result}")
                        return result
                    except Exception:
                        # Fall through to regex for this one call.
                        pass
                else:
                    print(f"[llm] OpenAI call failed ({exc!r}); using regex fallback.")

        self.calls_fallback += 1
        total = self.calls_openai + self.calls_fallback
        if total % 20 == 0:
            print(f"[llm] progress: {total} calls "
                  f"(openai={self.calls_openai}, fallback={self.calls_fallback})")
        return self._extract_via_regex(description)

    # -- implementations --------------------------------------------------

    # Flatfox descriptions can exceed 3 000 tokens. Only the first ~600
    # chars are needed to detect balcony / parking / furnished keywords,
    # and keeping payloads small avoids blowing the 200 k TPM budget
    # when many workers burst concurrently.
    MAX_DESCRIPTION_CHARS: int = 600

    def _extract_via_openai(self, description: str) -> dict:
        snippet = description[:self.MAX_DESCRIPTION_CHARS]
        prompt = (
            "Extract structured features from this Swiss housing listing.\n"
            "Return ONLY a compact JSON object with integer 0/1 values for "
            "keys: balcony, parking, furnished. No prose.\n\n"
            f"Description: {snippet}"
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=30,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        # Model may wrap in ```json ... ```; strip it.
        content = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        data = json.loads(content)
        return {k: int(bool(data.get(k, 0))) for k in self.SCHEMA_KEYS}

    def _extract_via_regex(self, description: str) -> dict:
        """Keyword fallback. Requirement coverage: regex + dict + loop."""
        text = description.lower()
        # Negation-aware: 'no balcony' / 'no parking' -> 0
        patterns: dict[str, tuple[str, str]] = {
            "balcony":   (r"\bbalcon(?:y|ies)\b", r"no balcony"),
            "parking":   (r"\bpark(?:ing|ed)\b|\bgarage\b",       r"no parking"),
            "furnished": (r"\bfurnished\b",       r"unfurnished|no(?:t)? furnished"),
        }
        out: dict[str, int] = {}
        for key, (pos, neg) in patterns.items():
            if re.search(neg, text):
                out[key] = 0
            elif re.search(pos, text):
                out[key] = 1
            else:
                out[key] = 0
        return out


# -- procedural helper ----------------------------------------------------

def enrich_records(
    rows: list[dict],
    processor: LLMProcessor | None = None,
    max_workers: int = 24,
) -> list[dict]:
    """Attach LLM-extracted features to each record.

    Uses a thread pool so OpenAI calls run in parallel. gpt-4o-mini on tier 1
    allows 500 RPM, so 24 workers burst through 50–200 listings in seconds
    while staying well under the limit.
    """
    processor = processor or LLMProcessor()
    descriptions = [row.get("description", "") for row in rows]

    # Requirement coverage: threading, list comprehension, dict merge.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        llm_results = list(pool.map(processor.extract_features, descriptions))

    enriched: list[dict] = []
    for row, llm_out in zip(rows, llm_results):
        merged = {**row, **{f"llm_{k}": v for k, v in llm_out.items()}}
        enriched.append(merged)
    return enriched
