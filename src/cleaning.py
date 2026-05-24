"""Regex-based data cleaning.

Requirement coverage: OOP class, regex transformation, conditionals,
loops, tuples (return type), procedural helper.
"""

from __future__ import annotations

import re
from typing import Iterable


class DataCleaner:
    """Cleans raw scraped string values into numeric/typed values."""

    # Requirement coverage: regex
    _PRICE_RE = re.compile(r"[^\d]")
    _NUMERIC_RE = re.compile(r"[^\d.]")
    _ROOMS_RE = re.compile(r"(\d+(?:\.\d+)?)")
    _LOCATION_RE = re.compile(r"^(?P<city>[^,]+?)(?:,\s*(?P<canton>[A-Z]{2}))?$")

    def clean_price(self, value: str) -> int | None:
        """'CHF 2\\'850.—' -> 2850"""
        if not value:
            return None
        digits = self._PRICE_RE.sub("", value)
        return int(digits) if digits else None

    def clean_size(self, value: str) -> float | None:
        """'85 m²' -> 85.0"""
        if not value:
            return None
        numeric = self._NUMERIC_RE.sub("", value)
        return float(numeric) if numeric else None

    def clean_rooms(self, value: str) -> float | None:
        """'3.5 rooms' -> 3.5; also handles '1 ½' -> 1.5."""
        if not value:
            return None
        # Normalize unicode fraction glyphs, then strip whitespace around them.
        normalized = (value
                      .replace("½", ".5")
                      .replace("¼", ".25")
                      .replace("¾", ".75"))
        normalized = re.sub(r"(\d)\s+\.", r"\1.", normalized)  # '1 .5' -> '1.5'
        match = self._ROOMS_RE.search(normalized)
        return float(match.group(1)) if match else None

    def clean_location(self, value: str) -> tuple[str, str]:
        """'Zurich, ZH' -> ('Zurich', 'ZH'). Requirement coverage: tuple."""
        if not value:
            return ("", "")
        match = self._LOCATION_RE.match(value.strip())
        if not match:
            return (value.strip(), "")
        return (match.group("city").strip(), (match.group("canton") or "").strip())

    def clean_features(self, features: Iterable[str]) -> list[str]:
        """Normalize the features list: lowercased, stripped, unique, sorted."""
        # Requirement coverage: set (deduplication) + list + loop
        seen: set[str] = set()
        cleaned: list[str] = []
        for f in features:
            norm = f.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                cleaned.append(norm)
        return sorted(cleaned)

    def clean_record(self, raw: dict) -> dict:
        """Apply all cleaners to a single raw listing dict."""
        city, canton = self.clean_location(raw.get("location", ""))
        features = self.clean_features(raw.get("features", []))
        # Requirement coverage: conditional binary flags derived from features.
        # Substring match handles Flatfox's compound labels like "Balcony/Garden".
        has_balcony = 1 if any("balcony" in f for f in features) else 0
        has_parking = 1 if any(k in f for f in features for k in ("parking", "garage")) else 0
        return {
            "title": raw.get("title", "").strip(),
            "price": self.clean_price(raw.get("price", "")),
            "size": self.clean_size(raw.get("size", "")),
            "rooms": self.clean_rooms(raw.get("rooms", "")),
            "city": city,
            "canton": canton,
            "zip_code": str(raw.get("zip_code", "") or "").strip(),
            "features": features,
            "balcony": has_balcony,
            "parking": has_parking,
            "description": raw.get("description", "").strip(),
            "listing_url": raw.get("listing_url", "").strip(),
        }


# Requirement coverage: procedural function operating on a list.
def clean_records(rows: list[dict]) -> list[dict]:
    """Clean many records in one go; drops rows without a usable price."""
    cleaner = DataCleaner()
    out: list[dict] = []
    for row in rows:
        cleaned = cleaner.clean_record(row)
        if cleaned["price"] is not None and cleaned["size"] is not None:
            out.append(cleaned)
    return out
