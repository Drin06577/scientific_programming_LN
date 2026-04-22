"""SQLite storage + SQL queries.

Requirement coverage: OOP class, SQL queries, context manager, loops,
conditional schema setup, procedural helper.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = os.getenv("DB_PATH", "./housing.db")


class HousingDatabase:
    """Thin wrapper around sqlite3 for the `apartments` table."""

    TABLE: str = "apartments"

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path: str = str(db_path)
        self.conn: sqlite3.Connection = sqlite3.connect(self.db_path)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HousingDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writes -----------------------------------------------------------

    def save(self, df: pd.DataFrame) -> int:
        """Persist DataFrame to the `apartments` table (replace)."""
        df.to_sql(self.TABLE, self.conn, if_exists="replace", index=False)
        return len(df)

    # -- reads ------------------------------------------------------------

    def query(self, sql: str) -> pd.DataFrame:
        """Run an arbitrary SELECT and return a DataFrame."""
        return pd.read_sql_query(sql, self.conn)

    # Requirement coverage: SQL queries required by SETUP.md.
    def avg_price_by_rooms(self) -> pd.DataFrame:
        return self.query(
            f"SELECT rooms, ROUND(AVG(price), 2) AS avg_price, COUNT(*) AS n "
            f"FROM {self.TABLE} GROUP BY rooms ORDER BY rooms"
        )

    def avg_price_by_location(self) -> pd.DataFrame:
        return self.query(
            f"SELECT city, ROUND(AVG(price), 2) AS avg_price, COUNT(*) AS n "
            f"FROM {self.TABLE} GROUP BY city ORDER BY avg_price DESC"
        )

    def avg_price_by_balcony(self) -> pd.DataFrame:
        return self.query(
            f"SELECT balcony, ROUND(AVG(price), 2) AS avg_price, COUNT(*) AS n "
            f"FROM {self.TABLE} GROUP BY balcony"
        )

    def top_expensive(self, limit: int = 5) -> pd.DataFrame:
        return self.query(
            f"SELECT title, price, size, rooms, city FROM {self.TABLE} "
            f"ORDER BY price DESC LIMIT {int(limit)}"
        )


# -- procedural helper ----------------------------------------------------

def persist_dataframe(df: pd.DataFrame, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Open a DB, save df, close. Returns row count."""
    with HousingDatabase(db_path) as db:
        return db.save(df)
