"""SQLite storage + SQL queries.

Requirement coverage: OOP class, SQL queries (incl. window functions, CTEs,
CASE expressions, aggregations), context manager, conditional schema, procedural
helpers. SQLite 3.25+ supports window functions; we use them to demonstrate
advanced SQL beyond basic GROUP BY aggregations.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_DB_PATH = os.getenv("DB_PATH", "./housing.db")


class HousingDatabase:
    """Thin wrapper around sqlite3 for the `apartments` table.

    Exposes a catalog of named SQL queries — each catalog method returns a
    DataFrame so the dashboard can both display the SQL string AND its results.
    """

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
        # Create a helper view so the catalog queries below can reference
        # chf_per_m2 without recomputing it everywhere.
        self.conn.executescript(f"""
            DROP VIEW IF EXISTS v_apartments;
            CREATE VIEW v_apartments AS
            SELECT *,
                   ROUND(CAST(price AS REAL) / NULLIF(size, 0), 2) AS chf_per_m2
            FROM {self.TABLE};
        """)
        self.conn.commit()
        return len(df)

    # -- introspection ----------------------------------------------------

    def schema(self) -> pd.DataFrame:
        """Return the `apartments` table schema as a DataFrame for the dashboard."""
        return self.query(f"PRAGMA table_info({self.TABLE})")

    def row_count(self) -> int:
        cur = self.conn.execute(f"SELECT COUNT(*) FROM {self.TABLE}")
        return int(cur.fetchone()[0])

    # -- reads ------------------------------------------------------------

    def query(self, sql: str) -> pd.DataFrame:
        """Run an arbitrary SELECT and return a DataFrame."""
        return pd.read_sql_query(sql, self.conn)

    # ----- core aggregations (kept for backwards compatibility) ---------

    def avg_price_by_rooms(self) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT rooms,
                   ROUND(AVG(price), 0) AS avg_price,
                   ROUND(AVG(price / NULLIF(size, 0)), 2) AS avg_chf_per_m2,
                   COUNT(*) AS n
            FROM {self.TABLE}
            GROUP BY rooms
            ORDER BY rooms
            """
        )

    def avg_price_by_location(self, min_n: int = 1) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT city,
                   ROUND(AVG(price), 0) AS avg_price,
                   ROUND(AVG(price / NULLIF(size, 0)), 2) AS avg_chf_per_m2,
                   ROUND(AVG(size), 1) AS avg_size,
                   COUNT(*) AS n
            FROM {self.TABLE}
            GROUP BY city
            HAVING COUNT(*) >= {int(min_n)}
            ORDER BY avg_price DESC
            """
        )

    def avg_price_by_balcony(self) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT CASE balcony WHEN 1 THEN 'with balcony' ELSE 'no balcony' END AS balcony_status,
                   ROUND(AVG(price), 0) AS avg_price,
                   ROUND(MIN(price), 0) AS min_price,
                   ROUND(MAX(price), 0) AS max_price,
                   COUNT(*) AS n
            FROM {self.TABLE}
            GROUP BY balcony
            """
        )

    def top_expensive(self, limit: int = 5) -> pd.DataFrame:
        return self.query(
            f"""
            SELECT title, price, size, rooms, city,
                   ROUND(price / NULLIF(size, 0), 2) AS chf_per_m2,
                   listing_url
            FROM {self.TABLE}
            ORDER BY price DESC
            LIMIT {int(limit)}
            """
        )

    # ----- new advanced queries -----------------------------------------

    def top_cheapest(self, limit: int = 10) -> pd.DataFrame:
        """Cheapest listings — paired with top_expensive for the dashboard."""
        return self.query(
            f"""
            SELECT title, price, size, rooms, city,
                   ROUND(price / NULLIF(size, 0), 2) AS chf_per_m2,
                   listing_url
            FROM {self.TABLE}
            ORDER BY price ASC
            LIMIT {int(limit)}
            """
        )

    def price_distribution_by_category(self) -> pd.DataFrame:
        """Bucket listings into market tiers using NTILE — a window function.

        Demonstrates: window functions (NTILE), CTE, derived column, ordering.
        """
        return self.query(
            f"""
            WITH tiles AS (
                SELECT price, size, rooms, city,
                       NTILE(4) OVER (ORDER BY price) AS quartile
                FROM {self.TABLE}
            )
            SELECT CASE quartile
                       WHEN 1 THEN 'cheap (Q1)'
                       WHEN 2 THEN 'medium (Q2)'
                       WHEN 3 THEN 'expensive (Q3)'
                       WHEN 4 THEN 'luxury (Q4)'
                   END AS price_category,
                   COUNT(*)         AS n,
                   ROUND(MIN(price)) AS min_price,
                   ROUND(MAX(price)) AS max_price,
                   ROUND(AVG(price)) AS avg_price,
                   ROUND(AVG(size), 1) AS avg_size
            FROM tiles
            GROUP BY quartile
            ORDER BY quartile
            """
        )

    def city_ranking_with_premium(self, min_n: int = 3) -> pd.DataFrame:
        """City-by-city ranking + CHF/m² premium vs the median Zurich city.

        Demonstrates: subquery, window function (RANK), CASE, HAVING.
        """
        return self.query(
            f"""
            WITH city_stats AS (
                SELECT city,
                       COUNT(*) AS n,
                       ROUND(AVG(price), 0) AS avg_price,
                       ROUND(AVG(price / NULLIF(size, 0)), 2) AS avg_chf_per_m2
                FROM {self.TABLE}
                WHERE city IS NOT NULL AND city <> ''
                GROUP BY city
                HAVING COUNT(*) >= {int(min_n)}
            ),
            ranked AS (
                SELECT city, n, avg_price, avg_chf_per_m2,
                       RANK() OVER (ORDER BY avg_chf_per_m2 DESC) AS rank_chf_per_m2,
                       (SELECT AVG(avg_chf_per_m2) FROM city_stats) AS market_avg
                FROM city_stats
            )
            SELECT rank_chf_per_m2 AS rank,
                   city, n, avg_price, avg_chf_per_m2,
                   ROUND(100.0 * (avg_chf_per_m2 - market_avg) / market_avg, 1)
                       AS premium_vs_market_pct
            FROM ranked
            ORDER BY rank
            """
        )

    def feature_premium(self) -> pd.DataFrame:
        """Side-by-side balcony + parking + furnished premia in one table.

        Demonstrates: UNION ALL across three CASE-driven aggregations.
        """
        return self.query(
            f"""
            SELECT 'balcony' AS feature,
                   ROUND(AVG(CASE WHEN balcony = 1 THEN price END), 0) AS avg_with,
                   ROUND(AVG(CASE WHEN balcony = 0 THEN price END), 0) AS avg_without,
                   ROUND(AVG(CASE WHEN balcony = 1 THEN price END)
                         - AVG(CASE WHEN balcony = 0 THEN price END), 0) AS difference
            FROM {self.TABLE}
            UNION ALL
            SELECT 'parking',
                   ROUND(AVG(CASE WHEN parking = 1 THEN price END), 0),
                   ROUND(AVG(CASE WHEN parking = 0 THEN price END), 0),
                   ROUND(AVG(CASE WHEN parking = 1 THEN price END)
                         - AVG(CASE WHEN parking = 0 THEN price END), 0)
            FROM {self.TABLE}
            UNION ALL
            SELECT 'furnished (LLM)',
                   ROUND(AVG(CASE WHEN llm_furnished = 1 THEN price END), 0),
                   ROUND(AVG(CASE WHEN llm_furnished = 0 THEN price END), 0),
                   ROUND(AVG(CASE WHEN llm_furnished = 1 THEN price END)
                         - AVG(CASE WHEN llm_furnished = 0 THEN price END), 0)
            FROM {self.TABLE}
            """
        )

    def running_avg_price_by_size(self) -> pd.DataFrame:
        """Cumulative average rent as listings are walked from smallest to largest.

        Demonstrates: window function with frame (AVG ... OVER ORDER BY).
        Useful for spotting where price-per-m² stabilises with apartment size.
        """
        return self.query(
            f"""
            SELECT size, price,
                   ROUND(AVG(price) OVER (
                       ORDER BY size
                       ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                   ), 0) AS rolling_avg_10
            FROM {self.TABLE}
            WHERE size IS NOT NULL
            ORDER BY size
            """
        )

    # -- catalog metadata for the dashboard -------------------------------

    QUERY_CATALOG: list[tuple[str, str]] = [
        ("Average price by room count",
         "GROUP BY rooms — basic aggregation"),
        ("Average price by city (top by mean rent)",
         "GROUP BY city HAVING COUNT(*) >= N — filtered aggregation"),
        ("Balcony vs no-balcony pricing",
         "CASE expression to label boolean groups"),
        ("Top 10 most expensive listings",
         "ORDER BY price DESC LIMIT N — sorted scan"),
        ("Top 10 cheapest listings",
         "ORDER BY price ASC LIMIT N — sorted scan"),
        ("Price tiers via NTILE window function",
         "WITH … NTILE(4) OVER (ORDER BY price) — quartile bucketing"),
        ("City ranking + premium vs market average",
         "Two-step CTE + RANK() OVER + subquery — demonstrates advanced SQL"),
        ("Side-by-side feature premium (balcony / parking / furnished)",
         "UNION ALL of CASE-driven AVGs — wide-format comparison"),
        ("Rolling 10-listing average price by size",
         "AVG(...) OVER (ORDER BY size ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)"),
    ]


# -- procedural helper ----------------------------------------------------

def persist_dataframe(df: pd.DataFrame, db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Open a DB, save df, close. Returns row count."""
    with HousingDatabase(db_path) as db:
        return db.save(df)
