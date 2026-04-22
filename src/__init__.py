"""Swiss Housing Price Analysis — source package.

Modules:
    scraper        Swiss housing listings scraper (OOP + requests + BeautifulSoup).
    cleaning       Regex-based data cleaning (OOP).
    database       SQLite persistence and SQL queries (OOP).
    analysis       Pandas analysis (procedural).
    statistics     Correlation + hypothesis tests with p-values (procedural).
    visualization  Seaborn/matplotlib charts (procedural).
    llm_helper     OpenAI-backed feature extraction with regex fallback (OOP).
    pipeline       End-to-end orchestration: scrape → clean → LLM → SQLite → analyze.
"""
