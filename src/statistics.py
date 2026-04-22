"""Statistical testing with p-values.

Requirement coverage: Pearson correlation, t-test, p-value interpretation,
procedural functions, tuples returned from scipy, conditionals, dict output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_ind

ALPHA: float = 0.05  # Requirement coverage: standard significance level.


@dataclass
class TestResult:
    """Generic container for a named statistical test result."""
    name: str
    statistic: float
    p_value: float
    interpretation: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "significant": self.p_value < ALPHA,
            "interpretation": self.interpretation,
        }


def _interpret(p_value: float, effect_desc: str) -> str:
    # Requirement coverage: conditional, p-value interpretation string.
    if p_value < ALPHA:
        return f"p = {p_value:.4f} < {ALPHA} — {effect_desc} is statistically significant."
    return f"p = {p_value:.4f} >= {ALPHA} — no statistically significant evidence of {effect_desc}."


def correlation_price_size(df: pd.DataFrame) -> TestResult:
    """Pearson correlation between size and price."""
    sub = df[["size", "price"]].dropna()
    if len(sub) < 3:
        return TestResult("pearson_size_price", float("nan"), float("nan"),
                          "Not enough data for correlation.")
    # Requirement coverage: scipy returns a tuple (tuple usage).
    corr, p_value = pearsonr(sub["size"], sub["price"])
    return TestResult(
        name="pearson_size_price",
        statistic=float(corr),
        p_value=float(p_value),
        interpretation=_interpret(p_value, f"correlation between size and price (r={corr:.3f})"),
    )


def correlation_price_rooms(df: pd.DataFrame) -> TestResult:
    """Pearson correlation between number of rooms and price."""
    sub = df[["rooms", "price"]].dropna()
    if len(sub) < 3:
        return TestResult("pearson_rooms_price", float("nan"), float("nan"),
                          "Not enough data for correlation.")
    corr, p_value = pearsonr(sub["rooms"], sub["price"])
    return TestResult(
        name="pearson_rooms_price",
        statistic=float(corr),
        p_value=float(p_value),
        interpretation=_interpret(p_value, f"correlation between rooms and price (r={corr:.3f})"),
    )


def ttest_balcony(df: pd.DataFrame) -> TestResult:
    """Independent-samples t-test: price with vs. without balcony."""
    if "balcony" not in df.columns:
        return TestResult("ttest_balcony", float("nan"), float("nan"),
                          "balcony column missing.")
    group1 = df.loc[df["balcony"] == 1, "price"].dropna()
    group0 = df.loc[df["balcony"] == 0, "price"].dropna()
    # Requirement coverage: conditional guard on sample size.
    if len(group1) < 2 or len(group0) < 2:
        return TestResult("ttest_balcony", float("nan"), float("nan"),
                          "Not enough data in both groups for a t-test.")
    t_stat, p_value = ttest_ind(group1, group0, equal_var=False)
    direction = "higher" if group1.mean() > group0.mean() else "lower"
    return TestResult(
        name="ttest_balcony",
        statistic=float(t_stat),
        p_value=float(p_value),
        interpretation=_interpret(
            p_value,
            f"mean price of balcony listings being {direction} than non-balcony "
            f"(CHF {group1.mean():.0f} vs CHF {group0.mean():.0f})",
        ),
    )


def run_all_tests(df: pd.DataFrame) -> list[TestResult]:
    """Run the full battery of tests used in the report."""
    # Requirement coverage: list of results + loop over tests.
    tests = [correlation_price_size, correlation_price_rooms, ttest_balcony]
    return [fn(df) for fn in tests]
