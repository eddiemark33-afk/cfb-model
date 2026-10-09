"""Does the rating model add anything once the market price is already known?

A logistic regression predicts the home-team win from model_prob and the
vig-free book probability together. Statsmodels is not installed here, so
the model_prob coefficient is judged with a 1,000-draw bootstrap: a 90%
interval that lies entirely above zero is treated as a real positive signal.
Only then is that fitted blend bet against the market.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibrate import backtest, build_dataset, profit_concentration

# Filled by test_incremental_signal. blended_prob reads these fitted weights.
_FIT: dict[str, float] = {}


def _logistic() -> LogisticRegression:
    # C=inf is unpenalized maximum likelihood, the same fit statsmodels reports.
    return LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)


def _coefficients(features: np.ndarray, outcome: np.ndarray) -> tuple[float, float, float]:
    model = _logistic()
    model.fit(features, outcome)
    return float(model.intercept_[0]), float(model.coef_[0, 0]), float(model.coef_[0, 1])


def test_incremental_signal() -> dict:
    """Logistic regression of the home win on the model and the book together.

    Returns the model_prob coefficient, its 90% bootstrap interval, and
    whether that interval lies entirely above zero.
    """
    data = build_dataset()
    features = data[["model_prob", "book_home_prob_novig"]].to_numpy(dtype=float)
    outcome = data["actual_home_win"].to_numpy(dtype=int)
    intercept, model_coef, book_coef = _coefficients(features, outcome)

    rng = np.random.default_rng(0)
    n_rows = len(outcome)
    draws = []
    for _ in range(1000):
        sample = rng.integers(0, n_rows, n_rows)
        if len(np.unique(outcome[sample])) < 2:
            continue
        try:
            _, draw, _ = _coefficients(features[sample], outcome[sample])
        except ValueError:
            continue
        draws.append(draw)
    low, high = (float(x) for x in np.percentile(draws, [5, 95]))
    meaningful_positive = low > 0
    _FIT.clear()
    _FIT.update(intercept=intercept, model_coef=model_coef, book_coef=book_coef)

    result = {
        "model_coef": model_coef,
        "book_coef": book_coef,
        "intercept": intercept,
        "interval_low": low,
        "interval_high": high,
        "n_bootstrap": len(draws),
        "meaningful_positive": meaningful_positive,
    }
    print("=== Incremental signal: actual_home_win ~ model_prob + book_home_prob_novig ===")
    print(f"  games: {n_rows}")
    print(f"  model_prob coefficient: {model_coef:.3f}")
    print(f"  book_home_prob_novig coefficient: {book_coef:.3f}")
    print(f"  model_prob 90% bootstrap interval: {low:.3f} to {high:.3f} ({len(draws)} resamples)")
    if meaningful_positive:
        print("  model_prob has a positive relationship with the outcome after the book price is included.")
    elif high < 0:
        print("  model_prob's 90% interval is entirely below zero. That is not a usable positive signal.")
    else:
        print("  model_prob's 90% interval includes zero, so its effect is indistinguishable from noise.")
    return result


def blended_prob(model_prob: float, book_prob: float) -> float:
    """Home win probability from the fitted logistic blend of model and book."""
    if not _FIT:
        raise RuntimeError("Call test_incremental_signal() before blended_prob().")
    score = _FIT["intercept"] + _FIT["model_coef"] * model_prob + _FIT["book_coef"] * book_prob
    return float(1.0 / (1.0 + np.exp(-score)))


def _no_independent_signal() -> None:
    print(
        "The model's rating-based signal does not carry independent predictive "
        "information beyond what the betting market already prices in, based on this test."
    )


def main() -> None:
    result = test_incremental_signal()
    if result["interval_high"] < 0:
        print("Step 2 is skipped because that independent coefficient is negative.")
        return
    if not result["meaningful_positive"]:
        _no_independent_signal()
        return

    data = build_dataset()
    blended = [
        blended_prob(float(model), float(book))
        for model, book in zip(data["model_prob"], data["book_home_prob_novig"])
    ]
    print("\n=== Betting the blended probability instead of the raw model ===")
    backtest(probabilities=blended, prob_name="blended prob")
    profit_concentration(edge_threshold=0.02, probabilities=blended)
    profit_concentration(edge_threshold=0.05, probabilities=blended)


if __name__ == "__main__":
    main()
