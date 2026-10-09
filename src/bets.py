"""Moneyline prices, win probabilities, and quarter-Kelly stake sizes.

Same edge threshold and Kelly fraction as the NFL and NBA models. The win
probability uses the residual SD from the margin regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load import load_betting_lines
from src.margin import fit_margin_model

KELLY_FRACTION = 0.25  # quarter-Kelly
MIN_EDGE = 0.02


def american_to_decimal(odds: int) -> float:
    """American odds as a decimal payout, including the stake."""
    if odds > 0:
        return 1 + odds / 100
    if odds < 0:
        return 1 + 100 / abs(odds)
    raise ValueError("American odds of 0 are not a valid price")


def implied_prob(odds: int) -> float:
    """Book implied win probability from American odds, vig still included."""
    return 1 / american_to_decimal(odds)


def win_prob(predicted_margin: float, residual_sd: float) -> float:
    """P(home wins) = P(actual margin > 0) under Normal(predicted_margin, residual_sd)."""
    return float(1 - norm.cdf(0, loc=predicted_margin, scale=residual_sd))


def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    """Quarter-Kelly stake as a fraction of bankroll. Negative edge stakes 0."""
    if decimal_odds <= 1:
        return 0.0
    full_kelly = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
    return max(0.0, full_kelly * KELLY_FRACTION)


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def select_moneyline(betting_lines_for_game: pd.DataFrame) -> dict | None:
    """Home and away American moneylines for one game.

    Prefer the provider named ``consensus`` when that row has a moneyline.
    Otherwise average the home and away prices across providers that posted
    one. Return None when the game has no moneyline at all.
    """
    if betting_lines_for_game is None or betting_lines_for_game.empty:
        return None
    if "homeMoneyline" not in betting_lines_for_game.columns or "awayMoneyline" not in betting_lines_for_game.columns:
        return None

    frame = betting_lines_for_game.copy()
    frame["homeMoneyline"] = pd.to_numeric(frame["homeMoneyline"], errors="coerce")
    frame["awayMoneyline"] = pd.to_numeric(frame["awayMoneyline"], errors="coerce")
    usable = frame.dropna(subset=["homeMoneyline", "awayMoneyline"], how="all")
    if usable.empty:
        return None

    if "provider" in usable.columns:
        consensus = usable.loc[usable["provider"].astype(str).str.strip().str.lower().eq("consensus")]
        consensus = consensus.dropna(subset=["homeMoneyline", "awayMoneyline"], how="all")
        if not consensus.empty:
            row = consensus.iloc[0]
            return {
                "home_ml": _optional_float(row["homeMoneyline"]),
                "away_ml": _optional_float(row["awayMoneyline"]),
            }

    chosen = {
        "home_ml": _optional_float(usable["homeMoneyline"].mean(skipna=True)),
        "away_ml": _optional_float(usable["awayMoneyline"].mean(skipna=True)),
    }
    if chosen["home_ml"] is None and chosen["away_ml"] is None:
        return None
    return chosen


def main() -> None:
    fit = fit_margin_model()
    residual_sd = float(fit["residual_sd"])
    print(f"KELLY_FRACTION = {KELLY_FRACTION}")
    print(f"MIN_EDGE = {MIN_EDGE}")
    print(f"residual_sd = {residual_sd:.2f}")

    # Model probabilities are made up and printed so the Kelly numbers can be
    # checked by hand. +150 implies 0.40; -200 implies 0.667.
    examples = (
        ("underdog", 150, 0.50),
        ("favorite", -200, 0.75),
    )
    for label, odds, model_prob in examples:
        decimal_odds = american_to_decimal(odds)
        print(f"\n{label} {odds:+d}")
        print(f"implied_prob = {implied_prob(odds):.3f}")
        print(
            f"kelly_fraction(model_prob={model_prob:.2f}, decimal_odds={decimal_odds:.3f})"
            f" = {kelly_fraction(model_prob, decimal_odds):.4f}"
        )

    # Consensus rows in the cached lines are spread-only, so a real game falls
    # through to the average. A second frame shows consensus winning when it
    # actually has a moneyline.
    lines = load_betting_lines(2024)
    priced = lines.dropna(subset=["homeMoneyline", "awayMoneyline"])
    game_id = priced.groupby("id").filter(lambda rows: len(rows) >= 2)["id"].iloc[0]
    print(f"\naveraged moneyline for game {game_id}: {select_moneyline(lines.loc[lines['id'] == game_id])}")
    consensus_demo = pd.DataFrame(
        {
            "provider": ["consensus", "DraftKings"],
            "homeMoneyline": [-150, -170],
            "awayMoneyline": [130, 145],
        }
    )
    print(f"consensus moneyline preferred: {select_moneyline(consensus_demo)}")


if __name__ == "__main__":
    main()
