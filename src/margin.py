"""Stage 3: turn opponent-adjusted ratings into a predicted point margin.

raw_margin is in PPA, not points. An OLS fit of actual home margin on that
raw number supplies the scale and intercept, same correction the NFL and NBA
margin models use after ridge shrinkage. The residual SD is the scale of the
miss, which is what a later win probability needs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load import load_games
from src.ratings import get_weekly_ratings, home_field_term

DATA_DIR = ROOT / "data"
MARGINS_PATH = DATA_DIR / "margins.csv"
MODEL_PATH = DATA_DIR / "margin_model.json"
MARGIN_COLUMNS = ["season", "week", "home", "away", "raw_margin", "actual_margin"]


def raw_margin(
    home_off: float,
    home_def: float,
    away_off: float,
    away_def: float,
    home_field_term: float,
) -> float:
    """Predicted home margin in rating units, before the points correction.

    ``home_def`` and ``away_def`` are defensive quality, higher meaning a
    better defense. The stored ``def_rating`` is PPA allowed (lower is
    better), so the caller passes ``-def_rating``.
    """
    return (home_off - away_def) - (away_off - home_def) + home_field_term


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _covers(cached: pd.DataFrame, start_year: int, current_year: int) -> bool:
    if cached.empty:
        return False
    return int(cached["season"].min()) <= start_year and int(cached["season"].max()) >= current_year


def _compute_margins(start_year: int) -> pd.DataFrame:
    """One row per completed game whose home and away ratings are already known.

    The rating for week W is the one from get_weekly_ratings, which uses only
    games before that week: the state at the end of the previous week.
    Neutral sites get no home-field term, matching the home dummy in the
    ratings fit (0 on a neutral field).
    """
    hfa = home_field_term()
    current_year = datetime.now().year
    frames = []
    for year in range(start_year, current_year + 1):
        games = load_games(year)
        ratings = get_weekly_ratings(year)
        if games.empty or ratings.empty:
            continue

        played = games.loc[_as_bool(games["completed"])].copy()
        played["homePoints"] = pd.to_numeric(played["homePoints"], errors="coerce")
        played["awayPoints"] = pd.to_numeric(played["awayPoints"], errors="coerce")
        played = played.dropna(subset=["homePoints", "awayPoints"])
        played = played.rename(columns={"homeTeam": "home", "awayTeam": "away"})

        home_ratings = ratings.rename(
            columns={"team": "home", "off_rating": "home_off", "def_rating": "home_def"}
        )
        away_ratings = ratings.rename(
            columns={"team": "away", "off_rating": "away_off", "def_rating": "away_def"}
        )
        matched = played.merge(
            home_ratings[["season", "week", "home", "home_off", "home_def"]],
            on=["season", "week", "home"],
            how="inner",
        )
        matched = matched.merge(
            away_ratings[["season", "week", "away", "away_off", "away_def"]],
            on=["season", "week", "away"],
            how="inner",
        )
        if matched.empty:
            continue

        # def_rating is PPA allowed. Negate it so raw_margin subtracts
        # defensive quality, and a good defense lowers the opponent's margin.
        field = np.where(_as_bool(matched["neutralSite"]), 0.0, hfa)
        matched["raw_margin"] = [
            raw_margin(home_off, -home_def, away_off, -away_def, field_term)
            for home_off, home_def, away_off, away_def, field_term in zip(
                matched["home_off"],
                matched["home_def"],
                matched["away_off"],
                matched["away_def"],
                field,
            )
        ]
        matched["actual_margin"] = matched["homePoints"] - matched["awayPoints"]
        frames.append(matched[MARGIN_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=MARGIN_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["season", "week", "home", "away"]).reset_index(drop=True)


def walk_forward_margins(start_year: int = 2016) -> pd.DataFrame:
    """Walk-forward raw and actual home margins from ``start_year`` on.

    Cached at ``data/margins.csv``. Games are skipped when either team has
    no rating for that week (an FCS opponent, or a team not yet in the table).
    """
    current_year = datetime.now().year
    if MARGINS_PATH.is_file():
        cached = pd.read_csv(MARGINS_PATH)
        if _covers(cached, start_year, current_year):
            return cached.loc[cached["season"] >= start_year].reset_index(drop=True)

    frame = _compute_margins(start_year)
    # Only the full 2016-through-current pull is the shared cache. A later
    # start year is a filtered view and must not replace that file.
    if len(frame) and int(frame["season"].min()) <= 2016:
        MARGINS_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(MARGINS_PATH, index=False)
    return frame


def fit_margin_model() -> dict:
    """OLS of actual home point margin on raw_margin.

    Returns scale, intercept, residual SD, and R-squared. Cached at
    ``data/margin_model.json``.
    """
    if MODEL_PATH.is_file():
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))

    margins = walk_forward_margins().dropna(subset=["raw_margin", "actual_margin"])
    raw = margins["raw_margin"].to_numpy(dtype=float)
    actual = margins["actual_margin"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(actual)), raw])
    intercept, scale = np.linalg.lstsq(design, actual, rcond=None)[0]
    fitted = intercept + scale * raw
    residual = actual - fitted
    residual_sd = float(np.std(residual))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / total) if total else 0.0

    result = {
        "scale": float(scale),
        "intercept": float(intercept),
        "residual_sd": residual_sd,
        "r_squared": r_squared,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(result), encoding="utf-8")
    return result


def main() -> None:
    margins = walk_forward_margins()
    print(f"walk-forward games: {len(margins)}")
    fit = fit_margin_model()
    print(f"\nscale (how much to multiply raw_margin by): {fit['scale']:.3f}")
    print(f"intercept (should be near 0): {fit['intercept']:.3f}")
    print(f"residual SD: {fit['residual_sd']:.2f} points")
    print(f"R-squared: {fit['r_squared']:.3f}")


if __name__ == "__main__":
    main()
