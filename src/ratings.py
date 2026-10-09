"""Walk-forward opponent-adjusted offensive and defensive ratings.

Each team-game is one observation:

    off_ppa = off_rating[team] + def_rating[opponent] + hfa * home + intercept

``def_rating`` is PPA allowed, so a better defense is lower (more negative).
That is the same split the NFL and NBA models use. Offensive output equals a
team's own offensive rating minus the opponent's defensive quality once that
quality is defined as minus ``def_rating``. Season-end rank uses
``off_rating - def_rating``.

Ratings for week W use only games from weeks before W. The preseason prior is
fixed for the whole season; ridge shrinks the adjustment away from that prior
toward zero, so the prior fades as games accumulate instead of on a hand-set
schedule.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load import load_games, load_returning_production, load_team_ppa

DATA_DIR = ROOT / "data"
FIRST_SEASON = 2016

# A game five weeks older counts half as much. Five weeks still separates
# September from November inside a 12-15 game season. The NFL model's 10-week
# half-life would barely decay inside one CFB season, and these fits do not
# pool previous seasons (that information is in the prior instead).
HALF_LIFE_WEEKS = 5

# NBA ratings use alpha=10. NFL ratings use alpha=1 on a similar per-play
# scale. Alpha stays at 10 here: with about 12 games, a coefficient is kept
# at n / (n + alpha) of its unshrunk value, so the preseason prior still
# matters in September and gives way by the end of the year.
RIDGE_ALPHA = 10.0

WEEKLY_COLUMNS = ["season", "week", "team", "off_rating", "def_rating"]
FINAL_COLUMNS = ["season", "team", "off_rating", "def_rating"]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _cache_path(kind: str, year: int) -> Path:
    return DATA_DIR / f"ratings_{kind}_{year}.csv"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        return pd.read_csv(path)
    return None


def _write_cache(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _returning_weights(year: int) -> dict[str, float]:
    """Overall returning production as a mixing weight in [0, 1].

    ``PlayersApi.get_returning_production`` stores this in ``percentPPA``.
    In the cached responses it is already a fraction (medians around 0.4-0.7),
    not a 0-100 percentage, so it is not divided by 100. There is no separate
    offensive or defensive returning-production field; the same weight is used
    for both priors. A few team-seasons fall outside [0, 1] (including a large
    negative glitch), and those weights are clipped so the prior stays between
    last year's rating and the league average.
    """
    rp = load_returning_production(year)
    if rp.empty or "percentPPA" not in rp.columns:
        return {}
    weights = rp.drop_duplicates("team").set_index("team")["percentPPA"]
    weights = pd.to_numeric(weights, errors="coerce").clip(0.0, 1.0)
    return {team: float(value) for team, value in weights.dropna().items()}


def _league_average(year: int) -> tuple[float, float]:
    """Mean offensive and defensive rating, one number per side.

    After 2016 this is the mean of the previous season's final ratings. The
    2016 season has no previous ratings, so the mean is taken from that
    season's per-team regular-season PPA.
    """
    if year > FIRST_SEASON:
        previous = get_final_ratings(year - 1)
        return float(previous["off_rating"].mean()), float(previous["def_rating"].mean())

    ppa = load_team_ppa(year)
    ppa = ppa.loc[ppa["seasonType"].astype(str).eq("regular")].copy()
    ppa["offense.overall"] = pd.to_numeric(ppa["offense.overall"], errors="coerce")
    ppa["defense.overall"] = pd.to_numeric(ppa["defense.overall"], errors="coerce")
    team_means = ppa.groupby("team")[["offense.overall", "defense.overall"]].mean()
    return float(team_means["offense.overall"].mean()), float(team_means["defense.overall"].mean())


def _priors(
    year: int,
    teams: list[str],
    league_off: float,
    league_def: float,
) -> pd.DataFrame:
    """Preseason offensive and defensive priors on the PPA scale.

    prior = returning_pct * last_final + (1 - returning_pct) * league_average

    2016 has no previous season, so every team starts at the league average.
    A team missing returning production, or missing a final rating from last
    year, also falls back to the league average.
    """
    previous = None if year <= FIRST_SEASON else get_final_ratings(year - 1)
    last_off: dict[str, float] = {}
    last_def: dict[str, float] = {}
    if previous is not None:
        last_off = previous.set_index("team")["off_rating"].astype(float).to_dict()
        last_def = previous.set_index("team")["def_rating"].astype(float).to_dict()
    weights = {} if year <= FIRST_SEASON else _returning_weights(year)

    off_values = []
    def_values = []
    for team in teams:
        weight = weights.get(team)
        previous_off = last_off.get(team)
        previous_def = last_def.get(team)
        if weight is None or previous_off is None or previous_def is None:
            off_values.append(league_off)
            def_values.append(league_def)
        else:
            off_values.append(weight * previous_off + (1.0 - weight) * league_off)
            def_values.append(weight * previous_def + (1.0 - weight) * league_def)

    return pd.DataFrame(
        {"prior_off": off_values, "prior_def": def_values},
        index=pd.Index(teams, name="team"),
    )


def _observations(year: int) -> tuple[pd.DataFrame, list[str]]:
    """Completed regular-season team-games and the FBS teams to publish.

    Each row is one offense: that team's ``offense.overall`` against the
    opponent. When the opponent has no row of their own (an FCS opponent),
    the FBS team's ``defense.overall`` is added as the opponent's offensive
    PPA so the FBS defense still updates. Games that already have both team
    rows are not doubled.
    """
    games = load_games(year)
    ppa = load_team_ppa(year)
    fbs_teams = sorted(
        set(games.loc[games["homeClassification"].astype(str).eq("fbs"), "homeTeam"])
        | set(games.loc[games["awayClassification"].astype(str).eq("fbs"), "awayTeam"])
    )
    if games.empty or ppa.empty:
        return pd.DataFrame(columns=["week", "team", "opponent", "off_ppa", "home"]), fbs_teams

    completed = games.loc[_as_bool(games["completed"]), ["id", "week", "homeTeam", "awayTeam", "neutralSite"]]
    stats = ppa.loc[
        ppa["seasonType"].astype(str).eq("regular"),
        ["gameId", "team", "opponent", "offense.overall", "defense.overall"],
    ].copy()
    merged = stats.merge(completed, left_on="gameId", right_on="id", how="inner")
    merged["offense.overall"] = pd.to_numeric(merged["offense.overall"], errors="coerce")
    merged["defense.overall"] = pd.to_numeric(merged["defense.overall"], errors="coerce")
    merged = merged.dropna(subset=["offense.overall"])
    if merged.empty:
        return pd.DataFrame(columns=["week", "team", "opponent", "off_ppa", "home"]), fbs_teams

    neutral = _as_bool(merged["neutralSite"]).to_numpy()
    home_team = merged["homeTeam"].to_numpy()

    def home_indicator(offensive_team: pd.Series) -> np.ndarray:
        return (offensive_team.to_numpy() == home_team) & ~neutral

    offense = pd.DataFrame(
        {
            "week": merged["week"].to_numpy(dtype=int),
            "team": merged["team"].to_numpy(),
            "opponent": merged["opponent"].to_numpy(),
            "off_ppa": merged["offense.overall"].to_numpy(dtype=float),
            "home": home_indicator(merged["team"]).astype(float),
        }
    )

    present = merged[["gameId", "team"]].drop_duplicates()
    present["seen"] = True
    opponent_key = merged[["gameId", "opponent"]].rename(columns={"opponent": "team"})
    seen = opponent_key.merge(present, on=["gameId", "team"], how="left")["seen"].notna().to_numpy()
    extra_source = merged.loc[~seen].dropna(subset=["defense.overall"])
    if not extra_source.empty:
        extra_neutral = _as_bool(extra_source["neutralSite"]).to_numpy()
        extra_home = extra_source["homeTeam"].to_numpy()
        extra = pd.DataFrame(
            {
                "week": extra_source["week"].to_numpy(dtype=int),
                "team": extra_source["opponent"].to_numpy(),
                "opponent": extra_source["team"].to_numpy(),
                "off_ppa": extra_source["defense.overall"].to_numpy(dtype=float),
                "home": (
                    (extra_source["opponent"].to_numpy() == extra_home) & ~extra_neutral
                ).astype(float),
            }
        )
        offense = pd.concat([offense, extra], ignore_index=True)

    return offense, fbs_teams


def _time_weights(weeks: np.ndarray) -> np.ndarray:
    """Exponential decay. The newest week in the sample has weight 1."""
    age = weeks.max() - weeks
    return 0.5 ** (age / HALF_LIFE_WEEKS)


def _fit_ratings(obs: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Ridge on deviations from each team's prior, then add the prior back.

    Sklearn's Ridge only shrinks toward zero. Fitting
    ``off_ppa - prior_off[team] - prior_def[opponent]`` makes zero the
    "no adjustment" point, which is this team's own prior rather than a
    single league constant.
    """
    base = priors.copy()
    if obs.empty:
        base.attrs["hfa"] = 0.0
        return base

    names = sorted(set(obs["team"]) | set(obs["opponent"]))
    missing = [name for name in names if name not in base.index]
    if missing:
        fill = pd.DataFrame(
            {
                "prior_off": base["prior_off"].mean(),
                "prior_def": base["prior_def"].mean(),
            },
            index=pd.Index(missing, name="team"),
        )
        base = pd.concat([base, fill])

    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    rows = len(obs)
    design = np.zeros((rows, 2 * n + 1))
    row_index = np.arange(rows)
    design[row_index, obs["team"].map(index).to_numpy()] = 1.0
    design[row_index, n + obs["opponent"].map(index).to_numpy()] = 1.0
    design[:, -1] = obs["home"].to_numpy(dtype=float)

    baseline = (
        base.loc[obs["team"].to_numpy(), "prior_off"].to_numpy()
        + base.loc[obs["opponent"].to_numpy(), "prior_def"].to_numpy()
    )
    target = obs["off_ppa"].to_numpy(dtype=float) - baseline
    weights = _time_weights(obs["week"].to_numpy(dtype=int))
    weights = weights / weights.mean()

    model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
    model.fit(design, target, sample_weight=weights)
    coef = model.coef_

    rated = base.copy()
    rated.loc[names, "prior_off"] = base.loc[names, "prior_off"].to_numpy() + coef[:n]
    rated.loc[names, "prior_def"] = base.loc[names, "prior_def"].to_numpy() + coef[n : 2 * n]
    # Last column is the home dummy. Same units as off_rating / def_rating (PPA).
    rated.attrs["hfa"] = float(coef[-1])
    return rated


def _rating_frame(year: int, ratings: pd.DataFrame, teams: list[str], week: int | None) -> pd.DataFrame:
    selected = ratings.reindex(teams)
    frame = pd.DataFrame(
        {
            "season": year,
            "team": teams,
            "off_rating": selected["prior_off"].to_numpy(),
            "def_rating": selected["prior_def"].to_numpy(),
        }
    )
    if week is not None:
        frame.insert(1, "week", week)
    return frame


def _season_inputs(year: int) -> tuple[pd.DataFrame, list[str], pd.DataFrame, list[int]]:
    obs, fbs_teams = _observations(year)
    league_off, league_def = _league_average(year)
    everyone = sorted(set(fbs_teams) | set(obs["team"]) | set(obs["opponent"]))
    priors = _priors(year, everyone, league_off, league_def)

    games = load_games(year)
    scheduled = sorted(int(week) for week in games["week"].dropna().unique())
    completed = games.loc[_as_bool(games["completed"]), "week"]
    last_completed = int(completed.max()) if len(completed) else 0
    # Week W is knowable once every earlier week is done. Do not emit later
    # weeks of an in-progress season; those ratings would skip unplayed games.
    weeks = [week for week in scheduled if week <= last_completed + 1]
    return obs, fbs_teams, priors, weeks


def _compute_weekly(year: int) -> pd.DataFrame:
    obs, fbs_teams, priors, weeks = _season_inputs(year)
    if not weeks:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)

    frames = []
    for week in weeks:
        history = obs.loc[obs["week"] < week]
        ratings = _fit_ratings(history, priors)
        frames.append(_rating_frame(year, ratings, fbs_teams, week))
    return pd.concat(frames, ignore_index=True)[WEEKLY_COLUMNS]


def _compute_final(year: int) -> pd.DataFrame:
    obs, fbs_teams, priors, _weeks = _season_inputs(year)
    ratings = _fit_ratings(obs, priors)
    return _rating_frame(year, ratings, fbs_teams, week=None)[FINAL_COLUMNS]


def home_field_term() -> float:
    """Home-field coefficient from the ratings ridge, in PPA per game.

    Each fit already estimates this as the home-dummy coefficient. Weekly
    ratings do not store it, so this averages the end-of-sample coefficient
    across seasons (weighted by team-games) and caches that one constant.
    Margin conversion reuses the cached value instead of refitting.
    """
    path = DATA_DIR / "home_field.json"
    if path.is_file():
        return float(json.loads(path.read_text(encoding="utf-8"))["home_field_term"])

    current_year = datetime.now().year
    weighted_sum = 0.0
    weight = 0
    for year in range(FIRST_SEASON, current_year + 1):
        obs, _fbs_teams, priors, _weeks = _season_inputs(year)
        if obs.empty:
            continue
        ratings = _fit_ratings(obs, priors)
        n_games = len(obs)
        weighted_sum += float(ratings.attrs["hfa"]) * n_games
        weight += n_games
    if weight == 0:
        raise RuntimeError("No games available to estimate the home-field term.")

    value = weighted_sum / weight
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"home_field_term": value}), encoding="utf-8")
    return value


def get_weekly_ratings(year: int) -> pd.DataFrame:
    """Opponent-adjusted ratings entering each week of ``year``.

    One row per FBS team per week, using only games from earlier weeks.
    Cached at ``data/ratings_weekly_<year>.csv``.
    """
    path = _cache_path("weekly", year)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    return _write_cache(path, _compute_weekly(year))


def get_final_ratings(year: int) -> pd.DataFrame:
    """Ratings after every completed regular-season game in ``year``.

    This is the next season's prior input. Cached at
    ``data/ratings_final_<year>.csv``.
    """
    path = _cache_path("final", year)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    return _write_cache(path, _compute_final(year))


def _print_ends(year: int, final: pd.DataFrame) -> None:
    ranked = final.copy()
    ranked["net"] = ranked["off_rating"] - ranked["def_rating"]
    ranked = ranked.sort_values(["net", "team"], ascending=[False, True])
    show = ranked[["team", "off_rating", "def_rating", "net"]].round(3)
    print(f"\n{year} final ratings by off_rating - def_rating")
    print("Top 10")
    print(show.head(10).to_string(index=False))
    print("Bottom 10")
    print(show.tail(10).to_string(index=False))


def main() -> None:
    current_year = datetime.now().year
    for year in range(FIRST_SEASON, current_year + 1):
        weekly = get_weekly_ratings(year)
        final = get_final_ratings(year)
        print(
            f"{year}: weekly rows={len(weekly)}, "
            f"teams={final['team'].nunique()}, weeks={weekly['week'].nunique() if len(weekly) else 0}"
        )
        _print_ends(year, final)


if __name__ == "__main__":
    main()
