"""Pull College Football Data API responses and cache them as local CSV files.

Each loader writes ``data/raw/<name>_<year>.csv`` on the first call and reads
that file on later calls so the API is not hit again.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import cfbd
import pandas as pd
from cfbd.models.division_classification import DivisionClassification
from cfbd.models.season_type import SeasonType

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _api_key() -> str:
    key = os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY is not set. Export your College Football Data API "
            "key in the CFBD_API_KEY environment variable before fetching."
        )
    return key


def _client() -> cfbd.ApiClient:
    # cfbd 5.x only attaches Authorization when Configuration.access_token is
    # set (Bearer <token>). Configuration.api_key is not applied to requests.
    return cfbd.ApiClient(cfbd.Configuration(access_token=_api_key()))


def _cache_path(name: str, year: int) -> Path:
    return RAW_DIR / f"{name}_{year}.csv"


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _plain(model: object) -> dict:
    return json.loads(json.dumps(model.to_dict(), default=_json_default))


def _frame_from_models(models: list | None) -> pd.DataFrame:
    records = [_plain(model) for model in models or []]
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def _lines_frame(games: list | None) -> pd.DataFrame:
    """One row per provider, with the game id on every row.

    ``BettingApi.get_lines`` nests every book under ``lines``. Those objects
    are expanded so a later step can pick one provider or average across books.
    Games with no posted lines are kept as a single row with empty line fields.
    """
    records = [_plain(game) for game in games or []]
    if not records:
        return pd.DataFrame()

    frame = pd.json_normalize(records)
    if "lines" not in frame.columns:
        return frame

    frame["lines"] = frame["lines"].apply(
        lambda lines: lines if isinstance(lines, list) and len(lines) > 0 else [{}]
    )
    exploded = frame.explode("lines", ignore_index=True)
    line_rows = [
        row if isinstance(row, dict) else {} for row in exploded["lines"].tolist()
    ]
    line_frame = pd.json_normalize(line_rows)
    return pd.concat(
        [exploded.drop(columns=["lines"]).reset_index(drop=True), line_frame],
        axis=1,
    )


def _cached(name: str, year: int, fetch) -> pd.DataFrame:
    path = _cache_path(name, year)
    if path.is_file():
        return pd.read_csv(path)

    frame = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_games(year: int) -> pd.DataFrame:
    """All FBS regular-season games for ``year``.

    Cached at ``data/raw/games_<year>.csv``.
    """

    def fetch() -> pd.DataFrame:
        # GamesApi.get_games accepts classification. "fbs" is a valid
        # DivisionClassification in this package, so the FBS filter is applied
        # by the API together with season_type="regular".
        games = cfbd.GamesApi(_client()).get_games(
            year=year,
            season_type=SeasonType.REGULAR,
            classification=DivisionClassification.FBS,
        )
        return _frame_from_models(games)

    return _cached("games", year, fetch)


def load_team_ppa(year: int) -> pd.DataFrame:
    """Per-game team offensive and defensive PPA for ``year``.

    Cached at ``data/raw/team_ppa_<year>.csv``. Offense and defense fields are
    flattened to columns such as ``offense.overall`` and ``defense.overall``.
    """

    def fetch() -> pd.DataFrame:
        # Substituted MetricsApi.get_predicted_points_added_by_game. This
        # package has no team_ppa / get_team_ppa method. That call is the
        # per-game team PPA endpoint (predicted points added) and returns
        # offense and defense for each team-game.
        rows = cfbd.MetricsApi(_client()).get_predicted_points_added_by_game(
            year=year,
            classification=DivisionClassification.FBS,
        )
        return _frame_from_models(rows)

    return _cached("team_ppa", year, fetch)


def load_betting_lines(year: int) -> pd.DataFrame:
    """Per-game betting lines for ``year``, one row per provider.

    ``id`` is the game id. The provider argument is left unset so every book
    returned by the API is kept. Cached at ``data/raw/betting_lines_<year>.csv``.
    """

    def fetch() -> pd.DataFrame:
        # BettingApi.get_lines is the lines method in this package.
        games = cfbd.BettingApi(_client()).get_lines(year=year)
        return _lines_frame(games)

    return _cached("betting_lines", year, fetch)


def load_returning_production(year: int) -> pd.DataFrame:
    """Per-team returning production for ``year``.

    Cached at ``data/raw/returning_production_<year>.csv``.
    """

    def fetch() -> pd.DataFrame:
        # PlayersApi.get_returning_production matches the returning-production
        # method in this package.
        rows = cfbd.PlayersApi(_client()).get_returning_production(year=year)
        return _frame_from_models(rows)

    return _cached("returning_production", year, fetch)


def main() -> None:
    current_year = datetime.now().year
    loaders = (
        ("games", load_games),
        ("team_ppa", load_team_ppa),
        ("betting_lines", load_betting_lines),
        ("returning_production", load_returning_production),
    )
    for year in range(2016, current_year + 1):
        counts = [f"{name}={len(loader(year))}" for name, loader in loaders]
        print(f"{year}: {', '.join(counts)}")


if __name__ == "__main__":
    main()
