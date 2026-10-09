"""Streamlit UI for the college football card, bet log, and settled results.

Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bets import (  # noqa: E402
    KELLY_FRACTION,
    MIN_EDGE,
    american_to_decimal,
    implied_prob,
    kelly_fraction,
    select_moneyline,
    win_prob,
)
from src.load import load_betting_lines, load_games  # noqa: E402
from src.margin import fit_margin_model, raw_margin  # noqa: E402
from src.ratings import get_weekly_ratings, home_field_term  # noqa: E402

BET_LOG = ROOT / "data" / "cfb_bet_log.csv"
LOG_COLUMNS = [
    "date_logged",
    "season",
    "week",
    "away",
    "home",
    "side",
    "odds_taken",
    "stake",
    "sportsbook",
    "model_prob",
    "book_prob",
]

st.set_page_config(page_title="CFB Model", layout="wide")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def _blank(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    return float(value)


@st.cache_data(show_spinner=False)
def cached_games(season: int) -> pd.DataFrame:
    return load_games(int(season))


@st.cache_data(show_spinner=False)
def cached_lines(season: int) -> pd.DataFrame:
    return load_betting_lines(int(season))


@st.cache_data(show_spinner=False)
def cached_ratings(season: int) -> pd.DataFrame:
    return get_weekly_ratings(int(season))


@st.cache_data(show_spinner=False)
def cached_margin_fit() -> dict:
    return fit_margin_model()


@st.cache_data(show_spinner=False)
def cached_home_field() -> float:
    return float(home_field_term())


def ensure_log() -> pd.DataFrame:
    BET_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not BET_LOG.is_file():
        empty = pd.DataFrame(columns=LOG_COLUMNS)
        empty.to_csv(BET_LOG, index=False)
        return empty
    log = pd.read_csv(BET_LOG)
    for column in LOG_COLUMNS:
        if column not in log.columns:
            log[column] = pd.NA
    return log[LOG_COLUMNS]


def write_log(frame: pd.DataFrame) -> None:
    BET_LOG.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    for column in LOG_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out = out[LOG_COLUMNS]
    out = out.dropna(subset=["away", "home", "side"], how="all")
    out.to_csv(BET_LOG, index=False)


def build_card(season: int, week: int) -> tuple[pd.DataFrame, int | None]:
    """One row per game in the selected week, using ratings known by that week.

    Returns the card and the ratings week actually used (the latest weekly
    rating at or before ``week``).
    """
    games = cached_games(season)
    if games.empty:
        return pd.DataFrame(), None
    week_games = games.loc[games["week"].astype(int) == int(week)].copy()
    if week_games.empty:
        return pd.DataFrame(), None

    ratings = cached_ratings(season)
    ratings_week = None
    snapshot = pd.DataFrame(columns=["team", "off_rating", "def_rating"]).set_index("team")
    if not ratings.empty:
        known = ratings.loc[ratings["week"].astype(int) <= int(week)]
        if not known.empty:
            ratings_week = int(known["week"].max())
            snapshot = known.loc[known["week"].astype(int) == ratings_week].set_index("team")

    fit = cached_margin_fit()
    scale = float(fit["scale"])
    intercept = float(fit["intercept"])
    residual_sd = float(fit["residual_sd"])
    field_edge = cached_home_field()

    lines = cached_lines(season)
    by_game = {int(game_id): group for game_id, group in lines.groupby("id")} if not lines.empty else {}

    rows = []
    for game in week_games.itertuples(index=False):
        home = game.homeTeam
        away = game.awayTeam
        neutral = bool(_as_bool(pd.Series([game.neutralSite])).iloc[0])
        home_off = home_def = away_off = away_def = None
        if home in snapshot.index and away in snapshot.index:
            home_off = float(snapshot.loc[home, "off_rating"])
            home_def = float(snapshot.loc[home, "def_rating"])
            away_off = float(snapshot.loc[away, "off_rating"])
            away_def = float(snapshot.loc[away, "def_rating"])

        model_prob = None
        if None not in (home_off, home_def, away_off, away_def):
            # def_rating is PPA allowed, so raw_margin receives defensive quality.
            margin_units = raw_margin(
                home_off,
                -home_def,
                away_off,
                -away_def,
                0.0 if neutral else field_edge,
            )
            predicted = scale * margin_units + intercept
            model_prob = win_prob(predicted, residual_sd)

        home_ml = away_ml = book_prob = edge = None
        group = by_game.get(int(game.id))
        chosen = select_moneyline(group) if group is not None else None
        if chosen:
            home_ml = _blank(chosen.get("home_ml"))
            away_ml = _blank(chosen.get("away_ml"))
        if home_ml not in (None, 0) and away_ml not in (None, 0) and model_prob is not None:
            home_implied = implied_prob(home_ml)
            away_implied = implied_prob(away_ml)
            book_prob = home_implied / (home_implied + away_implied)
            edge = model_prob - book_prob

        rows.append(
            {
                "away": away,
                "home": home,
                "model_prob": model_prob,
                "book_prob": book_prob,
                "edge": edge,
                "home_ml": home_ml,
                "away_ml": away_ml,
            }
        )

    card = pd.DataFrame(rows)
    if card.empty:
        return card, ratings_week
    card["_abs_edge"] = card["edge"].abs()
    card = card.sort_values("_abs_edge", ascending=False, na_position="last").drop(columns="_abs_edge")
    return card.reset_index(drop=True), ratings_week


def _side_prices(row: pd.Series, side: str) -> tuple[float | None, float | None, int]:
    """Model probability, vig-free book probability, and American odds for one side."""
    is_home = side == row["home"]
    model = _blank(row["model_prob"])
    book = _blank(row["book_prob"])
    odds = _blank(row["home_ml"] if is_home else row["away_ml"])
    if model is not None and not is_home:
        model = 1.0 - model
    if book is not None and not is_home:
        book = 1.0 - book
    default_odds = -110 if odds is None else int(round(odds))
    return model, book, default_odds


def _default_stake(model_prob: float | None, odds: int, bankroll: float) -> float:
    if model_prob is None or odds == 0:
        return 0.0
    try:
        fraction = kelly_fraction(model_prob, american_to_decimal(odds))
    except ValueError:
        return 0.0
    return round(fraction * bankroll, 2)


def _match_game(games: pd.DataFrame, season: int, week: int, home: str, away: str) -> pd.Series | None:
    if games.empty:
        return None
    hit = games.loc[
        (games["week"].astype(int) == int(week))
        & (games["homeTeam"] == home)
        & (games["awayTeam"] == away)
    ]
    if hit.empty:
        return None
    return hit.iloc[0]


def settle_log(log: pd.DataFrame) -> pd.DataFrame:
    """Add result and profit for bets whose games are final."""
    settled = log.copy()
    results = []
    profits = []
    seasons = [int(value) for value in settled["season"].dropna().unique()] if not settled.empty else []
    games_by_season = {season: cached_games(season) for season in seasons}
    for bet in settled.itertuples(index=False):
        if pd.isna(bet.season) or pd.isna(bet.week) or pd.isna(bet.home) or pd.isna(bet.away):
            results.append("pending")
            profits.append(np.nan)
            continue
        game = _match_game(games_by_season.get(int(bet.season), pd.DataFrame()), int(bet.season), int(bet.week), bet.home, bet.away)
        if game is None or not bool(_as_bool(pd.Series([game["completed"]])).iloc[0]):
            results.append("pending")
            profits.append(np.nan)
            continue
        home_points = pd.to_numeric(game["homePoints"], errors="coerce")
        away_points = pd.to_numeric(game["awayPoints"], errors="coerce")
        if pd.isna(home_points) or pd.isna(away_points):
            results.append("pending")
            profits.append(np.nan)
            continue
        if bet.side not in (bet.home, bet.away):
            results.append("unmatched")
            profits.append(np.nan)
            continue
        if home_points == away_points:
            results.append("push")
            profits.append(0.0)
            continue
        home_won = home_points > away_points
        won = home_won if bet.side == bet.home else not home_won
        stake = float(bet.stake) if pd.notna(bet.stake) else 0.0
        odds = int(round(float(bet.odds_taken))) if pd.notna(bet.odds_taken) else 0
        if won:
            results.append("won")
            profits.append(round(stake * (american_to_decimal(odds) - 1), 2))
        else:
            results.append("lost")
            profits.append(round(-stake, 2))
    settled["result"] = results
    settled["profit"] = profits
    return settled


def _show_card(card: pd.DataFrame) -> None:
    view = card.rename(
        columns={
            "away": "Away",
            "home": "Home",
            "model_prob": "Model home win probability",
            "book_prob": "Book home win probability",
            "edge": "Edge",
            "home_ml": "Home moneyline",
            "away_ml": "Away moneyline",
        }
    )
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            "Model home win probability": st.column_config.NumberColumn(format="percent"),
            "Book home win probability": st.column_config.NumberColumn(format="percent"),
            "Edge": st.column_config.NumberColumn(format="percent"),
            "Home moneyline": st.column_config.NumberColumn(format="%d"),
            "Away moneyline": st.column_config.NumberColumn(format="%d"),
        },
    )


st.title("CFB Model")
st.warning(
    "This model has NOT demonstrated a validated betting edge. A logistic regression test showed "
    "its rating-based predictions carry no statistically independent signal once the market's own "
    "price is accounted for (90% confidence interval for the model's coefficient: -0.351 to 0.745, "
    "straddling zero). Treat all probabilities below as reference only, not betting advice."
)

with st.sidebar:
    st.header("Bankroll")
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
    st.caption(f"Stake suggestions use quarter-Kelly ({KELLY_FRACTION:.0%}). MIN_EDGE is {MIN_EDGE:.0%}.")

tab_card, tab_log, tab_results = st.tabs(["Card", "Log a bet", "Results"])

with tab_card:
    season = st.number_input("Season", min_value=2016, max_value=datetime.now().year + 1, value=datetime.now().year, step=1)
    week = st.number_input("Week", min_value=1, max_value=20, value=1, step=1)
    card, ratings_week = build_card(int(season), int(week))
    if card.empty:
        st.info("No games for that season and week.")
    else:
        if ratings_week is None:
            st.caption("No walk-forward ratings are available at or before this week.")
        else:
            st.caption(f"Ratings are the walk-forward values from week {ratings_week}, not later weeks.")
        _show_card(card)

with tab_log:
    if card.empty:
        st.info("No games for that season and week.")
    else:
        labels = [f"{row.away} at {row.home}" for row in card.itertuples(index=False)]
        game_label = st.selectbox("Game", labels, key="log_game")
        selected = card.loc[[f"{row.away} at {row.home}" == game_label for row in card.itertuples(index=False)]].iloc[0]
        side = st.selectbox("Side", [selected["away"], selected["home"]], key=f"log_side_{game_label}")
        side_model, side_book, default_odds = _side_prices(selected, side)
        default_stake = _default_stake(side_model, default_odds, float(bankroll))
        # Form fields are unmounted when the game or side changes, and Streamlit
        # then drops their widget state. Keep a copy so a typed price comes back.
        kept = st.session_state.setdefault("kept_log", {})
        for state_key in list(st.session_state):
            if isinstance(state_key, str) and state_key.startswith(("log_odds_", "log_stake_", "log_book_")):
                kept[state_key] = st.session_state[state_key]
        odds_key = f"log_odds_{game_label}_{side}"
        stake_key = f"log_stake_{game_label}_{side}"
        book_key = f"log_book_{game_label}_{side}"
        if odds_key not in st.session_state:
            st.session_state[odds_key] = kept.get(odds_key, default_odds)
        if stake_key not in st.session_state:
            st.session_state[stake_key] = kept.get(stake_key, float(default_stake))
        if book_key not in st.session_state:
            st.session_state[book_key] = kept.get(book_key, "")
        with st.form("log_bet"):
            odds_taken = st.number_input("Odds taken", step=1, key=odds_key)
            stake = st.number_input("Stake ($)", min_value=0.0, step=1.0, key=stake_key)
            sportsbook = st.text_input("Sportsbook", key=book_key)
            submitted = st.form_submit_button("Log bet")
        kept[odds_key] = odds_taken
        kept[stake_key] = stake
        kept[book_key] = sportsbook
        if submitted:
            entry = pd.DataFrame(
                [
                    {
                        "date_logged": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "season": int(season),
                        "week": int(week),
                        "away": selected["away"],
                        "home": selected["home"],
                        "side": side,
                        "odds_taken": int(odds_taken),
                        "stake": float(stake),
                        "sportsbook": sportsbook,
                        "model_prob": side_model,
                        "book_prob": side_book,
                    }
                ]
            )
            write_log(pd.concat([ensure_log(), entry], ignore_index=True))
            st.session_state["log_version"] = st.session_state.get("log_version", 0) + 1
            st.success(f"Logged {side} at {int(odds_taken):+d} for ${float(stake):.2f}.")

with tab_results:
    log = ensure_log()
    if "log_version" not in st.session_state:
        st.session_state["log_version"] = 0
    st.caption("Edit or delete rows here, then save. Nothing is written until you click Save changes.")
    edited = st.data_editor(
        log,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        key=f"bet_editor_{st.session_state['log_version']}",
    )
    if st.session_state.pop("log_saved", False):
        st.success("Saved.")
    if st.button("Save changes"):
        write_log(edited)
        st.session_state["log_version"] += 1
        st.session_state["log_saved"] = True
        st.rerun()

    if edited.empty:
        st.info("No bets logged yet.")
    else:
        settled = settle_log(edited)
        done = settled.loc[settled["result"].isin(["won", "lost", "push"])]
        profit = float(done["profit"].sum()) if not done.empty else 0.0
        staked = float(pd.to_numeric(done["stake"], errors="coerce").fillna(0).sum()) if not done.empty else 0.0
        roi = profit / staked if staked else 0.0
        left, middle, right = st.columns(3)
        left.metric("Settled bets", int(len(done)))
        middle.metric("Profit / loss", f"${profit:,.2f}")
        right.metric("ROI", f"{roi:.1%}")
        st.dataframe(
            settled,
            hide_index=True,
            width="stretch",
            column_config={
                "model_prob": st.column_config.NumberColumn(format="percent"),
                "book_prob": st.column_config.NumberColumn(format="percent"),
                "profit": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
