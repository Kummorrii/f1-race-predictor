"""
F1 Data Collector — Phase 1
============================
Pulls historical race results using the FastF1 library and saves them
as clean CSV files, one per season, ready for feature engineering later.

Setup (run once):
    pip install fastf1 pandas

Run:
    python collect_f1_data.py

Output:
    f1_data/race_results_2021.csv
    f1_data/race_results_2022.csv
    ... etc.
"""

import fastf1
import pandas as pd
from pathlib import Path

# FastF1 caches every API response locally so re-runs are instant and you
# don't hammer the server. First run per season will be slow (several
# minutes) — that's expected, it's downloading real timing data.
CACHE_DIR = Path("f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path("f1_data")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_season_schedule(year: int) -> pd.DataFrame:
    """Return the list of races for a given season."""
    try:
        return fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"  Could not fetch schedule for {year}: {e}")
        return pd.DataFrame()


def collect_race_results(year: int, round_number: int):
    """
    Pull the final classification for a single race weekend.
    Returns None if the session can't be loaded (e.g. race hasn't
    happened yet, or it's a testing event).
    """
    try:
        session = fastf1.get_session(year, round_number, "R")  # "R" = Race
        # We only ever read session.results below, so tell FastF1 to skip
        # laps, telemetry, weather, and messages — those are large and
        # require many extra API calls we don't need. This alone cuts API
        # usage per race by roughly 80-90%.
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"  Skipped {year} round {round_number}: {e}")
        return None

    results = session.results.copy()

    # Keep only the columns we actually need for modeling later.
    # (Full results has ~20 columns, most of it isn't useful yet.)
    keep_cols = [
        "DriverNumber", "BroadcastName", "FullName", "Abbreviation", "TeamName",
        "GridPosition", "Position", "Points", "Status",
        "Q1", "Q2", "Q3",
    ]
    results = results[[c for c in keep_cols if c in results.columns]]

    results["Season"] = year
    results["Round"] = round_number
    results["EventName"] = session.event["EventName"]

    return results


def _to_seconds(value):
    """Convert a lap/session time (pandas Timedelta, or its string form) to
    plain seconds, or None if it's missing/unparseable. Used for turning
    qualifying and practice lap times into a single comparable number."""
    try:
        if pd.isna(value) or value == "":
            return None
        return pd.to_timedelta(value).total_seconds()
    except Exception:
        return None


def collect_qualifying(year: int, round_number: int):
    """
    Qualifying sets the grid and happens BEFORE the race, so it's
    legitimate same-weekend predictive signal (not a leak of future
    information) — much like grid position already is.

    Returns a DataFrame of [Abbreviation, QualiPosition, QualiTimeSeconds]
    or None if the session can't be loaded.
    """
    try:
        session = fastf1.get_session(year, round_number, "Q")  # "Q" = Qualifying
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception:
        return None

    results = session.results.copy()
    if "Abbreviation" not in results.columns:
        return None

    # A driver's best time is whichever of Q1/Q2/Q3 they set — drivers
    # eliminated early only have a Q1 time, drivers who reach Q3 have all
    # three. Taking the minimum of whatever exists gives their best lap
    # regardless of how far they advanced.
    time_cols = [c for c in ["Q1", "Q2", "Q3"] if c in results.columns]
    if time_cols:
        results["QualiTimeSeconds"] = results[time_cols].apply(
            lambda row: min([s for s in (_to_seconds(v) for v in row) if s is not None], default=None),
            axis=1,
        )
    else:
        results["QualiTimeSeconds"] = None

    out_cols = ["Abbreviation", "QualiTimeSeconds"]
    if "Position" in results.columns:
        results = results.rename(columns={"Position": "QualiPosition"})
        out_cols.insert(1, "QualiPosition")

    return results[out_cols]


def collect_practice(year: int, round_number: int, session_code: str):
    """
    Practice session best lap time, as a rough pre-race pace indicator.
    NOTE: this is the least certain part of data collection — FastF1's
    results table is well-documented for Race/Qualifying/Sprint, but
    practice session results can be sparser (not every driver sets a
    representative lap, e.g. if a session is rain-affected). Treat this
    column as a useful-but-noisy signal, not a fully reliable one.

    session_code: "FP1", "FP2", or "FP3"
    Returns [Abbreviation, "{session_code}TimeSeconds"] or None.
    """
    try:
        session = fastf1.get_session(year, round_number, session_code)
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception:
        return None  # Common on sprint weekends — FP2/FP3 don't exist then

    results = session.results.copy()
    if "Abbreviation" not in results.columns:
        return None

    time_col = None
    for candidate in ["Time", "Q1"]:  # some FastF1 versions reuse "Q1" for best-lap in practice
        if candidate in results.columns:
            time_col = candidate
            break
    if time_col is None:
        return None

    col_name = f"{session_code}TimeSeconds"
    results[col_name] = results[time_col].apply(_to_seconds)
    valid = results[["Abbreviation", col_name]].dropna()
    return valid if not valid.empty else None


def collect_sprint(year: int, round_number: int):
    """
    Some race weekends include a Sprint race worth extra championship
    points (up to 8 for the winner) — a completely separate session from
    the main Race, and one that happens BEFORE it. Not every round has
    one, so a failure here is the NORMAL case for a non-sprint weekend,
    not an error.

    Returns [Abbreviation, SprintPosition, SprintPoints] or None.
    """
    try:
        session = fastf1.get_session(year, round_number, "S")  # "S" = Sprint
        session.load(laps=False, telemetry=False, weather=False, messages=False)
    except Exception:
        return None  # No sprint this weekend — expected most of the time

    results = session.results.copy()
    if "Abbreviation" not in results.columns or "Points" not in results.columns:
        return None

    out = results[["Abbreviation", "Points"]].rename(columns={"Points": "SprintPoints"})
    if "Position" in results.columns:
        out["SprintPosition"] = results["Position"]
    return out


def collect_season(year: int) -> pd.DataFrame:
    """Collect every race weekend — race, qualifying, practice, and sprint
    (where it exists) — into one DataFrame, one row per driver per round."""
    schedule = get_season_schedule(year)
    all_results = []

    for _, event in schedule.iterrows():
        round_number = event["RoundNumber"]
        if round_number == 0:  # pre-season testing, not a real race
            continue

        print(f"Collecting {year} Round {round_number}: {event['EventName']}")
        race_df = collect_race_results(year, round_number)

        if race_df is None:
            continue

        # --- Sprint (points + position), where it exists ---
        sprint = collect_sprint(year, round_number)
        if sprint is not None:
            race_df = race_df.merge(sprint, on="Abbreviation", how="left")
            race_df["SprintPoints"] = race_df["SprintPoints"].fillna(0)
            race_df["Points"] = race_df["Points"] + race_df["SprintPoints"]
            print(f"  + Sprint found for round {round_number}, points added")
        else:
            race_df["SprintPoints"] = 0

        # --- Qualifying: sets the grid, happens before the race ---
        quali = collect_qualifying(year, round_number)
        if quali is not None:
            race_df = race_df.merge(quali, on="Abbreviation", how="left")
            print(f"  + Qualifying data found for round {round_number}")

        # --- Practice: rough pre-race pace indicator ---
        for session_code in ["FP1", "FP2", "FP3"]:
            practice = collect_practice(year, round_number, session_code)
            if practice is not None:
                race_df = race_df.merge(practice, on="Abbreviation", how="left")

        all_results.append(race_df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def main():
    # Only 2026 needs to be collected going forward. Past seasons (like
    # 2025) are already complete and will never produce new data, so
    # there's no reason to keep re-fetching them — this also keeps the
    # model focused on the current regulations era instead of blending
    # in patterns from before the 2026 rule changes.
    #
    # If you ever want historical seasons collected again (e.g. to
    # rebuild everything from scratch), just add more years to this list.
    seasons_to_collect = [2026]

    for year in seasons_to_collect:
        print(f"\n=== Collecting {year} season ===")
        try:
            season_df = collect_season(year)
        except Exception as e:
            # A rate limit or network hiccup on ONE season shouldn't wipe
            # out results we already saved for other seasons. Log it and
            # move on — next scheduled run will pick up where this left off.
            print(f"  {year} season failed entirely, skipping: {e}")
            continue

        if season_df.empty:
            print(f"No data collected for {year}")
            continue

        out_path = OUTPUT_DIR / f"race_results_{year}.csv"
        season_df.to_csv(out_path, index=False)
        print(f"Saved {len(season_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
