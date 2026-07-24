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
    # NOTE: "Time" is each driver's race duration (for the winner) or gap
    # to the winner (everyone else), as a pandas Timedelta. "FullName" is
    # just for nicer display later — Abbreviation stays the primary key
    # used throughout the pipeline.
    keep_cols = [
        "DriverNumber", "BroadcastName", "FullName", "Abbreviation", "TeamName",
        "GridPosition", "Position", "Points", "Status",
        "Q1", "Q2", "Q3", "Time",
    ]
    results = results[[c for c in keep_cols if c in results.columns]]

    results["Season"] = year
    results["Round"] = round_number
    results["EventName"] = session.event["EventName"]

    return results


def collect_season(year: int) -> pd.DataFrame:
    """Collect every race in a season into one DataFrame."""
    schedule = get_season_schedule(year)
    all_results = []

    for _, event in schedule.iterrows():
        round_number = event["RoundNumber"]
        if round_number == 0:  # pre-season testing, not a real race
            continue

        print(f"Collecting {year} Round {round_number}: {event['EventName']}")
        race_df = collect_race_results(year, round_number)

        if race_df is not None:
            all_results.append(race_df)

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)


def main():
    # Start small on your first run — try just [2024] to make sure
    # everything works before pulling multiple seasons.
    seasons_to_collect = [2021, 2022, 2023, 2024, 2025]

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
