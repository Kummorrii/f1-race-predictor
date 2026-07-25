"""
F1 Feature Engineering — Phase 2
==================================
Scans every race_results_*.csv in f1_data/, combines them, and builds
features a model can actually learn from: recent form, team strength,
and track history — instead of just raw single-race results.

Run:
    python build_features.py

Output:
    f1_data/processed_features.csv
    f1_data/train.csv
    f1_data/test.csv
"""

import pandas as pd
import glob
from pathlib import Path

DATA_DIR = Path("f1_data")


def load_all_races() -> pd.DataFrame:
    """
    Auto-scan f1_data/ for every race_results_*.csv and combine them.
    This means you never have to touch this function again — just drop
    more season CSVs in the folder and re-run.
    """
    files = sorted(glob.glob(str(DATA_DIR / "race_results_*.csv")))
    if not files:
        raise FileNotFoundError(
            "No race_results_*.csv files found in f1_data/. "
            "Run collect_f1_data.py first."
        )

    print(f"Found {len(files)} season file(s): {[Path(f).name for f in files]}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Sort chronologically — critical, since every feature below depends
    # on knowing what happened "before" a given race.
    df = df.sort_values(["Season", "Round"]).reset_index(drop=True)
    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create clean target variables for the model to predict."""
    df["DidFinish"] = df["Status"].apply(
        lambda s: 1 if isinstance(s, str) and ("Finished" in s or "Lap" in s) else 0
    )
    df["Podium"] = (df["Position"] <= 3).astype(int)
    df["Win"] = (df["Position"] == 1).astype(int)
    return df


def add_driver_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each driver, compute rolling averages from their PAST races only.

    This is the single most important detail in this whole script: we use
    .shift(1) before the rolling window so that a driver's features for
    "today's" race are built only from races that already happened.
    Skip that shift and you leak the answer into the input — the model
    would look great in testing and then be useless in real prediction.
    """
    df = df.sort_values(["Abbreviation", "Season", "Round"])

    grouped = df.groupby("Abbreviation")

    df["RecentAvgFinish"] = grouped["Position"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["RecentAvgGrid"] = grouped["GridPosition"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["RecentAvgPoints"] = grouped["Points"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["RecentDNFRate"] = grouped["DidFinish"].transform(
        lambda x: (1 - x.shift(1)).rolling(5, min_periods=1).mean()
    )

    return df.sort_values(["Season", "Round"]).reset_index(drop=True)


def add_team_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same idea, but averaged across the whole team (both cars) recently."""
    df = df.sort_values(["TeamName", "Season", "Round"])

    grouped = df.groupby("TeamName")
    df["TeamRecentAvgPoints"] = grouped["Points"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    return df.sort_values(["Season", "Round"]).reset_index(drop=True)


def add_track_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A driver's historical average finish at THIS specific track, using
    only prior years/visits (same shift logic as above).
    """
    df = df.sort_values(["Abbreviation", "EventName", "Season", "Round"])

    grouped = df.groupby(["Abbreviation", "EventName"])
    df["TrackHistoryAvgFinish"] = grouped["Position"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    return df.sort_values(["Season", "Round"]).reset_index(drop=True)


def build_train_test_split(df: pd.DataFrame, test_races: int = 5):
    """
    Time-based split, NOT random. We hold out the most recent N races as
    the test set — this mimics reality, where you always predict the
    future from the past, and gives an honest sense of how the model
    would do on races it hasn't seen yet.
    """
    unique_races = df[["Season", "Round"]].drop_duplicates().sort_values(
        ["Season", "Round"]
    )
    test_race_keys = unique_races.tail(test_races)

    is_test = df.set_index(["Season", "Round"]).index.isin(
        test_race_keys.set_index(["Season", "Round"]).index
    )

    train_df = df[~is_test].copy()
    test_df = df[is_test].copy()
    return train_df, test_df


def main():
    df = load_all_races()
    print(f"Loaded {len(df)} driver-race rows total across all collected seasons")
    df = add_target_columns(df)

    # Track history is inherently a multi-season idea — "this driver has
    # historically been strong at this track" — and each Grand Prix only
    # happens once per season, so within a single season there's never a
    # "prior visit" to look back at. We compute it from the FULL history
    # (all seasons) here, before filtering, then carry just that one
    # column forward. Recent-form and team-form features, by contrast,
    # SHOULD stay season-scoped (see below) since we don't want those
    # blending in patterns from before the 2026 regulation change.
    track_history_full = add_track_history_features(df.copy())
    track_history_lookup = track_history_full[
        ["Abbreviation", "Season", "Round", "TrackHistoryAvgFinish"]
    ]

    # Focus the model on the CURRENT season only. Older seasons (like
    # 2025) may have run under different technical regulations, so
    # blending them in risks teaching the model outdated patterns rather
    # than how the field actually stacks up right now. This does mean
    # less training data, especially early in a season — a deliberate
    # tradeoff favoring "recent and relevant" over "large but stale."
    # (Uses whichever season is latest in the data, so this doesn't need
    # editing again next year.)
    current_season = df["Season"].max()
    df = df[df["Season"] == current_season].reset_index(drop=True)
    print(f"Focusing on season {current_season} only: {len(df)} rows")

    df = add_driver_form_features(df)
    df = add_team_form_features(df)
    df = df.merge(track_history_lookup, on=["Abbreviation", "Season", "Round"], how="left")

    # Early-season rows won't have enough history for some features —
    # that's expected, not a bug. We leave the NaNs in the saved file so
    # you can see exactly what's missing, and handle it at model-training
    # time instead (dropping rows loses data you might still want).
    out_path = DATA_DIR / "processed_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} processed rows to {out_path}")

    train_df, test_df = build_train_test_split(df, test_races=3)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")
    print(f"Test set covers the most recent {test_df[['Season','Round']].drop_duplicates().shape[0]} races")


if __name__ == "__main__":
    main()
