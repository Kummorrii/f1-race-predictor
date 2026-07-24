"""
F1 Monte Carlo Simulation — Phase 4
=====================================
Instead of asking the model for ONE guess at the finishing order, we run
the race thousands of times with randomness added — representing things
like driving mistakes, strategy variance, and DNFs — and see how often
each driver wins, podiums, or scores points across all those runs.

This turns a single fragile guess into a probability, which is a much
more honest way to represent something as unpredictable as racing.

Run:
    python simulate_race.py

Output:
    Printed probability table
    simulation_results.csv
"""

import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "f1_model.joblib"
DATA_PATH = "f1_data/processed_features.csv"
N_SIMULATIONS = 5000

# Must match the exact column order the model was trained on.
FEATURE_COLUMNS = [
    "GridPosition",
    "RecentAvgFinish",
    "RecentAvgGrid",
    "RecentAvgPoints",
    "RecentDNFRate",
    "TeamRecentAvgPoints",
    "TrackHistoryAvgFinish",
]


def get_current_grid(df: pd.DataFrame):
    """
    We don't have a real 'next race' yet (it hasn't happened), so we treat
    the most recent race in your dataset as a stand-in for 'who's racing
    right now' — the current grid of active drivers.
    """
    latest_season = df["Season"].max()
    latest_round = df[df["Season"] == latest_season]["Round"].max()
    current = df[(df["Season"] == latest_season) & (df["Round"] == latest_round)]
    return current["Abbreviation"].unique(), latest_season, latest_round


def build_driver_profiles(df: pd.DataFrame, drivers, upcoming_track: str = None) -> dict:
    """
    For each driver, build the feature values they'd carry INTO the next
    race, based on everything that's happened so far — same logic as
    Phase 2's rolling features, just evaluated at 'right now' instead of
    at a specific past race.
    """
    profiles = {}

    for drv in drivers:
        history = df[df["Abbreviation"] == drv].sort_values(["Season", "Round"])
        if history.empty:
            continue

        recent = history.tail(5)
        team = history.iloc[-1]["TeamName"]
        team_history = df[df["TeamName"] == team].sort_values(["Season", "Round"]).tail(10)

        # Volatility = how much this driver's results normally swing race
        # to race. A wildly inconsistent driver gets more randomness in
        # the simulation; a metronome-consistent one gets less.
        volatility = history["Position"].std()
        if pd.isna(volatility) or volatility == 0:
            volatility = 3.0  # sensible default for a driver with little history

        profile = {
            "Abbreviation": drv,
            "TeamName": team,
            # We don't know next race's real grid position yet (that's set
            # by qualifying), so we use recent average grid as our best
            # guess until you plug in real qualifying results.
            "GridPosition": recent["GridPosition"].mean(),
            "RecentAvgFinish": recent["Position"].mean(),
            "RecentAvgGrid": recent["GridPosition"].mean(),
            "RecentAvgPoints": recent["Points"].mean(),
            "RecentDNFRate": 1 - recent["DidFinish"].mean(),
            "TeamRecentAvgPoints": team_history["Points"].mean(),
            "Volatility": volatility,
        }

        if upcoming_track:
            track_hist = history[history["EventName"] == upcoming_track]
            profile["TrackHistoryAvgFinish"] = (
                track_hist["Position"].mean() if not track_hist.empty else profile["RecentAvgFinish"]
            )
        else:
            profile["TrackHistoryAvgFinish"] = profile["RecentAvgFinish"]

        profiles[drv] = profile

    return profiles


def run_monte_carlo(model, profiles: dict, n_sim: int = N_SIMULATIONS, seed: int = 42):
    rng = np.random.default_rng(seed)
    drivers = list(profiles.keys())
    n_drivers = len(drivers)

    # Ask the model for each driver's "expected" finishing position, once.
    # This is their skill-based baseline before we add race-day randomness.
    X = pd.DataFrame([profiles[d] for d in drivers])[FEATURE_COLUMNS]
    base_predictions = model.predict(X)

    all_results = np.zeros((n_sim, n_drivers))

    for sim in range(n_sim):
        scores = []
        for i, d in enumerate(drivers):
            profile = profiles[d]

            # Add randomness scaled to how volatile this driver normally is.
            # This is what makes every simulated race come out differently,
            # just like real races do.
            noise = rng.normal(0, profile["Volatility"])
            score = base_predictions[i] + noise

            # Roll the dice on a DNF, using this driver's own recent DNF rate.
            if rng.random() < profile["RecentDNFRate"]:
                score = n_drivers + rng.uniform(0, 3)  # sends them to the back

            scores.append(score)

        # Lower score = better finish. Convert scores into clean 1..N
        # finishing positions for this one simulated race.
        order = np.argsort(scores)
        positions = np.empty(n_drivers, dtype=int)
        positions[order] = np.arange(1, n_drivers + 1)
        all_results[sim] = positions

    return drivers, all_results


def summarize(drivers, all_results) -> pd.DataFrame:
    rows = []
    for i, d in enumerate(drivers):
        positions = all_results[:, i]
        rows.append({
            "Driver": d,
            "AvgFinish": positions.mean(),
            "WinProb%": (positions == 1).mean() * 100,
            "PodiumProb%": (positions <= 3).mean() * 100,
            "PointsProb%": (positions <= 10).mean() * 100,
            "BestFinish": int(positions.min()),
            "WorstFinish": int(positions.max()),
        })
    return pd.DataFrame(rows).sort_values("AvgFinish").reset_index(drop=True)


def main():
    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    drivers, season, rnd = get_current_grid(df)
    print(f"Using current grid from Season {season} Round {rnd}: {len(drivers)} drivers")

    profiles = build_driver_profiles(df, drivers)

    print(f"Running {N_SIMULATIONS} simulated races...")
    driver_list, all_results = run_monte_carlo(model, profiles)

    summary = summarize(driver_list, all_results)

    print("\n=== Simulated race outcome probabilities ===")
    print(summary.to_string(index=False, formatters={
        "AvgFinish": "{:.1f}".format,
        "WinProb%": "{:.1f}".format,
        "PodiumProb%": "{:.1f}".format,
        "PointsProb%": "{:.1f}".format,
    }))

    summary.to_csv("simulation_results.csv", index=False)
    print("\nSaved full results to simulation_results.csv")

    # Also save the RAW per-simulation results (not just the summary).
    # The dashboard needs this to draw each driver's full "spread" of
    # possible finishes, not just their average.
    n_sim, n_drivers = all_results.shape
    raw_df = pd.DataFrame({
        "Simulation": np.repeat(np.arange(n_sim), n_drivers),
        "Driver": np.tile(driver_list, n_sim),
        "Position": all_results.flatten().astype(int),
    })
    raw_df.to_csv("raw_simulation_results.csv", index=False)
    print("Saved raw simulation data to raw_simulation_results.csv")


if __name__ == "__main__":
    main()
