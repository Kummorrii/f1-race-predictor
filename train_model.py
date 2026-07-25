"""
F1 Model Training — Phase 3
=============================
Trains a model to predict a driver's finishing POSITION using the
features we built in Phase 2 (recent form, team strength, track
history). Then checks how good it actually is against real recent
races it never saw during training.

Run:
    python train_model.py

Output:
    f1_model.joblib          <- the trained model, reusable later
    Printed accuracy report + a side-by-side prediction table
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# These are the columns the model is allowed to "look at" when predicting.
# Notice GridPosition IS included — where you start is a huge predictor of
# where you finish — but Position, Points, Status are NOT included, since
# those are the answer we're trying to predict, not an input.
# QualiGapSeconds is also fair game even though it's from THIS race weekend
# — qualifying happens before the race, so it's pre-race information, not
# a leak of the result we're trying to predict (same logic as GridPosition).
FEATURE_COLUMNS = [
    "GridPosition",
    "QualiGapSeconds",
    "RecentAvgFinish",
    "RecentAvgGrid",
    "RecentAvgPoints",
    "RecentDNFRate",
    "TeamRecentAvgPoints",
    "TrackHistoryAvgFinish",
]

TARGET_COLUMN = "Position"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Early-season rows have no history yet, so some feature columns are
    # blank (NaN). A model can't learn from blanks, so we drop those rows
    # here. We only ever do this for TRAINING/TESTING — later, for live
    # predictions, we'll handle missing history differently.
    before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    after = len(df)
    print(f"  {path}: {before} rows -> {after} rows after dropping incomplete ones")

    return df


def main():
    print("Loading data...")
    train_df = load_and_clean("f1_data/train.csv")
    test_df = load_and_clean("f1_data/test.csv")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("\nTraining model...")
    # n_estimators=300 -> the "forest" is 300 individual decision trees,
    # each one votes, and we average their votes. More trees = generally
    # steadier predictions, at the cost of a little more compute time.
    # random_state=42 just makes the result reproducible when you re-run.
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("Done training.")

    # --- Evaluate ---
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print(f"\n=== Accuracy ===")
    print(f"Mean Absolute Error: {mae:.2f} positions")
    print("(This means: on average, the model's predicted finishing")
    print(" position is off by about this many places from reality.")
    print(" Lower is better. A MAE under ~3 is a solid start for a")
    print(" first model on this kind of data.)")

    # --- Show real examples side by side ---
    results_df = test_df[["Season", "Round", "EventName", "Abbreviation"]].copy()
    results_df["ActualPosition"] = y_test.values
    results_df["PredictedPosition"] = predictions.round(1)
    results_df["Error"] = (results_df["ActualPosition"] - results_df["PredictedPosition"]).abs().round(1)

    print(f"\n=== Sample predictions vs reality (most recent race) ===")
    latest_race = results_df.sort_values(["Season", "Round"]).tail(20)
    print(latest_race.sort_values("ActualPosition").to_string(index=False))

    # --- Feature importance: which inputs mattered most? ---
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values(ascending=False)

    print(f"\n=== What the model relied on most ===")
    for feature, score in importances.items():
        bar = "#" * int(score * 50)
        print(f"  {feature:<25} {score:.3f}  {bar}")

    # --- Save the model so Phase 4 (simulation) can reuse it ---
    joblib.dump(model, "f1_model.joblib")
    print(f"\nModel saved to f1_model.joblib")


if __name__ == "__main__":
    main()
