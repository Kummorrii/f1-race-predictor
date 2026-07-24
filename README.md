# 🏎️ F1 Race Predictor

A self-updating machine learning system that predicts Formula 1 race outcomes.
It collects real historical race data, engineers features from driver and team
form, trains a model to predict finishing position, runs a Monte Carlo
simulation to turn that into win/podium/points probabilities, and displays it
all in an interactive dashboard — then updates itself automatically after
every real race weekend, with no manual work required.

## How it works

```
Historical race data  →  Feature engineering  →  Prediction model
                                                         ↓
        Dashboard      ←   Monte Carlo simulation  ←────┘
```

1. **Data collection** (`collect_f1_data.py`) — pulls real race results from
   the [FastF1](https://github.com/theOehrly/Fast-F1) library, which sources
   official F1 timing data.
2. **Feature engineering** (`build_features.py`) — turns raw race-by-race
   results into predictive features: each driver's recent form, their team's
   recent strength, and their personal history at each specific track. All
   features are built using only *past* races relative to each row, to avoid
   leaking future information into training.
3. **Model training** (`train_model.py`) — trains a Random Forest Regressor
   to predict finishing position, and evaluates it against the most recent
   real races the model never saw during training.
4. **Monte Carlo simulation** (`simulate_race.py`) — runs the "next" race
   5,000 times with randomness layered on top of the model's predictions
   (representing race-day variance and DNF risk), producing realistic win,
   podium, and points probabilities per driver instead of a single fragile
   guess.
5. **Dashboard** (`dashboard.py`) — an interactive Streamlit app visualizing
   all of the above: probability charts, a full predictions table, and each
   driver's full spread of simulated finishing positions.
6. **Automation** (`.github/workflows/update_predictions.yml`) — a GitHub
   Actions workflow that runs the entire pipeline on a schedule (every Monday,
   after race weekends) and commits the refreshed data, model, and
   predictions straight back to this repository. No server or personal
   machine required to keep it current.

## Tech stack

- **Python** — pandas, scikit-learn, NumPy
- **[FastF1](https://github.com/theOehrly/Fast-F1)** — official F1 timing data
- **Streamlit** + **Altair** — interactive dashboard
- **GitHub Actions** — scheduled automation

## Running it locally

```bash
git clone https://github.com/Kummorrii/f1-race-predictor.git
cd f1-race-predictor
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt

python collect_f1_data.py    # pull historical race data
python build_features.py     # engineer features
python train_model.py        # train and evaluate the model
python simulate_race.py      # run the 5,000-race simulation
streamlit run dashboard.py   # launch the dashboard
```

## Model performance & honest limitations

This is a first-pass model trained on a relatively small dataset (roughly one
and a half seasons of races as of writing), which naturally limits accuracy —
Formula 1 is also a high-variance sport where mechanical failures, weather,
and racing incidents introduce real randomness that no model can fully
predict from pre-race data alone. The current model's mean absolute error is
in the 4-5 position range on held-out recent races.

Known limitations:
- Trained mostly on the 2026 season, which introduced new technical
  regulations — meaning historical patterns from prior seasons don't always
  transfer cleanly.
- Simulates finishing **position** probabilities, not real lap times or
  time gaps, since detailed telemetry wasn't part of the initial data
  collection.
- Doesn't yet account for qualifying results, weather forecasts, or
  mid-race strategy — all planned future additions.

## Possible future improvements

- Incorporate real qualifying session data instead of using recent-average
  grid position as a proxy
- Add weather data as a feature
- Use FastF1's lap-level telemetry to model actual predicted lap times, not
  just finishing order
- Expand the training dataset with more historical seasons
- Hyperparameter tuning / experiment with gradient boosting models

## Acknowledgments

Built using data from [FastF1](https://github.com/theOehrly/Fast-F1), an
open-source Python library for accessing Formula 1 timing and telemetry data.
