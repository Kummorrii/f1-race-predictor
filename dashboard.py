"""
F1 Prediction Dashboard — Phase 5 (Visualization)
====================================================
An interactive dashboard for exploring your simulation results.

This is a STREAMLIT app, not a normal Python script — it runs a small
local web server and opens in your browser. You don't "run" it with
plain `python`, you use the special `streamlit run` command (see below).

Setup (one time):
    pip install streamlit

Run:
    streamlit run dashboard.py

This opens automatically in your browser at http://localhost:8501
Press Ctrl+C in the terminal to stop it when you're done.
"""

import streamlit as st
import pandas as pd
import altair as alt


def sorted_bar_chart(data: pd.Series, label: str, color: str = "#3B82F6"):
    """
    st.bar_chart on its own always sorts categorical x-axes alphabetically,
    no matter how the data was sorted beforehand — that's just how it
    behaves under the hood. Using Altair directly gives us control over
    that, so the tallest bars actually show first, left to right.
    """
    chart_df = data.reset_index()
    chart_df.columns = ["Driver", label]
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=color)
        .encode(
            x=alt.X("Driver", sort="-y", title=None),
            y=alt.Y(label, title=label),
            tooltip=["Driver", label],
        )
        .properties(height=320)
    )
    return chart

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Race Prediction Dashboard")
st.caption(
    "Predictions generated from a Random Forest model trained on historical "
    "race data, run through a 5,000-race Monte Carlo simulation."
)

# --- Load the two files simulate_race.py produced ---
try:
    summary = pd.read_csv("simulation_results.csv")
    raw = pd.read_csv("raw_simulation_results.csv")
except FileNotFoundError:
    st.error(
        "Couldn't find simulation_results.csv or raw_simulation_results.csv. "
        "Run `python simulate_race.py` first, in this same folder."
    )
    st.stop()

# Sort by AvgFinish so the "best" drivers appear first everywhere.
summary = summary.sort_values("AvgFinish")

# --- Top row: three probability charts side by side ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Win Probability")
    win_data = summary.set_index("Driver")["WinProb%"].sort_values(ascending=False)
    st.altair_chart(sorted_bar_chart(win_data, "Win %"), use_container_width=True)

with col2:
    st.subheader("Podium Probability")
    podium_data = summary.set_index("Driver")["PodiumProb%"].sort_values(ascending=False)
    st.altair_chart(sorted_bar_chart(podium_data, "Podium %", color="#F59E0B"), use_container_width=True)

with col3:
    st.subheader("Points Probability")
    points_data = summary.set_index("Driver")["PointsProb%"].sort_values(ascending=False)
    st.altair_chart(sorted_bar_chart(points_data, "Points %", color="#10B981"), use_container_width=True)

st.divider()

# --- Full results table ---
st.subheader("Full Predictions Table")
st.dataframe(
    summary.rename(columns={
        "AvgFinish": "Avg. Predicted Finish",
        "WinProb%": "Win %",
        "PodiumProb%": "Podium %",
        "PointsProb%": "Points %",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --- Per-driver finish spread ---
# This is the closest thing to your original "predicted finish map" idea.
# We don't have real lap-time data (yet), so instead of a time gap, this
# shows HOW OFTEN a driver finished in each position across all 5,000
# simulated races — a probability spread rather than a single guess.
st.subheader("Predicted Finish Spread")
st.caption(
    "Pick a driver to see the full range of finishing positions the "
    "simulation produced for them, not just their single average."
)

chosen_driver = st.selectbox("Driver", summary["Driver"].tolist())

driver_positions = raw[raw["Driver"] == chosen_driver]["Position"]
position_counts = driver_positions.value_counts().sort_index()
# Reindex so every position 1..N shows on the chart, even ones with 0 hits.
full_range = pd.Series(0, index=range(1, raw["Position"].max() + 1))
position_counts = (position_counts + full_range).fillna(full_range).astype(int)

st.bar_chart(position_counts)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Most likely finish", int(driver_positions.mode()[0]))
col_b.metric("Best simulated finish", int(driver_positions.min()))
col_c.metric("Worst simulated finish", int(driver_positions.max()))
