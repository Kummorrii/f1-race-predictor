"""
F1 Prediction Dashboard — Phase 5 (Visualization)
====================================================
An interactive, professionally styled dashboard for exploring your
simulation results.

This is a STREAMLIT app, not a normal Python script — it runs a small
local web server and opens in your browser. You don't "run" it with
plain `python`, you use the special `streamlit run` command (see below).

Setup (one time):
    pip install streamlit

Run locally:
    streamlit run dashboard.py

This opens automatically in your browser at http://localhost:8501
Press Ctrl+C in the terminal to stop it when you're done.
"""

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLING
# ============================================================
# Everything in this block is cosmetic — it doesn't touch any of the
# data or model logic. It: (1) pulls in two Google Fonts, (2) hides
# Streamlit's default chrome (hamburger menu, footer, colored top bar)
# for a cleaner look, and (3) defines reusable "card" styles used below.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important; letter-spacing: 0.5px; }

#MainMenu, footer, header { visibility: hidden; }

.hero {
    padding: 1.75rem 2rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #1a1014 0%, #161B22 60%);
    border: 1px solid #2A2F3A;
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.4rem;
    margin: 0;
    color: #F5F5F7;
}
.hero .accent { color: #EF4444; }
.hero p {
    color: #9CA3AF;
    margin: 0.4rem 0 0 0;
    font-size: 0.95rem;
}
.badge {
    display: inline-block;
    margin-top: 0.75rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.12);
    color: #10B981;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.podium-card {
    border-radius: 12px;
    padding: 1.1rem 1rem;
    background: #161B22;
    border: 1px solid #2A2F3A;
    text-align: center;
}
.podium-card .medal { font-size: 1.6rem; }
.podium-card .driver { font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1.4rem; color: #F5F5F7; margin: 0.2rem 0; }
.podium-card .stat { color: #EF4444; font-size: 1.8rem; font-weight: 700; font-family: 'Rajdhani', sans-serif; }
.podium-card .label { color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }

.footer-note {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #2A2F3A;
    color: #6B7280;
    font-size: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


def styled_bar_chart(data: pd.Series, label: str, color: str = "#EF4444"):
    """
    A dark-theme-matched, properly sorted bar chart. st.bar_chart on its
    own always alphabetizes categories no matter how the data was sorted
    beforehand, so we use Altair directly for control over both sorting
    and the visual theme.
    """
    chart_df = data.reset_index()
    chart_df.columns = ["Driver", label]
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color=color, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Driver", sort="-y", title=None),
            y=alt.Y(label, title=label),
            tooltip=["Driver", label],
        )
        .properties(height=300, background="transparent")
        .configure_axis(labelColor="#9CA3AF", titleColor="#9CA3AF", gridColor="#2A2F3A")
        .configure_view(strokeWidth=0)
    )
    return chart


# ============================================================
# LOAD DATA
# ============================================================
try:
    summary = pd.read_csv("simulation_results.csv")
    raw = pd.read_csv("raw_simulation_results.csv")
except FileNotFoundError:
    st.error(
        "Couldn't find simulation_results.csv or raw_simulation_results.csv. "
        "Run `python simulate_race.py` first, in this same folder."
    )
    st.stop()

summary = summary.sort_values("AvgFinish").reset_index(drop=True)

# Last-updated timestamp, pulled straight from the file itself — this
# updates automatically every time your GitHub Action re-runs the pipeline.
try:
    updated = datetime.fromtimestamp(Path("simulation_results.csv").stat().st_mtime)
    updated_str = updated.strftime("%B %d, %Y")
except Exception:
    updated_str = "recently"

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(f"""
<div class="hero">
    <h1>🏁 F1 RACE <span class="accent">PREDICTOR</span></h1>
    <p>Machine learning predictions from a 5,000-race Monte Carlo simulation,
    trained on real historical race data.</p>
    <span class="badge">● Auto-updates weekly · Last updated {updated_str}</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP 3 FAVORITES — quick-glance cards
# ============================================================
top3 = summary.sort_values("WinProb%", ascending=False).head(3).reset_index(drop=True)
medals = ["🥇", "🥈", "🥉"]

cols = st.columns(3)
for i, col in enumerate(cols):
    if i < len(top3):
        row = top3.iloc[i]
        with col:
            st.markdown(f"""
            <div class="podium-card">
                <div class="medal">{medals[i]}</div>
                <div class="driver">{row['Driver']}</div>
                <div class="stat">{row['WinProb%']:.1f}%</div>
                <div class="label">Win Probability</div>
            </div>
            """, unsafe_allow_html=True)

st.write("")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "📋 Full Table", "🔍 Driver Deep Dive"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Win Probability")
        win_data = summary.set_index("Driver")["WinProb%"].sort_values(ascending=False)
        st.altair_chart(styled_bar_chart(win_data, "Win %", "#EF4444"), width='stretch')
    with col2:
        st.subheader("Podium Probability")
        podium_data = summary.set_index("Driver")["PodiumProb%"].sort_values(ascending=False)
        st.altair_chart(styled_bar_chart(podium_data, "Podium %", "#F59E0B"), width='stretch')
    with col3:
        st.subheader("Points Probability")
        points_data = summary.set_index("Driver")["PointsProb%"].sort_values(ascending=False)
        st.altair_chart(styled_bar_chart(points_data, "Points %", "#10B981"), width='stretch')

with tab2:
    st.subheader("Full Predictions Table")
    display_df = summary.rename(columns={
        "AvgFinish": "Avg. Predicted Finish",
        "WinProb%": "Win %",
        "PodiumProb%": "Podium %",
        "PointsProb%": "Points %",
    })
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Win %": st.column_config.ProgressColumn("Win %", min_value=0, max_value=100, format="%.1f%%"),
            "Podium %": st.column_config.ProgressColumn("Podium %", min_value=0, max_value=100, format="%.1f%%"),
            "Points %": st.column_config.ProgressColumn("Points %", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

with tab3:
    st.subheader("Predicted Finish Spread")
    st.caption(
        "Pick a driver to see the full range of finishing positions the "
        "simulation produced for them, not just their single average. "
        "This reflects finishing position across 5,000 simulated races — "
        "not lap times, since detailed timing data isn't part of the "
        "current dataset."
    )

    chosen_driver = st.selectbox("Driver", summary["Driver"].tolist())

    driver_positions = raw[raw["Driver"] == chosen_driver]["Position"]
    position_counts = driver_positions.value_counts().sort_index()
    full_range = pd.Series(0, index=range(1, raw["Position"].max() + 1))
    position_counts = (position_counts + full_range).fillna(full_range).astype(int)

    # Altair needs the count column named explicitly
    position_df = position_counts.reset_index()
    position_df.columns = ["Position", "count"]
    spread_chart = (
        alt.Chart(position_df)
        .mark_bar(color="#EF4444", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Position:O", title="Finishing Position"),
            y=alt.Y("count:Q", title="Simulations"),
            tooltip=["Position", "count"],
        )
        .properties(height=320, background="transparent")
        .configure_axis(labelColor="#9CA3AF", titleColor="#9CA3AF", gridColor="#2A2F3A")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(spread_chart, width='stretch')

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Most likely finish", int(driver_positions.mode()[0]))
    col_b.metric("Best simulated finish", int(driver_positions.min()))
    col_c.metric("Worst simulated finish", int(driver_positions.max()))

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="footer-note">
    Predictions are generated from historical data and should be read as
    probabilities, not certainties — motorsport is inherently unpredictable.
    Pipeline auto-updates weekly via GitHub Actions.
</div>
""", unsafe_allow_html=True)
