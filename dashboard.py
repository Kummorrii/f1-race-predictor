"""
F1 Prediction Dashboard — Phase 5 (Visualization)
====================================================
An interactive, professionally styled dashboard with animated charts AND
animated tables, showing real season standings plus machine learning race
predictions.

This is a STREAMLIT app, not a normal Python script — it runs a small
local web server and opens in your browser. You don't "run" it with
plain `python`, you use the special `streamlit run` command (see below).

Setup (one time):
    pip install streamlit

Run locally:
    streamlit run dashboard.py

This opens automatically in your browser at http://localhost:8501
Press Ctrl+C in the terminal to stop it when you're done.

NOTE ON ANIMATIONS: Streamlit's built-in charts AND its built-in data
table widget can't animate, and the table widget in particular has its
own separate dark/light rendering that doesn't reliably follow Streamlit's
theme config. So every chart AND every table in this file is hand-built
as HTML/CSS/SVG and rendered through st.components.v1.html, which runs in
an isolated iframe that (unlike st.markdown) actually supports CSS
animation and gives us full control over colors.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import math
import html as html_lib
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLING — gray page, white cards, red accents
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Rajdhani', sans-serif !important; font-weight: 700 !important; letter-spacing: 0.3px; color: #111827; }

#MainMenu, footer, header { visibility: hidden; }

.stApp { background-color: #F3F4F6; }
.block-container { padding-top: 2rem; }

.hero {
    padding: 2rem 2.25rem;
    border-radius: 18px;
    background: radial-gradient(circle at top left, rgba(220,38,38,0.06), transparent 60%), #FFFFFF;
    border: 1px solid #ECECEC;
    box-shadow: 0 4px 16px rgba(17,24,39,0.05);
    margin-bottom: 1.5rem;
}
.hero h1 { font-size: 2.6rem; margin: 0; color: #111827; font-weight: 800 !important; }
.hero .accent { color: #DC2626; }
.hero p { color: #6B7280; margin: 0.5rem 0 0 0; font-size: 0.98rem; }
.badge {
    display: inline-block;
    margin-top: 0.85rem;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.podium-card {
    border-radius: 16px;
    padding: 1.3rem 1rem;
    background: #FFFFFF;
    border: 1px solid #ECECEC;
    box-shadow: 0 4px 14px rgba(17,24,39,0.05);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.podium-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(220,38,38,0.14);
}
.podium-card .avatar {
    width: 48px; height: 48px; border-radius: 50%;
    background: linear-gradient(135deg, #EF4444, #B91C1C);
    color: white; display: flex; align-items: center; justify-content: center;
    font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1.15rem;
    margin: 0 auto 0.5rem auto;
    box-shadow: 0 4px 10px rgba(220,38,38,0.35);
}
.podium-card .medal { font-size: 1.3rem; }
.podium-card .driver { font-family: 'Rajdhani', sans-serif; font-weight: 700; font-size: 1.3rem; color: #111827; margin: 0.2rem 0; }
.podium-card .stat { color: #DC2626; font-size: 2rem; font-weight: 800; font-family: 'Rajdhani', sans-serif; }
.podium-card .label { color: #9CA3AF; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

.panel {
    background: #FFFFFF;
    border-radius: 16px;
    border: 1px solid #ECECEC;
    box-shadow: 0 4px 14px rgba(17,24,39,0.05);
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
}
.panel h4 { margin: 0 0 0.9rem 0; font-size: 1.05rem; }

.section-note { color: #6B7280; font-size: 0.85rem; margin-top: -0.5rem; margin-bottom: 1rem; }

.footer-note {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #E5E7EB;
    color: #9CA3AF;
    font-size: 0.8rem;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 999px; padding: 8px 18px; background: #FFFFFF;
    border: 1px solid #ECECEC; font-weight: 600; transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover { border-color: #DC2626; color: #DC2626; }
.stTabs [aria-selected="true"] { background: #DC2626 !important; color: white !important; border-color: #DC2626 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# ANIMATED HTML COMPONENTS
# ============================================================
def animated_bar_html(data: list, unit: str = "%", color_start: str = "#EF4444", color_end: str = "#B91C1C") -> str:
    """A leaderboard bar chart where every bar grows in from zero on load, staggered row by row."""
    max_value = max((v for _, v in data), default=1) or 1
    rows_html = ""
    for i, (label, value) in enumerate(data):
        pct = min(100, (value / max_value) * 100)
        safe_label = html_lib.escape(str(label))
        delay = 0.05 * i
        rows_html += f"""
        <div class="bar-row" style="animation-delay:{delay:.2f}s">
            <div class="bar-label" title="{safe_label}">{safe_label}</div>
            <div class="bar-track">
                <div class="bar-fill" style="--target:{pct:.1f}%; animation-delay:{delay + 0.15:.2f}s;"></div>
            </div>
            <div class="bar-value">{value:.1f}{unit}</div>
        </div>"""

    return f"""
    <div class="bars-wrap">{rows_html}</div>
    <style>
        .bars-wrap {{ font-family: 'Inter', sans-serif; padding: 4px 2px; }}
        .bar-row {{ display:flex; align-items:center; gap:10px; margin-bottom:12px;
                    opacity:0; animation: fadeIn 0.4s ease-out forwards; }}
        .bar-row:hover .bar-track {{ box-shadow: 0 0 0 2px rgba(220,38,38,0.15); }}
        .bar-label {{ width:150px; font-size:13px; color:#374151; font-weight:500;
                      flex-shrink:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
        .bar-track {{ flex:1; background:#F3F4F6; border-radius:8px; height:14px; overflow:hidden;
                      transition: box-shadow 0.15s ease; }}
        .bar-fill {{ height:100%; border-radius:8px; width:0;
                     background:linear-gradient(90deg, {color_start}, {color_end});
                     animation: growBar 1s cubic-bezier(0.22,1,0.36,1) forwards; }}
        .bar-value {{ width:54px; text-align:right; font-size:13px; font-weight:700; color:#111827; flex-shrink:0; }}
        @keyframes growBar {{ from {{ width:0; }} to {{ width: var(--target); }} }}
        @keyframes fadeIn {{ from {{ opacity:0; transform:translateY(4px); }} to {{ opacity:1; transform:translateY(0); }} }}
    </style>
    """


def animated_donut_html(segments: list, center_label: str = "", center_sub: str = "", size: int = 210) -> str:
    """A multi-segment donut where each segment draws itself around the ring on load."""
    radius = 68
    stroke_width = 26
    circumference = 2 * math.pi * radius
    cx = cy = size / 2

    circles_html = ""
    legend_html = ""
    cumulative = 0
    for i, (label, value, color) in enumerate(segments):
        frac = value / 100
        dash = frac * circumference
        gap = circumference - dash
        rotation = (cumulative / 100) * 360 - 90
        cumulative += value

        circles_html += f"""
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}"
            stroke-dasharray="{dash:.2f} {gap:.2f}" stroke-dashoffset="{dash:.2f}"
            transform="rotate({rotation:.1f} {cx} {cy})" stroke-linecap="butt"
            class="donut-seg" style="animation-delay:{i * 0.15:.2f}s;"/>"""

        legend_html += f"""
        <div class="legend-item">
            <span class="dot" style="background:{color}"></span>
            <span class="legend-label">{html_lib.escape(str(label))}</span>
            <span class="legend-value">{value:.0f}%</span>
        </div>"""

    safe_center_label = html_lib.escape(str(center_label))
    safe_center_sub = html_lib.escape(str(center_sub))

    return f"""
    <div class="donut-wrap">
        <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
            {circles_html}
            <text x="{cx}" y="{cy - 4}" text-anchor="middle" class="donut-center-label">{safe_center_label}</text>
            <text x="{cx}" y="{cy + 16}" text-anchor="middle" class="donut-center-sub">{safe_center_sub}</text>
        </svg>
        <div class="donut-legend">{legend_html}</div>
    </div>
    <style>
        .donut-wrap {{ display:flex; align-items:center; gap:18px; font-family:'Inter',sans-serif; }}
        .donut-seg {{ animation: drawSeg 1.1s cubic-bezier(0.22,1,0.36,1) forwards; }}
        @keyframes drawSeg {{ to {{ stroke-dashoffset: 0; }} }}
        .donut-center-label {{ font-size:20px; font-weight:700; fill:#111827; font-family:'Rajdhani',sans-serif; }}
        .donut-center-sub {{ font-size:10px; fill:#9CA3AF; text-transform:uppercase; letter-spacing:1px; }}
        .legend-item {{ display:flex; align-items:center; gap:8px; margin-bottom:8px; font-size:13px; color:#374151; }}
        .dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
        .legend-label {{ flex:1; }}
        .legend-value {{ font-weight:600; color:#111827; }}
    </style>
    """


def animated_table_html(columns: list, rows: list, bar_col: str = None, bar_max: float = None,
                         rank_col: str = None, color_start: str = "#EF4444", color_end: str = "#B91C1C") -> str:
    """
    A data table where rows fade in with a stagger on load, hover-highlight,
    and (optionally) one numeric column renders as an inline animated bar
    instead of plain text — this is what replaces st.dataframe everywhere,
    since Streamlit's own table widget doesn't reliably follow our theme.

    columns: list of (key, header_label) in display order
    rows: list of dicts
    bar_col: key of the column to render as an inline animated bar
    rank_col: key of the column that gets a medal emoji for values 1/2/3
    """
    if bar_col:
        if bar_max is None:
            bar_max = max((float(r.get(bar_col, 0)) for r in rows), default=1) or 1

    thead = "".join(f"<th>{html_lib.escape(label)}</th>" for _, label in columns)

    tbody_rows = ""
    for i, row in enumerate(rows):
        cells = ""
        for key, _ in columns:
            val = row.get(key, "")
            if key == bar_col:
                num = float(val) if val != "" else 0.0
                pct = min(100, (num / bar_max) * 100)
                cells += f"""<td class="bar-cell">
                    <div class="mini-bar-track"><div class="mini-bar-fill" style="--target:{pct:.1f}%; animation-delay:{0.2 + i * 0.03:.2f}s;"></div></div>
                    <span class="mini-bar-value">{num:.0f}</span>
                </td>"""
            elif rank_col and key == rank_col:
                medal = {1: "🥇 ", 2: "🥈 ", 3: "🥉 "}.get(val, "")
                cells += f'<td class="rank-cell">{medal}{html_lib.escape(str(val))}</td>'
            else:
                cells += f"<td>{html_lib.escape(str(val))}</td>"
        tbody_rows += f'<tr class="table-row" style="animation-delay:{i * 0.025:.2f}s">{cells}</tr>'

    return f"""
    <div class="table-wrap">
        <table class="animated-table">
            <thead><tr>{thead}</tr></thead>
            <tbody>{tbody_rows}</tbody>
        </table>
    </div>
    <style>
        .table-wrap {{ font-family:'Inter',sans-serif; overflow-x:auto; }}
        .animated-table {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
        .animated-table th {{ text-align:left; padding:10px 14px; color:#6B7280; font-weight:700; font-size:11px;
                               text-transform:uppercase; letter-spacing:0.6px; border-bottom:2px solid #E5E7EB; }}
        .animated-table td {{ padding:11px 14px; color:#111827; border-bottom:1px solid #F3F4F6; }}
        .table-row {{ opacity:0; animation: rowFadeIn 0.4s ease-out forwards; transition: background 0.15s ease; }}
        .table-row:hover {{ background:#FAFAFA; }}
        @keyframes rowFadeIn {{ from {{ opacity:0; transform:translateX(-8px); }} to {{ opacity:1; transform:translateX(0); }} }}
        .bar-cell {{ display:flex; align-items:center; gap:8px; min-width:150px; }}
        .mini-bar-track {{ flex:1; background:#F3F4F6; border-radius:6px; height:10px; overflow:hidden; }}
        .mini-bar-fill {{ height:100%; border-radius:6px; width:0; background:linear-gradient(90deg,{color_start},{color_end});
                           animation: growMiniBar 1s cubic-bezier(0.22,1,0.36,1) forwards; }}
        .mini-bar-value {{ font-weight:700; width:40px; text-align:right; flex-shrink:0; color:#111827; }}
        @keyframes growMiniBar {{ from {{ width:0; }} to {{ width:var(--target); }} }}
        .rank-cell {{ font-weight:700; font-family:'Rajdhani',sans-serif; font-size:14px; }}
    </style>
    """


def initials(name: str) -> str:
    parts = [p for p in str(name).replace(".", " ").split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return str(name)[:2].upper()


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

try:
    history = pd.read_csv("f1_data/processed_features.csv")
except FileNotFoundError:
    history = None

name_map = {}
if history is not None and "FullName" in history.columns:
    latest = history.sort_values(["Season", "Round"]).groupby("Abbreviation")["FullName"].last()
    name_map = latest.to_dict()

summary["DriverName"] = summary["Driver"].map(name_map).fillna(summary["Driver"])

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
    <p>Real season standings, plus machine learning predictions from a
    5,000-race Monte Carlo simulation trained on historical race data.</p>
    <span class="badge">● Auto-updates weekly · Last updated {updated_str}</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TOP 3 — NEXT RACE FAVORITES
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
                <div class="avatar">{initials(row['DriverName'])}</div>
                <div class="medal">{medals[i]}</div>
                <div class="driver">{html_lib.escape(str(row['DriverName']))}</div>
                <div class="stat">{row['WinProb%']:.1f}%</div>
                <div class="label">Next Race Win Chance</div>
            </div>
            """, unsafe_allow_html=True)

st.write("")

# ============================================================
# TABS
# ============================================================
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Standings", "📊 Next Race Predictions", "📋 Full Table", "🔍 Driver Deep Dive", "📅 2025 Season"
])

# ------------------------------------------------------------
# TAB 0 — Real season standings, animated table
# ------------------------------------------------------------
with tab0:
    st.subheader("Championship Standings")

    if history is None or history.empty:
        st.info(
            "No historical race data found (f1_data/processed_features.csv). "
            "Run the data pipeline first to populate standings."
        )
    else:
        current_season = int(history["Season"].max())
        st.markdown(
            f'<div class="section-note">Season {current_season} — actual results '
            f'from every completed race, not predictions.</div>',
            unsafe_allow_html=True,
        )

        season_df = history[history["Season"] == current_season].copy()
        season_df = season_df.sort_values(["Round"])

        agg_dict = {
            "Points": ("Points", "sum"),
            "AvgFinish": ("Position", "mean"),
            "BestFinish": ("Position", "min"),
        }
        if "Win" in season_df.columns:
            agg_dict["Wins"] = ("Win", "sum")
        if "Podium" in season_df.columns:
            agg_dict["Podiums"] = ("Podium", "sum")

        standings = season_df.groupby("Abbreviation", as_index=False).agg(**agg_dict)
        latest_team = season_df.groupby("Abbreviation")["TeamName"].last()
        standings["Team"] = standings["Abbreviation"].map(latest_team)
        standings["Driver"] = standings["Abbreviation"].map(name_map).fillna(standings["Abbreviation"])

        standings = standings.sort_values("Points", ascending=False).reset_index(drop=True)
        standings.insert(0, "Rank", range(1, len(standings) + 1))
        standings["AvgFinish"] = standings["AvgFinish"].round(1)

        columns = [("Rank", "Rank"), ("Driver", "Driver"), ("Team", "Team"), ("Points", "Points")]
        if "Wins" in standings.columns:
            columns.append(("Wins", "Wins"))
        if "Podiums" in standings.columns:
            columns.append(("Podiums", "Podiums"))
        columns += [("AvgFinish", "Avg. Finish"), ("BestFinish", "Best Finish")]

        table_rows = standings.to_dict("records")
        table_html = animated_table_html(columns, table_rows, bar_col="Points", rank_col="Rank")

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        components.html(table_html, height=52 + 44 * len(table_rows) + 20, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 1 — Model predictions, animated
# ------------------------------------------------------------
with tab1:
    st.markdown(
        '<div class="section-note">Predicted probabilities for the next race, '
        'from the simulation — not season standings.</div>',
        unsafe_allow_html=True,
    )

    TOP_N = 8
    win_ranked = summary.sort_values("WinProb%", ascending=False).head(TOP_N)

    col_main, col_side = st.columns([3, 2])

    with col_main:
        st.markdown('<div class="panel"><h4>Win Probability</h4>', unsafe_allow_html=True)
        bar_data = list(zip(win_ranked["DriverName"], win_ranked["WinProb%"]))
        components.html(animated_bar_html(bar_data, unit="%"), height=44 * len(bar_data) + 20, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="panel"><h4>Outcome Breakdown — Top Favorite</h4>', unsafe_allow_html=True)
        favorite = win_ranked.iloc[0]
        fav_counts = raw[raw["Driver"] == favorite["Driver"]][["Position", "Count"]]
        total_sims = fav_counts["Count"].sum()
        if total_sims > 0:
            def _pct(mask):
                return fav_counts.loc[mask, "Count"].sum() / total_sims * 100

            win_pct = _pct(fav_counts["Position"] == 1)
            podium_pct = _pct((fav_counts["Position"] <= 3) & (fav_counts["Position"] > 1))
            points_pct = _pct((fav_counts["Position"] <= 10) & (fav_counts["Position"] > 3))
            other_pct = max(100 - win_pct - podium_pct - points_pct, 0)

            segments = [
                ("Win", win_pct, "#DC2626"),
                ("Podium (not win)", podium_pct, "#F87171"),
                ("Points (not podium)", points_pct, "#FCA5A5"),
                ("No points", other_pct, "#F3F4F6"),
            ]
            components.html(
                animated_donut_html(segments, center_label=str(favorite["DriverName"]).split()[0],
                                     center_sub=f"{total_sims:,} sims", size=190),
                height=250, scrolling=False,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel"><h4>Podium Probability</h4>', unsafe_allow_html=True)
        podium_ranked = summary.sort_values("PodiumProb%", ascending=False).head(TOP_N)
        bar_data = list(zip(podium_ranked["DriverName"], podium_ranked["PodiumProb%"]))
        components.html(
            animated_bar_html(bar_data, unit="%", color_start="#F87171", color_end="#DC2626"),
            height=44 * len(bar_data) + 20, scrolling=False,
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel"><h4>Points Probability</h4>', unsafe_allow_html=True)
        points_ranked = summary.sort_values("PointsProb%", ascending=False).head(TOP_N)
        bar_data = list(zip(points_ranked["DriverName"], points_ranked["PointsProb%"]))
        components.html(
            animated_bar_html(bar_data, unit="%", color_start="#FCA5A5", color_end="#B91C1C"),
            height=44 * len(bar_data) + 20, scrolling=False,
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 2 — Full table, animated
# ------------------------------------------------------------
with tab2:
    st.subheader("Full Predictions Table")

    full_df = summary[["DriverName", "AvgFinish", "WinProb%", "PodiumProb%", "PointsProb%"]].copy()
    full_df["AvgFinish"] = full_df["AvgFinish"].round(1)
    full_df = full_df.sort_values("WinProb%", ascending=False).reset_index(drop=True)
    full_df.insert(0, "Rank", range(1, len(full_df) + 1))

    columns = [
        ("Rank", "Rank"), ("DriverName", "Driver"), ("AvgFinish", "Avg. Predicted Finish"),
        ("WinProb%", "Win %"), ("PodiumProb%", "Podium %"), ("PointsProb%", "Points %"),
    ]
    table_rows = full_df.to_dict("records")
    table_html = animated_table_html(columns, table_rows, bar_col="WinProb%", bar_max=100, rank_col="Rank")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    components.html(table_html, height=52 + 44 * len(table_rows) + 20, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# TAB 3 — Driver deep dive, animated
# ------------------------------------------------------------
with tab3:
    st.subheader("Predicted Finish Spread")
    st.caption(
        "Pick a driver to see the full range of finishing positions the "
        "simulation produced for them, not just their single average."
    )

    name_to_abbr = dict(zip(summary["DriverName"], summary["Driver"]))
    chosen_name = st.selectbox("Driver", summary["DriverName"].tolist())
    chosen_driver = name_to_abbr[chosen_name]

    driver_counts = raw[raw["Driver"] == chosen_driver][["Position", "Count"]].set_index("Position")["Count"]
    full_range = pd.Series(0, index=range(1, int(raw["Position"].max()) + 1))
    position_counts = (driver_counts.reindex(full_range.index, fill_value=0)).astype(int)

    bar_data = [(f"P{pos}", count) for pos, count in position_counts.items()]

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    components.html(animated_bar_html(bar_data, unit="", color_start="#F87171", color_end="#DC2626"),
                     height=32 * len(bar_data) + 20, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)

    nonzero = position_counts[position_counts > 0]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Most likely finish", int(position_counts.idxmax()))
    col_b.metric("Best simulated finish", int(nonzero.index.min()))
    col_c.metric("Worst simulated finish", int(nonzero.index.max()))

# ------------------------------------------------------------
# TAB 4 — 2025 season archive
# ------------------------------------------------------------
with tab4:
    st.subheader("2025 Season — Final Standings")
    st.markdown(
        '<div class="section-note">Archived results. The model and predictions '
        'elsewhere in this dashboard are based on 2026 data only, since 2026 runs '
        'under different technical regulations — this tab is a historical '
        'reference, not part of the live prediction pipeline.</div>',
        unsafe_allow_html=True,
    )

    try:
        archive_2025 = pd.read_csv("f1_data/race_results_2025.csv")
    except FileNotFoundError:
        archive_2025 = None

    if archive_2025 is None or archive_2025.empty:
        st.info("No 2025 season file found (f1_data/race_results_2025.csv).")
    else:
        # This raw file was never run through build_features.py, so Win/
        # Podium flags don't exist yet — derive them directly from Position.
        archive_2025["Win"] = (archive_2025["Position"] == 1).astype(int)
        archive_2025["Podium"] = (archive_2025["Position"] <= 3).astype(int)

        archive_standings = archive_2025.groupby("Abbreviation", as_index=False).agg(
            Points=("Points", "sum"),
            Wins=("Win", "sum"),
            Podiums=("Podium", "sum"),
            AvgFinish=("Position", "mean"),
            BestFinish=("Position", "min"),
        )
        latest_team_2025 = archive_2025.groupby("Abbreviation")["TeamName"].last()
        archive_standings["Team"] = archive_standings["Abbreviation"].map(latest_team_2025)

        if "FullName" in archive_2025.columns:
            latest_name_2025 = archive_2025.groupby("Abbreviation")["FullName"].last()
            archive_standings["Driver"] = archive_standings["Abbreviation"].map(latest_name_2025)
            archive_standings["Driver"] = archive_standings["Driver"].fillna(archive_standings["Abbreviation"])
        else:
            archive_standings["Driver"] = archive_standings["Abbreviation"]

        archive_standings = archive_standings.sort_values("Points", ascending=False).reset_index(drop=True)
        archive_standings.insert(0, "Rank", range(1, len(archive_standings) + 1))
        archive_standings["AvgFinish"] = archive_standings["AvgFinish"].round(1)

        columns = [
            ("Rank", "Rank"), ("Driver", "Driver"), ("Team", "Team"), ("Points", "Points"),
            ("Wins", "Wins"), ("Podiums", "Podiums"), ("AvgFinish", "Avg. Finish"), ("BestFinish", "Best Finish"),
        ]
        table_rows = archive_standings.to_dict("records")
        table_html = animated_table_html(columns, table_rows, bar_col="Points", rank_col="Rank")

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        components.html(table_html, height=52 + 44 * len(table_rows) + 20, scrolling=False)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(f"""
<div class="footer-note">
    Standings reflect actual race results. Predictions are model-generated
    probabilities for the next race, not certainties — motorsport is
    inherently unpredictable. Pipeline auto-updates weekly via GitHub Actions.
</div>
""", unsafe_allow_html=True)
