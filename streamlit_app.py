# file: movielens_kpi_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import altair as alt

st.set_page_config(page_title="MovieLens Ratings Dashboard", layout="wide")

@st.cache_data
def load_data():
    path_candidates = ["data/movie_ratings.csv", "movie_ratings.csv"]
    for p in path_candidates:
        try:
            df = pd.read_csv(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "rating"])
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 5)
            df = df.rename(columns={"timestamp": "DATE"})
            if "genres" in df.columns:
                df["genres"] = df["genres"].fillna("").astype(str)
            return df
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Couldn't find data/movie_ratings.csv (or movie_ratings.csv).")

def aggregate_movielens(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "Q":
        tmp = df.copy()
        tmp["QPER"] = tmp["DATE"].dt.to_period("Q")
        g = tmp.groupby("QPER")
    elif freq == "W":
        g = df.groupby(pd.Grouper(key="DATE", freq="W-MON"))
    else:  # 'D' or 'M'
        g = df.groupby(pd.Grouper(key="DATE", freq=freq))
    out = g.agg(
        N_RATINGS=("rating", "size"),
        MEAN_RATING=("rating", "mean"),
        N_USERS=("user_id", pd.Series.nunique),
        N_MOVIES=("movie_id", pd.Series.nunique),
    )
    if freq == "Q":
        out.index = out.index.astype(str)  # 'YYYYQ#'
    return out

def is_period_complete(last_index_label, freq: str) -> bool:
    now = datetime.now()
    if freq == "D":
        return pd.to_datetime(last_index_label).date() < now.date()
    if freq == "W":
        d = pd.to_datetime(last_index_label)
        return d + timedelta(days=6) < now
    if freq == "M":
        d = pd.to_datetime(last_index_label)
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        return nxt <= now
    if freq == "Q":
        last_q = pd.Period(last_index_label, freq="Q")
        curr_q = pd.Period(pd.Timestamp.now(), freq="Q")
        return last_q < curr_q
    return True

def format_with_commas(x):
    if pd.isna(x):
        return "—"
    if isinstance(x, float) and x != int(x):
        return f"{x:,.2f}"
    return f"{int(x):,}"

def create_metric_chart(ts_df: pd.DataFrame, column: str, chart_type: str, height=150):
    chart_data = ts_df[[column]].copy()
    if chart_type == "Bar":
        st.bar_chart(chart_data, y=column, height=height, use_container_width=True)
    else:
        st.area_chart(chart_data, y=column, height=height, use_container_width=True)

def calculate_delta(ts_df: pd.DataFrame, column: str):
    if len(ts_df) < 2:
        return 0.0, 0.0
    curr = ts_df[column].iloc[-1]
    prev = ts_df[column].iloc[-2]
    delta = (curr - prev) if pd.notna(curr) and pd.notna(prev) else 0.0
    pct = (delta / prev * 100.0) if pd.notna(prev) and prev != 0 else 0.0
    return float(delta), float(pct)

def display_metric(col, title, total_value, ts_df, series_col, chart_type, time_frame):
    with col:
        with st.container(border=True):
            d, p = calculate_delta(ts_df, series_col)
            if series_col == "MEAN_RATING":
                primary_val = f"{total_value:.2f}" if pd.notna(total_value) else "—"
                delta_str = f"{d:+.2f} ({p:+.2f}%)"
            else:
                primary_val = format_with_commas(total_value)
                delta_str = f"{d:+,.0f} ({p:+.2f}%)"
            st.metric(title, primary_val, delta=delta_str)
            create_metric_chart(ts_df, series_col, chart_type=chart_type)

            last_label = ts_df.index[-1]
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
            unit_map = {"D": "day", "W": "week", "M": "month", "Q": "quarter"}
            f = freq_map[time_frame]
            if not is_period_complete(last_label, f):
                st.caption(f"Note: The last {unit_map[f]} is incomplete.")

# Load  

df = load_data()
try:
    st.logo(
        image="streamlit-logo-primary-colormark-lighttext.png",
        icon_image="streamlit-mark-color.png",
    )
except Exception:
    pass

with st.sidebar:
    st.title("MovieLens Dashboard")
    st.header("⚙️ Settings")

    min_date = df["DATE"].min().date()
    max_date = df["DATE"].max().date()
    default_start = max(min_date, max_date - timedelta(days=365))

    start_date, end_date = st.date_input(
        "Date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    time_frame = st.selectbox(
        "Select time frame",
        ("Daily", "Weekly", "Monthly", "Quarterly"),
        index=2,
    )

    chart_selection = st.selectbox(
        "Select a chart type",
        ("Bar", "Area"),
        index=0,
    )


# Filter and time series

mask = (df["DATE"].dt.date >= start_date) & (df["DATE"].dt.date <= end_date)
df_f = df.loc[mask].copy()

if time_frame == "Daily":
    ts = aggregate_movielens(df_f, "D")
elif time_frame == "Weekly":
    ts = aggregate_movielens(df_f, "W")
elif time_frame == "Monthly":
    ts = aggregate_movielens(df_f, "M")
else:
    ts = aggregate_movielens(df_f, "Q")


# KPI row

st.subheader("All-Time (within filter)")
kpis = [
    ("Total Ratings", "N_RATINGS"),
    ("Average Rating", "MEAN_RATING"),
    ("Unique Users", "N_USERS"),
    ("Unique Movies", "N_MOVIES"),
]
totals = {
    "N_RATINGS": int(len(df_f)),
    "MEAN_RATING": float(df_f["rating"].mean()) if len(df_f) else np.nan,
    "N_USERS": int(df_f["user_id"].nunique()),
    "N_MOVIES": int(df_f["movie_id"].nunique()),
}
cols = st.columns(4)
for col, (title, series) in zip(cols, kpis):
    display_metric(col, title, totals[series], ts, series, chart_selection, time_frame)

st.markdown("---")


# Extra A: Ratings per Genre

if "genres" in df_f.columns and df_f["genres"].notna().any():
    expl = df_f.assign(genre=df_f["genres"].str.split("|")).explode("genre")
    expl["genre"] = expl["genre"].str.strip()
    expl = expl[expl["genre"] != ""]

    st.subheader("Ratings per Genre")
    genre_counts = (
        expl.groupby("genre", as_index=False)
            .agg(count=("rating", "size"))
            .sort_values("count", ascending=False)
    )
    chart = (
        alt.Chart(genre_counts)
        .mark_bar()
        .encode(
            y=alt.Y("genre:N", sort="-x", title="genre"),
            x=alt.X("count:Q", title="count",
                    scale=alt.Scale(domain=[0, float(genre_counts["count"].max() * 1.05)])),
            tooltip=["genre", "count"],
        )
        .properties(height=max(320, 18 * len(genre_counts)))
        .configure_view(stroke="#999", strokeWidth=1)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Exploded: each rating counted once for every genre a movie has.")

    st.markdown("---")


    # Extra B: Mean Rating by Genre
   
    st.subheader("Viewer Satisfaction by Genre (mean rating)")
    MIN_N = st.slider("Min ratings per genre", min_value=0, max_value=300, value=50, step=10)
    genre_stats = (
        expl.groupby("genre", as_index=False)
            .agg(n=("rating", "size"), mean_rating=("rating", "mean"))
    )
    genre_stats = genre_stats[genre_stats["n"] >= MIN_N].sort_values(
        ["mean_rating", "n"], ascending=[False, False]
    )
    chart2 = (
        alt.Chart(genre_stats)
        .mark_circle(size=90)
        .encode(
            x=alt.X("mean_rating:Q", title="Mean Rating (1–5)",
                    scale=alt.Scale(domain=[1, 5])),
            y=alt.Y("genre:N", sort="-x", title="genre"),
            size=alt.Size("n:Q", title="n (ratings)"),
            tooltip=["genre", alt.Tooltip("mean_rating:Q", format=".2f"), "n:Q"],
        )
        .properties(height=max(320, 18 * len(genre_stats)))
        .configure_view(stroke="#999", strokeWidth=1)
    )
    st.altair_chart(chart2, use_container_width=True)
