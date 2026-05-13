from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


DB_PATH = Path("db/blueberry_trials.duckdb")

st.set_page_config(
    page_title="Blueberry R&D Data Product",
    layout="wide"
)

st.title("Blueberry R&D Data Product")
st.markdown(
    """
    Prototype data product for horticultural R&D and breeding decision support.

    This dashboard integrates blueberry yield simulation data, environmental time-series,
    data quality checks, and model outputs.
    """
)


@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    con.close()
    return df


yield_df = load_table("blueberry_yield")
weather_df = load_table("weather_daily")
quality_df = pd.read_csv("data/processed/data_quality_report.csv")
metrics_df = pd.read_csv("data/processed/model_metrics.csv")
importance_df = pd.read_csv("data/processed/model_feature_importance.csv")


st.sidebar.header("Controls")

numeric_columns = yield_df.select_dtypes(include="number").columns.tolist()
feature_options = [
    col for col in numeric_columns
    if col not in ["simulation_id", "yield"]
]

selected_feature = st.sidebar.selectbox(
    "Select variable to compare with yield",
    feature_options,
    index=0
)

weather_columns = [
    col for col in weather_df.select_dtypes(include="number").columns.tolist()
]

selected_weather = st.sidebar.selectbox(
    "Select weather variable",
    weather_columns,
    index=0
)


st.header("1. Data overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Yield records", len(yield_df))
col2.metric("Weather days", len(weather_df))
col3.metric("Mean yield", f"{yield_df['yield'].mean():.1f}")
col4.metric("Failed QC checks", int((quality_df["status"] == "FAIL").sum()))


st.header("2. Blueberry yield analysis")

col_left, col_right = st.columns(2)

with col_left:
    fig = px.scatter(
        yield_df,
        x=selected_feature,
        y="yield",
        trendline="ols",
        title=f"Yield vs {selected_feature}"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    fig = px.histogram(
        yield_df,
        x="yield",
        nbins=30,
        title="Distribution of simulated blueberry yield"
    )
    st.plotly_chart(fig, use_container_width=True)


st.header("3. Environmental time-series: Sevilla, Spain")

fig = px.line(
    weather_df,
    x="date",
    y=selected_weather,
    title=f"Daily {selected_weather} in Sevilla, 2023"
)
st.plotly_chart(fig, use_container_width=True)


st.header("4. Data quality report")

st.dataframe(quality_df, use_container_width=True)

if (quality_df["status"] == "FAIL").any():
    st.warning("Some data quality checks failed.")
else:
    st.success("All data quality checks passed.")


st.header("5. Yield model")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Model metrics")
    st.dataframe(metrics_df, use_container_width=True)

with col_right:
    st.subheader("Top predictors")
    top_importance = importance_df.head(10)

    fig = px.bar(
        top_importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 feature importances"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


st.header("6. Raw data preview")

with st.expander("Blueberry yield data"):
    st.dataframe(yield_df.head(50), use_container_width=True)

with st.expander("Weather data"):
    st.dataframe(weather_df.head(50), use_container_width=True)
