from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


DB_PATH = Path("db/blueberry_trials.duckdb")
QUALITY_PATH = Path("data/processed/data_quality_report.csv")
METRICS_PATH = Path("data/processed/model_metrics.csv")
IMPORTANCE_PATH = Path("data/processed/model_feature_importance.csv")


st.set_page_config(
    page_title="Blueberry R&D Data Product",
    page_icon="🫐",
    layout="wide",
)


def pretty_name(name: str) -> str:
    return name.replace("_", " ").title()


@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        st.stop()

    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    con.close()

    return df


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()

    return pd.read_csv(path)


yield_df = load_table("blueberry_yield")
weather_df = load_table("weather_daily")
quality_df = load_csv(QUALITY_PATH)
metrics_df = load_csv(METRICS_PATH)
importance_df = load_csv(IMPORTANCE_PATH)

if "date" in weather_df.columns:
    weather_df["date"] = pd.to_datetime(weather_df["date"])


st.title("🫐 Blueberry R&D Data Product")

st.markdown(
    """
    End-to-end prototype for horticultural R&D and breeding decision support.

    This dashboard integrates blueberry yield data, environmental time-series,
    automated data quality checks, SQL-based storage, and machine learning outputs.
    """
)


with st.expander("Project context"):
    st.markdown(
        """
        This project mimics a small data product for a breeding or horticultural R&D team.

        It is designed to demonstrate:

        - integration of phenotypic and environmental data;
        - automated data quality checks;
        - SQL-based data storage;
        - yield prediction modeling;
        - stakeholder-facing visualization;
        - communication of technical results to non-technical users.
        """
    )


st.sidebar.header("Dashboard controls")

numeric_yield_columns = yield_df.select_dtypes(include="number").columns.tolist()

feature_options = [
    col for col in numeric_yield_columns
    if col not in ["simulation_id", "yield"]
]

if not feature_options:
    st.sidebar.error("No numeric features available for yield analysis.")
    st.stop()

selected_feature = st.sidebar.selectbox(
    "Variable to compare with yield",
    feature_options,
    format_func=pretty_name,
)

weather_numeric_columns = weather_df.select_dtypes(include="number").columns.tolist()

if not weather_numeric_columns:
    st.sidebar.error("No numeric weather variables available.")
    st.stop()

selected_weather = st.sidebar.selectbox(
    "Weather variable",
    weather_numeric_columns,
    format_func=pretty_name,
)


qc_failed = int((quality_df["status"] == "FAIL").sum())
qc_passed = int((quality_df["status"] == "PASS").sum())
qc_total = len(quality_df)
qc_pass_rate = qc_passed / qc_total if qc_total > 0 else 0

mean_yield = yield_df["yield"].mean()
min_yield = yield_df["yield"].min()
max_yield = yield_df["yield"].max()

model_r2 = metrics_df.loc[0, "r2"] if "r2" in metrics_df.columns else None
model_mae = metrics_df.loc[0, "mae"] if "mae" in metrics_df.columns else None

top_predictor = (
    importance_df.iloc[0]["feature"]
    if not importance_df.empty and "feature" in importance_df.columns
    else "Not available"
)


tab_overview, tab_yield, tab_weather, tab_quality, tab_model, tab_data = st.tabs(
    [
        "Executive overview",
        "Yield analysis",
        "Environmental data",
        "Data quality & provenance",
        "Model outputs",
        "Data preview",
    ]
)


with tab_overview:
    st.header("Executive overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Yield records", f"{len(yield_df):,}")
    col2.metric("Weather days", f"{len(weather_df):,}")
    col3.metric("Mean yield", f"{mean_yield:,.1f}")
    col4.metric("QC pass rate", f"{qc_pass_rate:.0%}")

    st.subheader("Current data product status")

    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        if qc_failed == 0:
            st.success("Data quality checks passed")
        else:
            st.error(f"{qc_failed} data quality checks failed")

    with status_col2:
        if model_r2 is not None:
            st.info(f"Yield model R²: {model_r2:.3f}")
        else:
            st.warning("Model R² not available")

    with status_col3:
        st.info(f"Top model predictor: {pretty_name(str(top_predictor))}")

    st.subheader("Interpretation for R&D stakeholders")

    st.markdown(
        f"""
        The prototype combines **{len(yield_df):,} blueberry yield records**
        with **{len(weather_df):,} daily weather observations**.

        The current dataset passed **{qc_passed}/{qc_total} automated quality checks**.
        The yield prediction model achieved an R² of **{model_r2:.3f}**
        with a mean absolute error of **{model_mae:,.2f}**, using the available numeric predictors.

        The strongest predictor in the current model is **{pretty_name(str(top_predictor))}**.
        In a real breeding workflow, this type of output could help prioritize which
        phenotypic, environmental, or management variables deserve closer investigation.
        """
    )

    st.subheader("Prototype architecture")

    architecture_df = pd.DataFrame(
        [
            {
                "Layer": "Raw data",
                "Example": "Blueberry yield CSV, NASA POWER weather API",
                "Purpose": "Capture heterogeneous phenotypic and environmental data",
            },
            {
                "Layer": "Processing",
                "Example": "Python ETL scripts",
                "Purpose": "Clean, standardize, and transform data",
            },
            {
                "Layer": "Storage",
                "Example": "DuckDB database",
                "Purpose": "Provide SQL-accessible analysis-ready tables",
            },
            {
                "Layer": "Quality control",
                "Example": "Automated validation checks",
                "Purpose": "Detect missing, negative, duplicated, or impossible values",
            },
            {
                "Layer": "Modeling",
                "Example": "Random forest yield model",
                "Purpose": "Estimate yield and rank predictive variables",
            },
            {
                "Layer": "Dashboard",
                "Example": "Streamlit app",
                "Purpose": "Communicate results to technical and non-technical users",
            },
        ]
    )

    st.dataframe(architecture_df, use_container_width=True, hide_index=True)


with tab_yield:
    st.header("Yield analysis")

    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.scatter(
            yield_df,
            x=selected_feature,
            y="yield",
            trendline="ols",
            title=f"Yield vs {pretty_name(selected_feature)}",
            labels={
                selected_feature: pretty_name(selected_feature),
                "yield": "Yield",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.histogram(
            yield_df,
            x="yield",
            nbins=30,
            title="Distribution of blueberry yield",
            labels={"yield": "Yield"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Yield summary statistics")

    summary = yield_df["yield"].describe().reset_index()
    summary.columns = ["Statistic", "Value"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        Yield ranges from **{min_yield:,.1f}** to **{max_yield:,.1f}**,
        with an average of **{mean_yield:,.1f}** across the simulated records.
        """
    )


with tab_weather:
    st.header("Environmental data")

    fig = px.line(
        weather_df,
        x="date",
        y=selected_weather,
        title=f"Daily {pretty_name(selected_weather)} in Sevilla, Spain",
        labels={
            "date": "Date",
            selected_weather: pretty_name(selected_weather),
        },
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weather variable summary")

    weather_summary = (
        weather_df[weather_numeric_columns]
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    weather_summary["variable"] = weather_summary["variable"].apply(pretty_name)

    st.dataframe(weather_summary, use_container_width=True, hide_index=True)


with tab_quality:
    st.header("Data quality & provenance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Checks passed", qc_passed)
    col2.metric("Checks failed", qc_failed)
    col3.metric("Total checks", qc_total)

    if qc_failed == 0:
        st.success("All automated data quality checks passed.")
    else:
        st.error("Some automated data quality checks failed.")

    st.subheader("Quality control report")
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.subheader("Data provenance")

    provenance_df = pd.DataFrame(
        [
            {
                "Dataset": "Blueberry yield",
                "Source": "Wild blueberry pollination simulation dataset",
                "Local table": "blueberry_yield",
                "Role": "Phenotypic and yield-related variables",
            },
            {
                "Dataset": "Weather data",
                "Source": "NASA POWER API",
                "Local table": "weather_daily",
                "Role": "Daily environmental time-series for Sevilla, Spain",
            },
            {
                "Dataset": "Model metrics",
                "Source": "Generated by scripts/04_train_yield_model.py",
                "Local table": "model_metrics",
                "Role": "Model performance tracking",
            },
            {
                "Dataset": "Feature importance",
                "Source": "Generated by scripts/04_train_yield_model.py",
                "Local table": "model_feature_importance",
                "Role": "Predictor ranking for model interpretation",
            },
        ]
    )

    st.dataframe(provenance_df, use_container_width=True, hide_index=True)


with tab_model:
    st.header("Model outputs")

    st.markdown(
        """
        A random forest regression model was trained to predict blueberry yield
        from the available numeric predictors. The goal is not to claim biological
        causality, but to demonstrate how model outputs can be integrated into
        a reproducible R&D data product.
        """
    )

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Model metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Top predictors")

        top_importance = importance_df.head(10).copy()
        top_importance["feature_label"] = top_importance["feature"].apply(pretty_name)

        fig = px.bar(
            top_importance,
            x="importance",
            y="feature_label",
            orientation="h",
            title="Top 10 feature importances",
            labels={
                "importance": "Importance",
                "feature_label": "Feature",
            },
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How to interpret this model")

    st.markdown(
        """
        For a breeding or horticultural team, this type of model can be useful for:

        - ranking variables associated with yield variation;
        - identifying data fields worth improving or collecting more consistently;
        - creating early prototypes for decision-support tools;
        - opening discussions between data scientists, breeders, agronomists, and IT teams.

        In a production context, this model should be extended with real trial structure,
        including genotype, plot, block, location, year, and management metadata.
        """
    )


with tab_data:
    st.header("Data preview")

    with st.expander("Blueberry yield table"):
        st.dataframe(yield_df.head(100), use_container_width=True)

    with st.expander("Weather daily table"):
        st.dataframe(weather_df.head(100), use_container_width=True)

    with st.expander("Model feature importance table"):
        st.dataframe(importance_df, use_container_width=True)

    with st.expander("Model metrics table"):
        st.dataframe(metrics_df, use_container_width=True)
