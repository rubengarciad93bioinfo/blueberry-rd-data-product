from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st


DB_PATH = Path("db/blueberry_trials.duckdb")
QUALITY_PATH = Path("data/processed/data_quality_report.csv")
METRICS_PATH = Path("data/processed/model_metrics.csv")
IMPORTANCE_PATH = Path("data/processed/model_feature_importance.csv")


DATA_DICTIONARY = [
    {
        "variable": "clonesize",
        "category": "Plant / field structure",
        "plain_english": "Approximate size of the wild blueberry clone or patch.",
        "interpretation": "Represents spatial structure of the plant population.",
        "model_note": "Useful contextual predictor.",
    },
    {
        "variable": "honeybee",
        "category": "Pollinator density/activity",
        "plain_english": "Honey bee density or activity in the simulated field scenario.",
        "interpretation": "Not a proportion. High values can represent high-density managed pollination scenarios.",
        "model_note": "Should be interpreted as a simulated pollinator input, not as a percentage.",
    },
    {
        "variable": "bumbles",
        "category": "Pollinator density/activity",
        "plain_english": "Bumble bee density or activity.",
        "interpretation": "Usually takes a limited number of simulated levels.",
        "model_note": "Better visualized with boxplots when values are discrete-like.",
    },
    {
        "variable": "andrena",
        "category": "Pollinator density/activity",
        "plain_english": "Density or activity of Andrena bees.",
        "interpretation": "Represents a native bee pollinator group.",
        "model_note": "Useful for comparing pollinator scenarios.",
    },
    {
        "variable": "osmia",
        "category": "Pollinator density/activity",
        "plain_english": "Density or activity of Osmia bees.",
        "interpretation": "Represents a native bee pollinator group.",
        "model_note": "Useful for comparing pollinator scenarios.",
    },
    {
        "variable": "maxofuppertrange",
        "category": "Simulated weather",
        "plain_english": "Maximum value of the upper temperature range during the simulated period.",
        "interpretation": "Represents warm temperature conditions in the simulation.",
        "model_note": "Weather predictor already included in the yield dataset.",
    },
    {
        "variable": "minofuppertrange",
        "category": "Simulated weather",
        "plain_english": "Minimum value of the upper temperature range.",
        "interpretation": "Represents variability in warmer daily temperature conditions.",
        "model_note": "Weather predictor already included in the yield dataset.",
    },
    {
        "variable": "averageofuppertrange",
        "category": "Simulated weather",
        "plain_english": "Average value of the upper temperature range.",
        "interpretation": "Summarizes warmer temperature conditions.",
        "model_note": "Weather predictor already included in the yield dataset.",
    },
    {
        "variable": "maxoflowertrange",
        "category": "Simulated weather",
        "plain_english": "Maximum value of the lower temperature range.",
        "interpretation": "Represents cooler temperature conditions in the simulation.",
        "model_note": "Weather predictor already included in the yield dataset.",
    },
    {
        "variable": "minoflowertrange",
        "category": "Simulated weather",
        "plain_english": "Minimum value of the lower temperature range.",
        "interpretation": "Represents the lower bound of cooler temperature conditions.",
        "model_note": "Weather predictor already included in the yield dataset.",
    },
    {
        "variable": "averageoflowertrange",
        "category": "Simulated weather",
        "plain_english": "Average value of the lower temperature range.",
        "interpretation": "Summarizes cooler temperature conditions.",
        "model_note": "Weather predictor already included in the yield dataset.",
    },
    {
        "variable": "rainingdays",
        "category": "Simulated weather",
        "plain_english": "Number of rainy days during the simulated flowering or growing period.",
        "interpretation": "Rain can affect pollinator activity and fruit development.",
        "model_note": "Useful environmental predictor.",
    },
    {
        "variable": "averagerainingdays",
        "category": "Simulated weather",
        "plain_english": "Average rainy days in the simulated period.",
        "interpretation": "Summarizes rainfall frequency.",
        "model_note": "Useful environmental predictor.",
    },
    {
        "variable": "fruitset",
        "category": "Fruit trait / outcome-adjacent",
        "plain_english": "Fruit set: how successfully flowers develop into berries after pollination.",
        "interpretation": "A biological indicator of successful pollination and fruit formation.",
        "model_note": "Highly informative, but may be too close to the final outcome for early yield prediction.",
    },
    {
        "variable": "fruitmass",
        "category": "Fruit trait / outcome-adjacent",
        "plain_english": "Average fruit mass.",
        "interpretation": "Larger fruits usually contribute directly to higher yield.",
        "model_note": "Very predictive, but can introduce leakage if the goal is early prediction.",
    },
    {
        "variable": "seeds",
        "category": "Fruit trait / outcome-adjacent",
        "plain_english": "Number of seeds.",
        "interpretation": "Related to pollination success and fruit development.",
        "model_note": "Useful biological trait, but likely outcome-adjacent.",
    },
    {
        "variable": "yield",
        "category": "Target variable",
        "plain_english": "Final simulated blueberry crop yield.",
        "interpretation": "This is the outcome the model tries to predict.",
        "model_note": "Target variable, not used as a predictor.",
    },
]


OUTCOME_ADJACENT_FEATURES = {"fruitset", "fruitmass", "seeds"}
ID_OR_INDEX_FEATURES = {"simulation_id", "row#", "row", "index"}

st.set_page_config(
    page_title="Blueberry R&D Data Product",
    page_icon="🫐",
    layout="wide",
)


def pretty_name(name: str) -> str:
    return name.replace("_", " ").title()


def get_variable_info(variable: str) -> dict:
    for row in DATA_DICTIONARY:
        if row["variable"] == variable:
            return row

    return {
        "variable": variable,
        "category": "Unknown",
        "plain_english": "No description available yet.",
        "interpretation": "This variable needs additional metadata.",
        "model_note": "Add to data dictionary.",
    }


def is_discrete_like(series: pd.Series, max_unique: int = 12) -> bool:
    return series.nunique(dropna=True) <= max_unique


def file_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_mtime


@st.cache_data
def load_table(table_name: str, db_mtime: float) -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        st.stop()

    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    con.close()

    return df


@st.cache_data
def load_csv(path: Path, csv_mtime: float) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()

    return pd.read_csv(path)


def file_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_mtime


@st.cache_data
def load_table(table_name: str, db_mtime: float) -> pd.DataFrame:
    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        st.stop()

    con = duckdb.connect(str(DB_PATH))
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    con.close()

    return df


@st.cache_data
def load_csv(path: Path, csv_mtime: float) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()

    return pd.read_csv(path)


yield_df = load_table("blueberry_yield", file_mtime(DB_PATH))
weather_df = load_table("weather_daily", file_mtime(DB_PATH))
quality_df = load_csv(QUALITY_PATH, file_mtime(QUALITY_PATH))
metrics_df = load_csv(METRICS_PATH, file_mtime(METRICS_PATH))
importance_df = load_csv(IMPORTANCE_PATH, file_mtime(IMPORTANCE_PATH))

if "model" not in metrics_df.columns:
    metrics_df["model"] = "full_model"

if "model_label" not in metrics_df.columns:
    metrics_df["model_label"] = metrics_df["model"].replace(
        {
            "full_model": "Full model: all numeric predictors",
            "pre_harvest_model": "Pre-harvest model: excluding fruit traits",
            "RandomForestRegressor": "Random forest model",
        }
    )

if "excluded_features" not in metrics_df.columns:
    metrics_df["excluded_features"] = "Not available"

if "model" not in importance_df.columns:
    importance_df["model"] = "full_model"

if "model_label" not in importance_df.columns:
    importance_df["model_label"] = importance_df["model"].replace(
        {
            "full_model": "Full model: all numeric predictors",
            "pre_harvest_model": "Pre-harvest model: excluding fruit traits",
            "RandomForestRegressor": "Random forest model",
        }
    )

if "date" in weather_df.columns:
    weather_df["date"] = pd.to_datetime(weather_df["date"])


st.title("🫐 Blueberry R&D Data Product")

st.markdown(
    """
    End-to-end prototype for horticultural R&D and breeding decision support.

    This dashboard integrates blueberry yield simulation data, environmental data ingestion,
    automated data quality checks, SQL-based storage, model outputs, and plain-language
    documentation for R&D stakeholders.
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
        - metadata documentation and data provenance;
        - critical interpretation of data limitations.
        """
    )


st.sidebar.header("Dashboard controls")
st.sidebar.caption(
    f"Model output version: {pd.to_datetime(file_mtime(METRICS_PATH), unit='s')}"
)
numeric_yield_columns = yield_df.select_dtypes(include="number").columns.tolist()

feature_options = [
    col for col in numeric_yield_columns
    if col not in ID_OR_INDEX_FEATURES
    and col != "yield"
]

selected_feature = st.sidebar.selectbox(
    "Variable to compare with yield",
    feature_options,
    format_func=pretty_name,
)

weather_numeric_columns = weather_df.select_dtypes(include="number").columns.tolist()

selected_weather = st.sidebar.selectbox(
    "External weather variable",
    weather_numeric_columns,
    format_func=pretty_name,
)

model_options = metrics_df["model"].tolist()
model_labels = dict(zip(metrics_df["model"], metrics_df["model_label"]))

default_model_index = (
    model_options.index("pre_harvest_model")
    if "pre_harvest_model" in model_options
    else 0
)

selected_model = st.sidebar.selectbox(
    "Model interpretation",
    model_options,
    index=default_model_index,
    format_func=lambda model: model_labels.get(model, model),
)


qc_failed = int((quality_df["status"] == "FAIL").sum())
qc_passed = int((quality_df["status"] == "PASS").sum())
qc_total = len(quality_df)
qc_pass_rate = qc_passed / qc_total if qc_total > 0 else 0

mean_yield = yield_df["yield"].mean()
min_yield = yield_df["yield"].min()
max_yield = yield_df["yield"].max()

selected_metrics = metrics_df.loc[metrics_df["model"] == selected_model].iloc[0]
selected_importance = importance_df.loc[
    importance_df["model"] == selected_model
].sort_values("importance", ascending=False)
if selected_importance.empty and selected_model == "RandomForestRegressor":
    selected_model_for_importance = "full_model"
    selected_importance = importance_df.loc[
        importance_df["model"] == selected_model_for_importance
    ].sort_values("importance", ascending=False)

if selected_importance.empty and "model_label" in importance_df.columns:
    selected_label = model_labels.get(selected_model)

    selected_importance = importance_df.loc[
        importance_df["model_label"] == selected_label
    ].sort_values("importance", ascending=False)
model_r2 = selected_metrics["r2"]
model_mae = selected_metrics["mae"]

top_predictor = (
    selected_importance.iloc[0]["feature"]
    if not selected_importance.empty
    else "Not available"
)


tab_overview, tab_decision, tab_yield, tab_weather, tab_quality, tab_model, tab_dictionary, tab_limitations, tab_data = st.tabs(
    [
        "Executive overview",
        "Decision support",
        "Yield analysis",
        "External weather ingestion",
        "Data quality & provenance",
        "Model outputs",
        "Data dictionary",
        "Known limitations",
        "Data preview",
    ]
)


with tab_overview:
    st.header("Executive overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Yield records", f"{len(yield_df):,}")
    col2.metric("External weather days", f"{len(weather_df):,}")
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
        st.info(f"{model_labels.get(selected_model, selected_model)} R²: {model_r2:.3f}")

    with status_col3:
        st.info(f"Top predictor: {pretty_name(str(top_predictor))}")

    st.subheader("Interpretation for R&D stakeholders")

    st.markdown(
        f"""
        The prototype contains **{len(yield_df):,} simulated blueberry yield records**
        and **{len(weather_df):,} daily external weather observations** for Sevilla, Spain.

        The core dataset passed **{qc_passed}/{qc_total} automated quality checks**.

        The selected model is **{model_labels.get(selected_model, selected_model)}**.
        It achieved an R² of **{model_r2:.3f}** with a mean absolute error of **{model_mae:,.2f}**.

        The strongest predictor in this model is **{pretty_name(str(top_predictor))}**.
        """
    )

    st.warning(
        """
        The external Sevilla weather time-series is included as an API ingestion example.
        It is not directly joined to yield records because the yield dataset does not contain
        real trial dates or geographic coordinates.
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
                "Example": "Full and pre-harvest random forest models",
                "Purpose": "Compare explanatory and more realistic prediction settings",
            },
            {
                "Layer": "Dashboard",
                "Example": "Streamlit app",
                "Purpose": "Communicate results to technical and non-technical users",
            },
        ]
    )

    st.dataframe(architecture_df, use_container_width=True, hide_index=True)


with tab_decision:
    st.header("Decision support")

    st.subheader("1. Are the data ready for exploratory analysis?")

    if qc_failed == 0:
        st.success(
            "Yes. The current prototype dataset passed all automated quality checks."
        )
    else:
        st.error(
            "Not yet. Some automated quality checks failed and should be reviewed before modeling."
        )

    st.subheader("2. Which variables appear most informative for yield?")

    top_features = selected_importance.head(5).copy()
    top_features["feature_label"] = top_features["feature"].apply(pretty_name)

    st.dataframe(
        top_features[["feature_label", "importance"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"""
        Under the selected model, the strongest predictor is
        **{pretty_name(str(top_predictor))}**.

        This does not prove causality. It indicates which variables the model used most
        strongly to explain variation in simulated yield.
        """
    )

    if any(feature in OUTCOME_ADJACENT_FEATURES for feature in top_features["feature"]):
        st.warning(
            """
            Some top predictors are fruit traits such as fruit set, fruit mass, or seeds.
            These are biologically meaningful, but they are close to the final outcome.
            They may be less useful for early-season prediction.
            """
        )

    st.subheader("3. How reliable is the current model?")

    st.markdown(
        f"""
        The selected model explains approximately **{model_r2:.1%}** of variation in the
        test set and has a mean absolute error of **{model_mae:,.2f}** yield units.

        For a real breeding workflow, this would be treated as exploratory decision support,
        not as a production-ready prediction system.
        """
    )

    st.subheader("4. What would improve this data product most?")

    priority_df = pd.DataFrame(
        [
            {
                "Priority": "Add trial metadata",
                "Why it matters": "Genotype, plot, block, location, year, and management data would allow real breeding trial analysis.",
            },
            {
                "Priority": "Join real weather by date and location",
                "Why it matters": "Environmental variables only become analytically useful when linked to each trial observation.",
            },
            {
                "Priority": "Separate early predictors from outcome-adjacent traits",
                "Why it matters": "Improves model interpretation and avoids overly optimistic yield prediction.",
            },
            {
                "Priority": "Add mixed-effect models",
                "Why it matters": "Breeding trials often need to account for genotype, environment, block, and repeated observations.",
            },
        ]
    )

    st.dataframe(priority_df, use_container_width=True, hide_index=True)


with tab_yield:
    st.header("Yield analysis")

    variable_info = get_variable_info(selected_feature)

    st.subheader(f"Selected variable: {pretty_name(selected_feature)}")

    st.markdown(
        f"""
        **Category:** {variable_info["category"]}

        **Meaning:** {variable_info["plain_english"]}

        **Interpretation:** {variable_info["interpretation"]}

        **Model note:** {variable_info["model_note"]}
        """
    )

    unique_values = yield_df[selected_feature].nunique(dropna=True)

    col_left, col_right = st.columns(2)

    with col_left:
        if is_discrete_like(yield_df[selected_feature]):
            plot_df = yield_df[[selected_feature, "yield"]].copy()
            plot_df["feature_level"] = plot_df[selected_feature].round(4).astype(str)

            fig = px.box(
                plot_df,
                x="feature_level",
                y="yield",
                points="all",
                title=f"Yield distribution by {pretty_name(selected_feature)} level",
                labels={
                    "feature_level": pretty_name(selected_feature),
                    "yield": "Yield",
                },
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info(
                f"""
                This variable has only **{unique_values} unique values**, so a boxplot is more
                appropriate than a standard scatterplot. This avoids over-interpreting vertical
                stacks of repeated values.
                """
            )

        else:
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

            st.info(
                f"""
                This variable has **{unique_values} unique values**, so a scatterplot with a
                trend line is appropriate for exploring continuous variation.
                """
            )

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
    st.header("External weather ingestion")

    st.warning(
        """
        This section demonstrates API-based environmental data ingestion for Sevilla, Spain.

        These weather records are not directly joined to the blueberry yield records, because
        the yield dataset is simulated and does not include real trial dates, field locations,
        or coordinates.
        """
    )

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

    st.subheader("How this would be used in a real R&D setting")

    st.markdown(
        """
        In a production breeding or horticultural workflow, weather data would be joined to
        each field observation using:

        - trial location;
        - GPS coordinates;
        - planting date;
        - flowering date;
        - harvest date;
        - observation date.

        This would allow derived features such as rainfall before harvest, heat accumulation,
        growing degree days, or stress periods to be included in yield and quality models.
        """
    )


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
                "Role": "Simulated phenotypic, pollinator, weather, and yield variables",
            },
            {
                "Dataset": "External weather data",
                "Source": "NASA POWER API",
                "Local table": "weather_daily",
                "Role": "Daily environmental time-series ingestion example for Sevilla, Spain",
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
        Two random forest regression models are compared:

        - **Full model:** uses all numeric predictors, including fruit traits.
        - **Pre-harvest model:** excludes fruit set, fruit mass, and seeds to reduce outcome leakage.

        The goal is not to claim causality, but to demonstrate how model outputs can be
        integrated into a reproducible R&D data product.
        """
    )

    st.subheader("Model metrics")

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.subheader(f"Feature importance: {model_labels.get(selected_model, selected_model)}")

    top_importance = selected_importance.head(10).copy()
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

    st.subheader("How to interpret these models")

    st.markdown(
        """
        The full model can be useful for explaining the simulated yield outcome, but it may
        be too optimistic because fruit traits are very close to final yield.

        The pre-harvest model is more conservative because it excludes fruit traits. This is
        closer to an early decision-support setting, where the goal would be to predict yield
        before final fruit measurements are available.
        """
    )


with tab_dictionary:
    st.header("Data dictionary")

    dictionary_df = pd.DataFrame(DATA_DICTIONARY)

    st.dataframe(dictionary_df, use_container_width=True, hide_index=True)

    st.subheader("Why this matters")

    st.markdown(
        """
        A data dictionary makes the dataset easier to use by breeders, agronomists,
        data scientists, and IT teams.

        It also supports:

        - data governance;
        - metadata tracking;
        - AI/data readiness;
        - clearer communication with non-technical stakeholders.
        """
    )


with tab_limitations:
    st.header("Known limitations")

    limitations_df = pd.DataFrame(
        [
            {
                "Limitation": "Yield data are simulated",
                "Why it matters": "The dataset is useful for prototyping, but it is not a real breeding trial dataset.",
                "How to improve": "Replace or extend it with real genotype, plot, block, location, and harvest records.",
            },
            {
                "Limitation": "External Sevilla weather is not joined to yield records",
                "Why it matters": "The yield records do not contain dates or coordinates, so a direct join would be misleading.",
                "How to improve": "Add trial dates and geolocation fields, then aggregate weather by relevant biological windows.",
            },
            {
                "Limitation": "Fruit traits may cause outcome leakage",
                "Why it matters": "Fruit set, fruit mass, and seeds are close to the final yield outcome.",
                "How to improve": "Use separate model versions for explanatory analysis and early prediction.",
            },
            {
                "Limitation": "No experimental design metadata",
                "Why it matters": "Real breeding trials require genotype, block, plot, year, environment, and management metadata.",
                "How to improve": "Add relational tables for trial design and use mixed-effect models.",
            },
            {
                "Limitation": "Feature importance is not causality",
                "Why it matters": "Random forest importance shows model usage, not biological cause.",
                "How to improve": "Combine exploratory models with experimental design and domain expertise.",
            },
        ]
    )

    st.dataframe(limitations_df, use_container_width=True, hide_index=True)

    st.info(
        """
        These limitations are intentionally documented. In an R&D setting, understanding what
        the data cannot answer is as important as producing plots or model metrics.
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
