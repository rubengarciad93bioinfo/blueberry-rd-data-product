from pathlib import Path

import duckdb
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


DB_PATH = Path("db/blueberry_trials.duckdb")
OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET = "yield"

# These variables are biologically close to the final crop outcome.
# They are useful for explaining yield, but may be inappropriate for early prediction.
OUTCOME_ADJACENT_FEATURES = ["fruitset", "fruitmass", "seeds"]
ID_OR_INDEX_FEATURES = ["simulation_id", "row#", "row", "index"]

def train_yield_model(df: pd.DataFrame, model_name: str, model_label: str, excluded_features=None):
    if excluded_features is None:
        excluded_features = []

    numeric_df = df.select_dtypes(include="number").copy()

    feature_cols = [
    	col for col in numeric_df.columns
    	if col not in ID_OR_INDEX_FEATURES
    	and col != TARGET
    	and col not in excluded_features
	]

    if not feature_cols:
        raise ValueError(f"No features available for model {model_name}")

    X = numeric_df[feature_cols]
    y = numeric_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=8,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "model": model_name,
        "model_label": model_label,
        "n_records": len(df),
        "n_features": len(feature_cols),
        "excluded_features": ", ".join(excluded_features) if excluded_features else "None",
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }

    feature_importance = pd.DataFrame(
        {
            "model": model_name,
            "model_label": model_label,
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return metrics, feature_importance


def main():
    con = duckdb.connect(str(DB_PATH))

    df = con.execute("SELECT * FROM blueberry_yield").fetchdf()

    if TARGET not in df.columns:
        raise ValueError(
            f"Expected target column '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    all_metrics = []
    all_importance = []

    full_metrics, full_importance = train_yield_model(
        df=df,
        model_name="full_model",
        model_label="Full model: all numeric predictors",
        excluded_features=[],
    )

    early_metrics, early_importance = train_yield_model(
        df=df,
        model_name="pre_harvest_model",
        model_label="Pre-harvest model: excluding fruit traits",
        excluded_features=[f for f in OUTCOME_ADJACENT_FEATURES if f in df.columns],
    )

    all_metrics.extend([full_metrics, early_metrics])
    all_importance.extend([full_importance, early_importance])

    metrics_df = pd.DataFrame(all_metrics)
    importance_df = pd.concat(all_importance, ignore_index=True)

    metrics_path = OUTDIR / "model_metrics.csv"
    importance_path = OUTDIR / "model_feature_importance.csv"

    metrics_df.to_csv(metrics_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    con.execute("DROP TABLE IF EXISTS model_metrics")
    con.execute("DROP TABLE IF EXISTS model_feature_importance")

    con.register("metrics_df", metrics_df)
    con.register("importance_df", importance_df)

    con.execute("CREATE TABLE model_metrics AS SELECT * FROM metrics_df")
    con.execute("CREATE TABLE model_feature_importance AS SELECT * FROM importance_df")

    print("Model metrics:")
    print(metrics_df)

    print("\nTop feature importances by model:")
    for model_name in importance_df["model"].unique():
        print(f"\n{model_name}")
        print(
            importance_df.loc[importance_df["model"] == model_name]
            .head(10)
            [["feature", "importance"]]
        )

    print("\nSaved:")
    print(metrics_path)
    print(importance_path)

    con.close()


if __name__ == "__main__":
    main()
