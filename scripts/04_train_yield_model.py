from pathlib import Path
import duckdb
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


DB_PATH = Path("db/blueberry_trials.duckdb")
OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

df = con.execute("SELECT * FROM blueberry_yield").fetchdf()

target = "yield"

if target not in df.columns:
    raise ValueError(f"Expected target column '{target}' not found. Available columns: {df.columns.tolist()}")

# Use numeric columns only
numeric_df = df.select_dtypes(include="number").copy()

# Remove ID and target from features
feature_cols = [
    col for col in numeric_df.columns
    if col not in ["simulation_id", target]
]

X = numeric_df[feature_cols]
y = numeric_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=8
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

metrics = pd.DataFrame([
    {
        "model": "RandomForestRegressor",
        "n_records": len(df),
        "n_features": len(feature_cols),
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions)
    }
])

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

metrics.to_csv(OUTDIR / "model_metrics.csv", index=False)
feature_importance.to_csv(OUTDIR / "model_feature_importance.csv", index=False)

con.execute("DROP TABLE IF EXISTS model_metrics")
con.execute("DROP TABLE IF EXISTS model_feature_importance")

con.register("metrics_df", metrics)
con.register("feature_importance_df", feature_importance)

con.execute("CREATE TABLE model_metrics AS SELECT * FROM metrics_df")
con.execute("CREATE TABLE model_feature_importance AS SELECT * FROM feature_importance_df")

print("Model metrics:")
print(metrics)

print("\nTop feature importances:")
print(feature_importance.head(10))

print("\nSaved:")
print(OUTDIR / "model_metrics.csv")
print(OUTDIR / "model_feature_importance.csv")

con.close()
