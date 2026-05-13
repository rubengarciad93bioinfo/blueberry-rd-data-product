from pathlib import Path
import pandas as pd
import duckdb

raw_path = Path("data/raw/WildBlueberryPollinationSimulationData.csv")
weather_path = Path("data/external/weather_sevilla_nasa_power.csv")
db_path = Path("db/blueberry_trials.duckdb")

db_path.parent.mkdir(parents=True, exist_ok=True)

yield_df = pd.read_csv(raw_path)
weather_df = pd.read_csv(weather_path)

# Standardize column names
yield_df.columns = (
    yield_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

weather_df.columns = (
    weather_df.columns
    .str.strip()
    .str.lower()
)

# Add simple IDs for portfolio-style relational structure
yield_df = yield_df.reset_index().rename(columns={"index": "simulation_id"})
yield_df["simulation_id"] = yield_df["simulation_id"] + 1

con = duckdb.connect(str(db_path))

con.execute("DROP TABLE IF EXISTS blueberry_yield")
con.execute("DROP TABLE IF EXISTS weather_daily")

con.register("yield_df", yield_df)
con.register("weather_df", weather_df)

con.execute("CREATE TABLE blueberry_yield AS SELECT * FROM yield_df")
con.execute("CREATE TABLE weather_daily AS SELECT * FROM weather_df")

print("Database created:", db_path)

print(con.execute("SHOW TABLES").fetchdf())
print(con.execute("SELECT COUNT(*) AS n FROM blueberry_yield").fetchdf())
print(con.execute("SELECT COUNT(*) AS n FROM weather_daily").fetchdf())

con.close()
