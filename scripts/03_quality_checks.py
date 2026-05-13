import duckdb
import pandas as pd
from pathlib import Path

db_path = Path("db/blueberry_trials.duckdb")
report_path = Path("data/processed/data_quality_report.csv")
report_path.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(db_path))

checks = []

def add_check(name, query, expected):
    result = con.execute(query).fetchone()[0]
    status = "PASS" if result == expected else "FAIL"
    checks.append({
        "check": name,
        "result": result,
        "expected": expected,
        "status": status
    })

add_check(
    "No missing yield values",
    "SELECT COUNT(*) FROM blueberry_yield WHERE yield IS NULL",
    0
)

add_check(
    "No negative yield values",
    "SELECT COUNT(*) FROM blueberry_yield WHERE yield < 0",
    0
)

add_check(
    "No missing weather dates",
    "SELECT COUNT(*) FROM weather_daily WHERE date IS NULL",
    0
)

add_check(
    "No impossible daily max temperature",
    "SELECT COUNT(*) FROM weather_daily WHERE t2m_max > 60 OR t2m_max < -40",
    0
)

add_check(
    "No negative precipitation",
    "SELECT COUNT(*) FROM weather_daily WHERE prectotcorr < 0",
    0
)

report = pd.DataFrame(checks)
report.to_csv(report_path, index=False)

print(report)
print(f"Saved: {report_path}")

con.close()
