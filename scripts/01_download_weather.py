import requests
import pandas as pd
from pathlib import Path

OUTDIR = Path("data/external")
OUTDIR.mkdir(parents=True, exist_ok=True)

latitude = 37.39
longitude = -5.99

params = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN",
]

url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?parameters={','.join(params)}"
    "&community=AG"
    f"&longitude={longitude}"
    f"&latitude={latitude}"
    "&start=20230101"
    "&end=20231231"
    "&format=JSON"
)

response = requests.get(url, timeout=60)
response.raise_for_status()
data = response.json()

records = data["properties"]["parameter"]

df = pd.DataFrame(records)
df.index.name = "date"
df = df.reset_index()
df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

outpath = OUTDIR / "weather_sevilla_nasa_power.csv"
df.to_csv(outpath, index=False)

print(f"Saved: {outpath}")
print(df.head())
