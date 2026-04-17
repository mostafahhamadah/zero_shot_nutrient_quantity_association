import pandas as pd

COLUMNS = ["image_id","nutrient","quantity","unit","context","nrv_percent","serving_size"]
df = pd.read_csv("data/annotations/gold_annotations.csv",
                 names=COLUMNS, skiprows=1, usecols=range(7),
                 encoding="utf-8", engine="python", on_bad_lines="warn")

valid_ctx = {"per_100g","per_serving","per_daily_dose","per_100ml"}
bad = df[~df["context"].isin(valid_ctx)]
print(f"Bad rows: {len(bad)}")
print(bad[["image_id","nutrient","quantity","unit","context"]].to_string())
