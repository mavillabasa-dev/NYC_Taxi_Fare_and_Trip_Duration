"""Re-derive every dataset number quoted in the pre-demo deck.

Run this to check the slides against the data. Each printed line maps to a
figure on a slide, so if the dataset is ever re-downloaded the deck can be
re-verified in one command instead of by memory.

Usage (from the repo root, with the project venv active):

    python ppt/tools/measure.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "dataset" / "yellow_tripdata_2022-05.parquet"
LOOKUP = ROOT / "dataset" / "taxi_zone_lookup.csv"

# The cleaning window used for the "after cleaning" correlation on slide 15.
# Deliberately loose -- it is a sanity filter, not the T-104 rule set.
FARE_LO, FARE_HI = 2.50, 250.0
DIST_LO, DIST_HI = 0.1, 100.0
DUR_LO, DUR_HI = 1.0, 360.0


def rule(title):
    print(f"\n{'-' * 66}\n{title}\n{'-' * 66}")


df = pd.read_parquet(SRC)

rule("SLIDE 9  -- dataset shape")
# Measured before deriving anything, so these match the raw file as shipped.
print(f"rows                     {len(df):,}")
print(f"columns                  {df.shape[1]}")
print(f"memory                   {df.memory_usage(deep=True).sum() / 1024**2:,.0f} MB")
print(f"exact duplicate rows     {int(df.duplicated().sum()):,}")

df["dur"] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60

pick = df.tpep_pickup_datetime
in_may = (pick >= "2022-05-01") & (pick < "2022-06-01")
print(f"rows outside May 2022    {int((~in_may).sum())}")
print(f"nulls (one shared block) {int(df.passenger_count.isna().sum()):,} "
      f"= {100 * df.passenger_count.isna().mean():.1f}%")

d = df[in_may]
rule("SLIDES 2 & 3  -- scale")
print(f"rides per day (avg)      {len(d) / 31:,.0f}")
by_hour = d.groupby(pick[in_may].dt.hour).size() / 31
print(f"peak hour                {int(by_hour.idxmax()):02d}:00 at {by_hour.max():,.0f}/h")
print(f"quietest hour            {int(by_hour.idxmin()):02d}:00 at {by_hour.min():,.0f}/h")
print(f"peak / trough ratio      {by_hour.max() / by_hour.min():.0f}x")

rule("SLIDE 9  -- target profile")
print(f"fare    median / p99 / max   ${d.fare_amount.median():,.2f} / "
      f"${d.fare_amount.quantile(0.99):,.2f} / ${d.fare_amount.max():,.2f}")
print(f"minutes median               {d.dur.median():.1f}")

rule("SLIDE 15  -- why the data work matters")
# log1p is only defined on the positive side, so the transformed skew is taken
# over positive values. The deck quotes these to one decimal, which is where
# this and 01_EDA.ipynb's slightly different filter agree.
pos_fare, pos_dur = d.fare_amount[d.fare_amount > 0], d.dur[d.dur > 0]
print(f"fare skew  raw / log1p       {d.fare_amount.skew():.1f} / "
      f"{np.log1p(pos_fare).skew():.1f}")
print(f"duration skew raw / log1p    {d.dur.skew():.1f} / "
      f"{np.log1p(pos_dur).skew():.1f}")

mph = (d.trip_distance / (d.dur / 60)).replace([np.inf, -np.inf], np.nan)
print(f"fastest implied speed        {mph.max():,.0f} mph")

m = (d.fare_amount.between(FARE_LO, FARE_HI)
     & d.trip_distance.between(DIST_LO, DIST_HI)
     & d.dur.between(DUR_LO, DUR_HI))
c = d[m]
print(f"corr(trip_distance, fare)    {d.trip_distance.corr(d.fare_amount):.2f} raw  ->  "
      f"{c.trip_distance.corr(c.fare_amount):.2f} after the sanity filter "
      f"({100 * len(c) / len(d):.1f}% of rows kept)")
print(f"corr(passenger_count, fare)  {c.passenger_count.corr(c.fare_amount):.2f}")

rule("SLIDE 15  -- rate codes are separate populations")
names = {1: "Standard", 2: "JFK", 3: "Newark", 4: "Nassau/Westchester",
         5: "Negotiated", 6: "Group ride", 99: "Undocumented (99)"}
rc = d.groupby("RatecodeID").agg(trips=("fare_amount", "size"),
                                 mean_fare=("fare_amount", "mean"),
                                 mean_min=("dur", "mean"))
for k, v in rc[rc.trips > 500].iterrows():
    print(f"  {names.get(int(k), int(k)):<20} {int(v.trips):>9,} trips   "
          f"${v.mean_fare:>6.2f}   {v.mean_min:>5.1f} min")

rule("SLIDE 16  -- temporal split (train = May 1-24, test = May 25-31)")
train = int((pick[in_may] < "2022-05-25").sum())
print(f"train rows               {train:,}  ({100 * train / len(d):.1f}%)")
print(f"test rows                {len(d) - train:,}  ({100 * (len(d) - train) / len(d):.1f}%)")

rule("SLIDE 10  -- zones")
z = pd.read_csv(LOOKUP)
print(f"zones in lookup          {len(z)}   boroughs: {z.Borough.nunique()}")
print(f"distinct pickup zones    {d.PULocationID.nunique()}")
top = d.PULocationID.value_counts().head(4)
for zid, n in top.items():
    row = z[z.LocationID == zid].iloc[0]
    print(f"  {row.Zone:<24} ({zid})  {n:,} pickups   [{row.Borough}]")
