# %% imports
from pathlib import Path
# import numpy as np
import pandas as pd

# %% load data

RAW = Path("../data/raw")
PROCESSED = Path("../data/processed")

results = pd.read_csv(RAW / "fsae_ev_results.csv")
efficiency = pd.read_csv(RAW / "fsae_ev_efficiency.csv")
print("results shape:", results.shape)
print("efficiency shape:", efficiency.shape)
results.head()

# %% see whats mising
print("=== missing values per column ===")
print(results.isnull().sum())
print("\n=== unique years ===")
print(results.year.value_counts())

# %% who ran endurance
# teams that DNF'd or never started endurance get a laps score of 2 or less
# (per the scoring table, 2 laps = 3 points = basically nothing)
# we keep them in the full dataset but flag them

EE_COLS = ["endurance", "efficiency"]
DYNAMIC_COLS = ["accel", "skidpad", "autocross", "endurance", "efficiency"]
STATIC_COLS = ["cost", "presentation", "design"]
ALL_EVENT_COLS = STATIC_COLS + DYNAMIC_COLS

# flags
results["completed_endurance"] = results["endurance"] > 25
results["ran_dynamics"] = results[["accel", "skidpad", "autocross"]].notna().any(axis=1)
results["scored_efficiency"] = results["efficiency"].notna()
print("teams that completed endurance:")
print(results.groupby("year")["completed_endurance"].value_counts())

# %% create the filtered datasets
df_full = results.copy()
df_full["penalty"] = df_full["penalty"].fillna(0)
df_dynamic = df_full[
    df_full["ran_dynamics"]
].copy()  # this is teams that at least ran at dynamic events
df_endurance = df_full[df_full["completed_endurance"]].copy() # this is teams that actually did endurance

# %% derived columns
ee_sum = df_endurance[EE_COLS].sum(axis=1, min_count=1)
df_endurance["total_ex_ee"] = df_endurance["total"] - ee_sum
 
# E+E combined score
df_endurance["ee_combined"] = df_endurance[EE_COLS].sum(axis=1, min_count=1)
 
# dynamic score only (accel + skidpad + autocross + endurance + efficiency)
df_endurance["dynamic_total"] = df_endurance[DYNAMIC_COLS].sum(axis=1, min_count=1)
 
# static score only
df_endurance["static_total"] = df_endurance[STATIC_COLS].sum(axis=1, min_count=1)
 
print("endurance dataset shape:", df_endurance.shape)
print("\nsample derived cols:")
print(df_endurance[["year", "team", "total", "ee_combined", "total_ex_ee"]].head(10))

# %% merge efficiency details into endurance dataset
df_endurance_eff = df_endurance.merge(
    efficiency[["year", "car_num", "energy_used_kwh", "avg_adj_laptime_s", "efficiency_factor"]],
    on=["year", "car_num"],
    how="left"
)
 
print("merged shape:", df_endurance_eff.shape)
print("teams with energy data:", df_endurance_eff["energy_used_kwh"].notna().sum())


# %% sanity check - who's in the endurance dataset per year
for yr in sorted(df_endurance.year.unique()):
    subset = df_endurance[df_endurance.year == yr]
    print(f"\n{yr} - {len(subset)} teams completed endurance:")
    print(subset[["team", "endurance", "efficiency", "total"]].to_string(index=False))
 
# %% save processed files
df_full.to_csv(PROCESSED / "fsae_full.csv", index=False)
df_dynamic.to_csv(PROCESSED / "fsae_dynamic.csv", index=False)
df_endurance.to_csv(PROCESSED / "fsae_endurance.csv", index=False)
df_endurance_eff.to_csv(PROCESSED / "fsae_endurance_eff.csv", index=False)
 
print("\nsaved to data/processed/:")
for f in PROCESSED.iterdir():
    print(f" - {f.name}")
