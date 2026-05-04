# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

df = pd.read_csv("../data/processed/fsae_endurance_eff.csv")

df["is_sjsu"] = df["team"].str.contains("San Jose", case=False, na=False)
colors = {2024: "#4C72B0", 2025: "#DD8452"}

# %% endurance vs efficiency score

fig, ax = plt.subplots(figsize=(8, 6))

for yr, grp in df.groupby("year"):
    scored = grp[grp["efficiency"].notna() & grp["endurance"].notna()]
    ax.scatter(
        scored["endurance"],
        scored["efficiency"],
        color=colors[yr],
        s=80,
        label=str(yr),
        zorder=3,
    )
    # label SJSU if present (NaN efficiency so won't show here — note below)

# annotate teams that DNF'd efficiency (SJSU) as text note
sjsu = df[df["is_sjsu"]]
for _, row in sjsu.iterrows():
    if pd.notna(row.get("endurance")):
        ax.annotate(
            f"SJSU {int(row['year'])}\n(275 end, 0 eff)",
            xy=(row["endurance"], 0),
            xytext=(row["endurance"] - 40, 15),
            fontsize=8,
            color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

# callout: Connecticut 2025 (best efficiency)
conn = df[
    (df["team"].str.contains("Connecticut", case=False, na=False))
    & (df["year"] == 2025)
]
for _, row in conn.iterrows():
    if pd.notna(row.get("efficiency")) and pd.notna(row.get("endurance")):
        ax.annotate(
            "Connecticut '25\n(100 eff)",
            xy=(row["endurance"], row["efficiency"]),
            xytext=(row["endurance"] - 60, row["efficiency"] + 8),
            fontsize=8,
            color=colors[2025],
            arrowprops=dict(arrowstyle="->", color=colors[2025], lw=0.8),
        )

ax.set_xlabel("Endurance Score")
ax.set_ylabel("Efficiency Score")
ax.set_title("Endurance score vs Efficiency score\n(teams that completed endurance)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/02_endurance_vs_efficiency.png", dpi=150)
plt.show()

# %% kWh vs adjusted lap time
fig, ax = plt.subplots(figsize=(8, 6))

for yr, grp in df.groupby("year"):
    valid = grp[grp["energy_used_kwh"].notna() & grp["avg_adj_laptime_s"].notna()]
    ax.scatter(
        valid["energy_used_kwh"],
        valid["avg_adj_laptime_s"],
        color=colors[yr],
        s=80,
        label=str(yr),
        zorder=3,
    )

# label SJSU
for _, row in df[df["is_sjsu"]].iterrows():
    if pd.notna(row.get("energy_used_kwh")) and pd.notna(row.get("avg_adj_laptime_s")):
        ax.annotate(
            f"SJSU {int(row['year'])}",
            xy=(row["energy_used_kwh"], row["avg_adj_laptime_s"]),
            xytext=(row["energy_used_kwh"] + 0.1, row["avg_adj_laptime_s"] + 1.5),
            fontsize=8,
            color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
        )

ax.set_xlabel("Energy Used (kWh)")
ax.set_ylabel("Avg Adjusted Lap Time (s)")
ax.set_title("Energy used vs Lap time\n(lower lap time = faster)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/02_kwh_vs_laptime.png", dpi=150)
plt.show()

# %% kWh vs endurance

fig, ax = plt.subplots(figsize=(8, 6))
 
for yr, grp in df.groupby("year"):
    valid = grp[grp["energy_used_kwh"].notna() & grp["endurance"].notna()]
    ax.scatter(
        valid["energy_used_kwh"], valid["endurance"],
        color=colors[yr], s=80, label=str(yr), zorder=3
    )
 
for _, row in df[df["is_sjsu"]].iterrows():
    if pd.notna(row.get("energy_used_kwh")) and pd.notna(row.get("endurance")):
        ax.annotate(
            f"SJSU {int(row['year'])}",
            xy=(row["energy_used_kwh"], row["endurance"]),
            xytext=(row["energy_used_kwh"] + 0.1, row["endurance"] - 15),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
        )
 
ax.set_xlabel("Energy Used (kWh)")
ax.set_ylabel("Endurance Score")
ax.set_title("Energy used vs Endurance score")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/02_kwh_vs_endurance.png", dpi=150)
plt.show()
# %% kWh vs efficiency

fig, ax = plt.subplots(figsize=(8, 6))

for yr, grp in df.groupby("year"):
   # teams that scored efficiency
   scored = grp[grp["energy_used_kwh"].notna() & grp["efficiency"].notna()]
   ax.scatter(
       scored["energy_used_kwh"], scored["efficiency"],
       color=colors[yr], s=80, label=str(yr), zorder=3
   )
   # teams that completed endurance but got zero efficiency (cap exceeded)
   zeroed = grp[grp["energy_used_kwh"].notna() & grp["efficiency"].isna() & grp["completed_endurance"]]
   ax.scatter(
       zeroed["energy_used_kwh"], [0] * len(zeroed),
       color=colors[yr], s=80, marker="x", zorder=3
   )

# annotate SJSU zeros
for _, row in df[df["is_sjsu"]].iterrows():
   if pd.notna(row.get("energy_used_kwh")):
       ax.annotate(
           f"SJSU {int(row['year'])}\n(cap exceeded)",
           xy=(row["energy_used_kwh"], 0),
           xytext=(row["energy_used_kwh"] - 1.5, 12),
           fontsize=8, color="gray",
           arrowprops=dict(arrowstyle="->", color="gray", lw=0.8)
       )

ax.set_xlabel("Energy Used (kWh)")
ax.set_ylabel("Efficiency Score")
ax.set_title("Energy used vs Efficiency score\n(x = cap exceeded, scored zero)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/02_kwh_vs_efficiency.png", dpi=150)
plt.show()

# %% energy vs efficiency factor
fig, ax = plt.subplots(figsize=(8, 6))
 
for yr, grp in df.groupby("year"):
    valid = grp[grp["energy_used_kwh"].notna() & grp["efficiency_factor"].notna()]
    ax.scatter(
        valid["energy_used_kwh"], valid["efficiency_factor"],
        color=colors[yr], s=80, label=str(yr), zorder=3
    )
 
ax.set_xlabel("Energy Used (kWh)")
ax.set_ylabel("Efficiency Factor")
ax.set_title("Energy used vs Efficiency factor\n(what the scoring formula actually uses)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../figures/02_kwh_vs_efficiency_factor.png", dpi=150)
plt.show()
