# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED = Path("../data/processed")
FIGURES = Path("../figures")
FIGURES.mkdir(exist_ok=True)

df = pd.read_csv(PROCESSED / "fsae_endurance.csv")

# %% correlation matrix
# only teams that completed endurance, both years combined
EVENT_COLS = ["cost", "presentation", "design", "accel", "skidpad", "autocross", "endurance", "efficiency"]

corr = df[EVENT_COLS].corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(EVENT_COLS)))
ax.set_yticks(range(len(EVENT_COLS)))
ax.set_xticklabels(EVENT_COLS, rotation=45, ha="right")
ax.set_yticklabels(EVENT_COLS)

# annotate cells
for i in range(len(EVENT_COLS)):
    for j in range(len(EVENT_COLS)):
        if not np.isnan(corr.iloc[i, j]):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

ax.set_title("Event Score Correlations (teams that completed endurance, 2024+2025)")
plt.tight_layout()
plt.savefig(FIGURES / "01_correlation_heatmap.png", dpi=150)
plt.show()

# %% print the endurance row specifically
print("=== correlation with endurance score ===")
print(corr["endurance"].sort_values(ascending=False).to_string())

print("\n=== correlation with efficiency score ===")
print(corr["efficiency"].sort_values(ascending=False).to_string())

# %% ee_combined vs total_ex_ee scatter
# do E+E strong teams sacrifice everything else?
fig, ax = plt.subplots(figsize=(7, 5))

for yr, grp in df.groupby("year"):
    ax.scatter(grp["ee_combined"], grp["total_ex_ee"], label=str(yr), s=50)
    for _, row in grp.iterrows():
        ax.annotate(row["team"].split(" - ")[0].split("Univ")[0].strip(),
                    (row["ee_combined"], row["total_ex_ee"]),
                    fontsize=6, alpha=0.7, xytext=(4, 2), textcoords="offset points")

ax.set_xlabel("E+E Combined Score (endurance + efficiency)")
ax.set_ylabel("Total Score excluding E+E")
ax.set_title("Does chasing E+E hurt everything else?")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "01_ee_vs_rest.png", dpi=150)
plt.show()

# %% endurance score vs total - how much does finishing matter?
fig, ax = plt.subplots(figsize=(7, 5))

for yr, grp in df.groupby("year"):
    ax.scatter(grp["endurance"], grp["total"], label=str(yr), s=50)

# fit a line across both years
mask = df["endurance"].notna() & df["total"].notna()
m, b = np.polyfit(df.loc[mask, "endurance"], df.loc[mask, "total"], 1)
x = np.linspace(df["endurance"].min(), df["endurance"].max(), 100)
ax.plot(x, m*x + b, "k--", alpha=0.4, label=f"fit (slope={m:.2f})")

ax.set_xlabel("Endurance Score")
ax.set_ylabel("Total Score")
ax.set_title("Endurance score vs total score")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "01_endurance_vs_total.png", dpi=150)
plt.show()

print(f"\nfor every 1 point gain in endurance score, total score goes up ~{m:.2f} points")
print(f"r = {df['endurance'].corr(df['total']):.3f}")
