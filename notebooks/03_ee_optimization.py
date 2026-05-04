# %%
# 03_ee_optimization.py
# goal: find the kWh sweet-spot that maximizes endurance + efficiency score
#
# strat:
# 1. fit a curve for kWh → endurance (curve A)
# 2. fit a curve for kWh → efficiency  (curve B)
# 3. add them: kWh → endurance + efficiency  (curve C)
# 4. mark the peak of curve C - that's the optimal kWh budget
#
# only teams that scored both events are used (teams that DNF'd efficiency
# are excluded because they're structural outliers, not an energy-tradeoff
# story)

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# paths
PROCESSED = Path("../data/processed")
FIGURES   = Path("../figures")
FIGURES.mkdir(exist_ok=True)

# data
df = pd.read_csv(PROCESSED / "fsae_endurance_eff.csv")

# keep only teams with all three values we need
mask = df["energy_used_kwh"].notna() & df["endurance"].notna() & df["efficiency"].notna()
df_scored = df[mask].copy().reset_index(drop=True)

# derived column
df_scored["ee_combined"] = df_scored["endurance"] + df_scored["efficiency"]

print(f"Teams with complete energy + score data: {len(df_scored)}")
print(df_scored[["year", "team", "energy_used_kwh", "endurance", "efficiency", "ee_combined"]]
      .sort_values("energy_used_kwh").to_string(index=False))


# helper: lowess smoother
def lowess(x, y, frac=0.55, n_pts=200):
    """Return (x_smooth, y_smooth) using statsmodels-free iterative LOWESS."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_grid = np.linspace(x.min(), x.max(), n_pts)
    y_grid = np.empty(n_pts)
    h = frac * (x.max() - x.min())          # bandwidth

    for i, xi in enumerate(x_grid):
        dist = np.abs(x - xi)
        w = np.maximum(0, 1 - (dist / h) ** 3) ** 3  # tricube weights
        if w.sum() == 0:
            y_grid[i] = np.nan
            continue
        # weighted least squares: fit a line locally
        W = np.diag(w)
        X = np.column_stack([np.ones_like(x), x])
        try:
            beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            y_grid[i] = beta[0] + beta[1] * xi
        except np.linalg.LinAlgError:
            y_grid[i] = np.nan

    return x_grid, y_grid


# colors & style - thank you claude
colors = {2024: "#4C72B0", 2025: "#DD8452"}
C_END  = "#2ECC71"   # endurance curve
C_EFF  = "#E74C3C"   # efficiency curve
C_COMB = "#8E44AD"   # combined curve

# ─────────────────────────────────────────────────────────────────────────────
# ── data vectors & smooth curves ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# %% prepare curves

kwh   = df_scored["energy_used_kwh"].values
end   = df_scored["endurance"].values
eff   = df_scored["efficiency"].values
comb  = df_scored["ee_combined"].values
years = df_scored["year"].values

# smooth curves
x_s, end_s  = lowess(kwh, end,  frac=0.55)
x_s, eff_s  = lowess(kwh, eff,  frac=0.55)
x_s, comb_s = lowess(kwh, comb, frac=0.55)

# find optimal kWh (peak of combined smooth curve)
opt_idx = np.nanargmax(comb_s)
opt_kwh = x_s[opt_idx]
opt_val = comb_s[opt_idx]


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2  –  single hero chart: all three curves overlaid + shaded zones
# ─────────────────────────────────────────────────────────────────────────────
# %% hero chart
fig, ax = plt.subplots(figsize=(10, 6))

# scatter (raw data) coloured by year, slightly transparent
for yr in [2024, 2025]:
    m = years == yr
    ax.scatter(kwh[m], comb[m], color=colors[yr], s=65, zorder=3,
               label=f"{yr} data", alpha=0.75, edgecolors="white", linewidths=0.5)

# individual curves (lighter)
ax.plot(x_s, end_s,  color=C_END,  lw=1.8, ls="-.",  alpha=0.7, label="Endurance fit")
ax.plot(x_s, eff_s,  color=C_EFF,  lw=1.8, ls="--",  alpha=0.7, label="Efficiency fit")

# combined curve (bold)
ax.plot(x_s, comb_s, color=C_COMB, lw=3,             label="Combined fit (Endurance + Efficiency)")

# zone shading
x_min, x_max = x_s[0], x_s[-1]
ax.axvspan(x_min,    opt_kwh - 0.5, alpha=0.06, color=C_EFF, label="Under-energy zone")
ax.axvspan(opt_kwh + 0.5, x_max,    alpha=0.06, color=C_END, label="Over-energy zone")
ax.axvspan(opt_kwh - 0.5, opt_kwh + 0.5, alpha=0.12, color=C_COMB, label="Sweet spot ±0.5 kWh")

# optimum marker
ax.scatter([opt_kwh], [opt_val], color=C_COMB, s=220, zorder=6,
           marker="*", edgecolors="white", linewidths=1)
ax.axvline(opt_kwh, color=C_COMB, lw=1.6, ls="--", alpha=0.8)
ax.annotate(
    f"Optimal  ≈ {opt_kwh:.2f} kWh\nPeak combined ≈ {opt_val:.0f} pts",
    xy=(opt_kwh, opt_val),
    xytext=(opt_kwh + 0.4, opt_val - 30),
    fontsize=10, color=C_COMB, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_COMB, alpha=0.85),
    arrowprops=dict(arrowstyle="->", color=C_COMB, lw=1.2),
)

# zone labels
y_bot = ax.get_ylim()[0]
ax.text(x_min + 0.05, y_bot + 5, "← Too low kWh\n  (great efficiency,\n   weak endurance)",
        fontsize=8, color=C_EFF, alpha=0.8, va="bottom")
ax.text(x_max - 0.05, y_bot + 5, "Too high kWh →\n  (great endurance,\n   weak efficiency)",
        fontsize=8, color=C_END, alpha=0.8, va="bottom", ha="right")

ax.set_xlabel("Energy Used (kWh)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title(
    "Energy Optimization: kWh vs. Endurance & Efficiency Scores\n",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=8.5, loc="upper left", ncol=2)
ax.grid(True, alpha=0.22)
plt.tight_layout()
plt.savefig(FIGURES / "03_ee_optimization_hero.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3  –  sensitivity table: what combined score do you get at each kWh?
# ─────────────────────────────────────────────────────────────────────────────
# %% sensitivity printout
print("\n=== Sensitivity: kWh vs predicted combined E+E score ===")
sample_kwh = np.arange(
    np.floor(kwh.min() * 10) / 10,
    np.ceil(kwh.max() * 10)  / 10 + 0.1,
    0.25
)

# interpolate smooth curves at those points
from numpy import interp
end_interp  = interp(sample_kwh, x_s, end_s)
eff_interp  = interp(sample_kwh, x_s, eff_s)
comb_interp = interp(sample_kwh, x_s, comb_s)

sens_df = pd.DataFrame({
    "kWh":              np.round(sample_kwh, 2),
    "pred_endurance":   np.round(end_interp, 1),
    "pred_efficiency":  np.round(eff_interp, 1),
    "pred_combined":    np.round(comb_interp, 1),
})
sens_df["delta_from_peak"] = np.round(comb_interp - opt_val, 1)
print(sens_df.to_string(index=False))
print(f"\n★ Optimal kWh: {opt_kwh:.3f}")
print(f"★ Peak combined score (smooth): {opt_val:.1f}")
print(f"★ Data range: {kwh.min():.3f} – {kwh.max():.3f} kWh")
