# %%
# 03_ee_optimization.py
# goal: find the kWh level that puts teams in the competitive endurance range
#
# strat:
# 1. fit endurance ~ log(kWh)          – diminishing returns, always increasing
# 2. fit efficiency ~ (E_max - kWh)    – linear decay constrained to 0 at E_max
#    (per FSAE rules, efficiency score reaches 0 at the maximum energy threshold)
# 3. add them → combined curve; green zone marks "competitive" energy (≥ 6 kWh)
#
# only teams that completed both events are used (DNF'd efficiency teams are
# structural outliers, not part of the energy-tradeoff story)

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
PROCESSED = Path("../data/processed")
FIGURES   = Path("../figures")
FIGURES.mkdir(exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED / "fsae_endurance_eff.csv")
mask = (
    df["energy_used_kwh"].notna()
    & df["endurance"].notna()
    & df["efficiency"].notna()
)
df_scored = df[mask].copy().reset_index(drop=True)
df_scored["ee_combined"] = df_scored["endurance"] + df_scored["efficiency"]

print(f"Teams with complete energy + score data: {len(df_scored)}")
print(
    df_scored[["year", "team", "energy_used_kwh", "endurance", "efficiency", "ee_combined"]]
    .sort_values("energy_used_kwh")
    .to_string(index=False)
)

# ── helper: monotone non-decreasing (PAVA / isotonic regression) ──────────────
def isotonic_increasing(y):
    """Return a non-decreasing array that is the best L2 fit to y (PAVA)."""
    y = np.array(y, dtype=float)
    blocks = [[yi, 1] for yi in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] > blocks[i + 1][0] / blocks[i + 1][1]:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1
    result = np.empty(len(y))
    idx = 0
    for val, cnt in blocks:
        result[idx : idx + cnt] = val / cnt
        idx += cnt
    return result


# ── colors ────────────────────────────────────────────────────────────────────
colors = {2024: "black", 2025: "#777"}
C_END  = "red"   # teal  – endurance curve
C_EFF  = "blue"   # amber – efficiency curve
C_COMB = "#8E44AD"   # purple – combined curve
C_COMP = "#108045"   # green  – competitive zone
C_COMP_2 = "#996600"   # green  – competitive zone

# ── constants ─────────────────────────────────────────────────────────────────
E_MAX     = 6.776   # kWh at which efficiency score → 0 (per FSAE rules)
COMP_KWH  = 5.0   # competitive energy lower bound
COMP_KWH_2  = 6.5   # competitive energy lower bound
X_LIM     = 7.5   # x-axis ceiling for all plots

# ── %% raw vectors ────────────────────────────────────────────────────────────
kwh   = df_scored["energy_used_kwh"].values
end   = df_scored["endurance"].values
eff   = df_scored["efficiency"].values
years = df_scored["year"].values

# ── %% parametric fits ────────────────────────────────────────────────────────

# Endurance: log curve  end ≈ a + b·ln(kwh)
# Rationale: each extra kWh raises the car's pace, but gains diminish as the
# car is already going fast; always increasing (monotone by construction).
sqrt_kwh       = np.minimum(np.sqrt(kwh), np.ones_like(kwh) * 275)
b_end, a_end  = np.polyfit(sqrt_kwh, end, 1)   # [slope, intercept]

# Efficiency: linear decay anchored to 0 at E_MAX  eff ≈ m·(E_MAX - kwh)
# Rationale: FSAE rules define efficiency score as proportional to energy saved
# vs. the fastest car; it must reach 0 at the maximum allowed energy budget.
# OLS with no intercept on the feature (E_MAX - kwh):
feat_eff = E_MAX - kwh
m_eff    = np.dot(feat_eff, eff) / np.dot(feat_eff, feat_eff)

# Dense grid for plotting
x_plot = np.linspace(kwh.min(), X_LIM, 400)

end_s  = np.minimum(isotonic_increasing(a_end + b_end * np.sqrt(x_plot)), np.ones_like(x_plot) * 275)   # log, monotone
eff_s  = np.maximum(m_eff * (E_MAX - x_plot), 0.0)             # linear → 0
comb_s = end_s + eff_s

# Peak of combined (analytical note: occurs where b/x = m_eff → x = b/m_eff,
# but isotonic + clip may shift it slightly; find numerically)
opt_idx = np.nanargmax(comb_s)
opt_kwh = x_plot[opt_idx]
opt_val = comb_s[opt_idx]

print(f"\nFit parameters:")
print(f"  Endurance  sqrt fit : end = {a_end:.1f} + {b_end:.1f}·sqrt(kWh)")
print(f"  Efficiency lin fit : eff = {m_eff:.1f}·({E_MAX} - kWh)")
print(f"  Peak combined at   : {opt_kwh:.2f} kWh  ({opt_val:.0f} pts)")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 – individual fits (scatter + curve, side by side)
# ─────────────────────────────────────────────────────────────────────────────
# %% individual fit panels
fig, axes = plt.subplots(3, 1, figsize=(6.5, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
fig.suptitle("Endurance + Efficiency Score vs Energy Usage", fontsize=13, fontweight="bold")

# ── Left panel: Endurance ─────────────────────────────────────────────────────
ax1 = axes[1]
for yr in [2024, 2025]:
    m = years == yr
    ax1.scatter(
        kwh[m], end[m], color=colors[yr], s=40, zorder=3,
        label=f"{yr}", alpha=0.8, edgecolors="white", linewidths=0.5,
    )
ax1.plot(x_plot, end_s, color=C_END, lw=2.5, label=f"Sqrt Fit  (R² = {np.corrcoef(end, a_end + b_end*sqrt_kwh)[0,1]**2:.2f})")
# ax1.set_xlabel("Energy Used (kWh)", fontsize=11)
ax1.set_ylabel("Endurance Score", fontsize=11)
# ax1.set_title("Endurance Score", fontsize=10, fontweight="bold")
ax1.set_xlim(right=X_LIM)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.22)

# ── Right panel: Efficiency ───────────────────────────────────────────────────
ax2 = axes[2]
for yr in [2024, 2025]:
    m = years == yr
    ax2.scatter(
        kwh[m], eff[m], color=colors[yr], s=40, zorder=3,
        label=f"{yr}", alpha=0.8, edgecolors="white", linewidths=0.5,
    )
eff_pred = np.maximum(m_eff * (E_MAX - kwh), 0.0)
ax2.plot(x_plot, eff_s, color=C_EFF, lw=2.5, label=f"Linear Fit  (R² = {np.corrcoef(eff, eff_pred)[0,1]**2:.2f})")
ax2.axvline(E_MAX, color=C_EFF, lw=1.4, ls=":", alpha=0.75, label=f"Rules Cap → 0 at {E_MAX} kWh")
ax2.set_xlabel("Energy Used (kWh)", fontsize=11)
ax2.set_ylabel("Efficiency Score", fontsize=11)
# ax2.set_title(f"Efficiency Score", fontsize=10, fontweight="bold")
ax2.set_xlim(right=X_LIM)
ax2.set_ylim(bottom=0)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.22)

# plt.tight_layout()
# plt.savefig(FIGURES / "03_ee_individual_fits.png", dpi=150, bbox_inches="tight")
# plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 – hero chart: all three curves + competitive zone (no raw scatter)
# ─────────────────────────────────────────────────────────────────────────────
# %% hero chart
# fig, ax = plt.subplots(figsize=(10, 6))

ax = axes[0]

# individual curves
ax.plot(x_plot, end_s,  color=C_END,  lw=2.0, ls=":",  alpha=0.85, label="End. Pts")
ax.plot(x_plot, eff_s,  color=C_EFF,  lw=2.0, ls=":",  alpha=0.85, label="Eff. Pts")

# combined curve (bold)
ax.plot(x_plot, comb_s, color=C_COMB, lw=3.0, label="Total Pts")

# competitive zone: green shading from COMP_KWH → X_LIM
ax.axvspan(COMP_KWH, COMP_KWH_2, alpha=0.10, color=C_COMP)
ax.axvline(COMP_KWH, color=C_COMP, lw=2.2, ls="--", alpha=0.95)

ax.axvspan(COMP_KWH_2, X_LIM, alpha=0.10, color=C_COMP_2)
ax.axvline(COMP_KWH_2, color=C_COMP_2, lw=2.2, ls="--", alpha=0.95)

# annotation — placed inside the green zone, low enough to clear all curves
y_ann = float(np.interp(COMP_KWH, x_plot, comb_s))
ax.annotate(
    f"≥ {COMP_KWH} kWh\nfor 250+ Points",
    xy=(COMP_KWH, y_ann),
    xytext=(COMP_KWH + 0.2, 80),
    fontsize=10, color=C_COMP, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_COMP, alpha=0.9),
    # arrowprops=dict(arrowstyle="->", color=C_COMP, lw=1.3),
)

y_ann = float(np.interp(COMP_KWH_2, x_plot, comb_s))
ax.annotate(
    f"≥ {COMP_KWH_2} kWh\nfor Max Points",
    xy=(COMP_KWH_2, y_ann),
    xytext=(COMP_KWH_2 + 0.2, 80),
    fontsize=10, color=C_COMP_2, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C_COMP_2, alpha=0.9),
    # arrowprops=dict(arrowstyle="->", color=C_COMP_2, lw=1.3),
)

# ax.set_xlabel("Energy Used (kWh)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
# ax.set_title(
#     "Energy Optimization: kWh vs. Endurance & Efficiency Scores\n",
#     fontsize=12, fontweight="bold",
# )
ax.set_xlim(right=X_LIM)
ax.legend(fontsize=9, loc="upper left", ncol=2)
ax.grid(True, alpha=0.22)

plt.tight_layout()
fig.align_ylabels()
plt.savefig(FIGURES / "03_ee_optimization_hero.png", dpi=150)
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 – sensitivity table
# ─────────────────────────────────────────────────────────────────────────────
# %% sensitivity printout
print("\n=== Sensitivity: kWh vs predicted combined E+E score ===")
sample_kwh = np.arange(
    np.floor(kwh.min() * 10) / 10,
    X_LIM + 0.1,
    0.25,
)
end_interp  = np.interp(sample_kwh, x_plot, end_s)
eff_interp  = np.interp(sample_kwh, x_plot, eff_s)
comb_interp = np.interp(sample_kwh, x_plot, comb_s)

sens_df = pd.DataFrame({
    "kWh":             np.round(sample_kwh, 2),
    "pred_endurance":  np.round(end_interp, 1),
    "pred_efficiency": np.round(eff_interp, 1),
    "pred_combined":   np.round(comb_interp, 1),
})
sens_df["delta_from_peak"] = np.round(comb_interp - opt_val, 1)
print(sens_df.to_string(index=False))
print(f"\n★ Peak combined at  : {opt_kwh:.3f} kWh  ({opt_val:.1f} pts)")
print(f"★ Competitive bound : {COMP_KWH} kWh")
print(f"★ Data range        : {kwh.min():.3f} – {kwh.max():.3f} kWh")
