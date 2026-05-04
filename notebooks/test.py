# %% imports
import pandas as pd
import matplotlib.pyplot as plt

# %% load data
results = pd.read_csv("../data/raw/fsae_ev_results.csv")
print(results.shape)
results.head()

# %% quick plot - 2025 top 10 total scores
top10 = results[results.year == 2025].nlargest(10, "total")

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(top10.team, top10.total)
ax.invert_yaxis()
ax.set_xlabel("Total Score")
ax.set_title("FSAE EV 2025 - Top 10")
plt.tight_layout()
plt.show()
