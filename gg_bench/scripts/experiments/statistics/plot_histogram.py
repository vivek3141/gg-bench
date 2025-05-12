import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. load the reference list of valid env IDs
# ---------------------------------------------------------------------
with open("gg_bench/data/splits/valid_envs.json") as f:
    valid_envs = {int(pair[0]) for pair in json.load(f)}  # set for O(1) lookup

# ---------------------------------------------------------------------
# 2. read the pair-similarity data
# ---------------------------------------------------------------------
PATH = "gg_bench/scripts/experiments/statistics/dolos_report/pairs.csv"
pairs = pd.read_csv(PATH)


# helper to turn ".../env_272.py" → 272
def env_id(path: str) -> int:
    return int(Path(path).stem.split("_")[1])


# validate *both* columns
for col in ("leftFilePath", "rightFilePath"):
    assert all(env_id(p) in valid_envs for p in pairs[col]), f"Bad paths in {col}"

# ---------------------------------------------------------------------
# 3. reshape → one (fileId, similarity) per row
# ---------------------------------------------------------------------
left = pairs[["leftFileId", "similarity"]].rename(columns={"leftFileId": "fileId"})
right = pairs[["rightFileId", "similarity"]].rename(columns={"rightFileId": "fileId"})
stacked = pd.concat([left, right], ignore_index=True)

# ---------------------------------------------------------------------
# 4. keep the *highest* similarity for each file
# ---------------------------------------------------------------------
best_per_file = stacked.groupby("fileId")["similarity"].max().sort_index()
n_files = len(best_per_file)
assert n_files == len(
    valid_envs
), f"Expected {len(valid_envs)} distinct files, got {n_files}"

# ---------------------------------------------------------------------
# 5. summary statistics
# ---------------------------------------------------------------------
print(f"Highest similarity per file (n = {n_files})\n")
print(best_per_file.describe())
print(f"\nVariance : {best_per_file.var():.6f}")
print(f"Skewness : {best_per_file.skew():.6f}")
print(f"Kurtosis : {best_per_file.kurt():.6f}")

# ---------------------------------------------------------------------
# 6. histogram (publication-ready)
#    – Freedman–Diaconis bin width, clipped to 10–60 bins
# ---------------------------------------------------------------------
data = best_per_file.to_numpy()
q25, q75 = np.percentile(data, [25, 75])
iqr = q75 - q25
fd_width = 2 * iqr / (len(data) ** (1 / 3))
bins = int(round((data.max() - data.min()) / fd_width)) if fd_width else 30
bins = max(10, min(bins, 60))  # sensible bounds

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
ax.hist(data, bins=bins, edgecolor="black", color="#30bcfc")
ax.set_title(f"Histogram of Highest Similarity per File (n = {n_files})", pad=10)
ax.set_xlabel("Highest similarity score")
ax.set_ylabel("Number of files")
ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
fig.tight_layout()

# save a vector PDF for the paper
fig.savefig("docs/figures/highest_similarity_histogram.pdf")
plt.show()
