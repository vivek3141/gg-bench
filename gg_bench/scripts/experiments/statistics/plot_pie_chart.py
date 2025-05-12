import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data and settings
sizes = [2971, 325, 184]
labels = ["Losses (cnt=2971)", "Draws (cnt=325)", "Faults (cnt=184)"]
percentages = [81.98, 8.97, 5.29]
palette = sns.color_palette("pastel")[:3]

fig, ax = plt.subplots(figsize=(3.2, 3.2))

# Plot pie without percentage labels
wedges, texts = ax.pie(
    sizes,
    labels=None,
    autopct=None,
    colors=palette,
    startangle=170,
    textprops={"fontsize": 10},
    pctdistance=0.65,
    radius=0.90,
)

# Custom label placement
for i, (pct, wedge) in enumerate(zip(percentages, wedges)):
    ang = (wedge.theta2 + wedge.theta1) / 2.0
    r = 0.65
    y_offset = 0.0

    if np.isclose(pct, 5.29, atol=0.1):
        y_offset = -0.015
    if np.isclose(pct, 81.98, atol=0.1):
        r = 0.40

    x = r * np.cos(np.deg2rad(ang))
    y = r * np.sin(np.deg2rad(ang)) + y_offset

    ax.text(x, y, f"{pct:.2f}%", ha="center", va="center", fontsize=10)

# Add legend
ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=15)
ax.axis("equal")
plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)
plt.show()
plt.savefig("docs/figures/win_loss_pie.pdf", bbox_inches="tight")
