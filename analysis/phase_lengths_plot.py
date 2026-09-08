import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "Exploration": [14.7, 11.4, 6.4, 11.0, 4.5],
    "Execution": [6.1, 6.4, 6.0, 8.8, 6.5],
    "Validation": [7.6, 6.3, 7.1, 4.8, 6.0]
})

# Convert to long format for seaborn
data_long = data.melt(id_vars="x", var_name="Phase Type", value_name="Value")

sns.set_theme(style="whitegrid")

plt.figure()

sns.lineplot(
    data=data_long,
    x="x",
    y="Value",
    hue="Phase Type",
    marker="o"
)

plt.xticks([1,2,3,4,5])
plt.ylim(bottom=0)  # start y-axis at 0

plt.xlabel("Cycle #")
plt.ylabel("Steps")
plt.title("Phase Length by Cycle #")

plt.savefig("phase_length_plot.pdf", bbox_inches="tight")
plt.show()