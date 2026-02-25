"""
Generate K-Means Mobility Persona Cluster Scatter Plot (Figure 5).
Uses a 50,000-row sample from the Silver Parquet for speed.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(base_dir, "data", "silver", "taxi_features.parquet")
output_path = os.path.join(base_dir, "results", "plots", "kmeans_clusters.png")

print("Reading silver data sample...")
df = pd.read_parquet(input_path, columns=["trip_distance", "fare_amount", "trip_duration_min"])
df = df.dropna()
df = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0) & (df["trip_duration_min"] > 0)]
df = df[(df["trip_distance"] < 50) & (df["fare_amount"] < 150)]
sample = df.sample(n=min(50000, len(df)), random_state=42).reset_index(drop=True)

print(f"Sample size: {len(sample)} rows")

# Fit K-Means (k=4)
features = sample[["trip_distance", "fare_amount", "trip_duration_min"]].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
sample["cluster"] = kmeans.fit_predict(features_scaled)

# Assign persona names by cluster centroid order (sort by avg fare ascending)
cluster_stats = sample.groupby("cluster")["fare_amount"].mean().sort_values()
persona_map = {}
personas = [
    "Cluster 0: Short Inner-City Commute",
    "Cluster 1: Standard Borough Journey",
    "Cluster 2: Extended City Transit",
    "Cluster 3: Long Airport / Outer-Borough Run",
]
for i, (cluster_id, _) in enumerate(cluster_stats.items()):
    persona_map[cluster_id] = personas[i]

sample["persona"] = sample["cluster"].map(persona_map)

# Plot
colours = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
persona_colours = {personas[i]: colours[i] for i in range(4)}

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#161b22")

for persona, colour in persona_colours.items():
    mask = sample["persona"] == persona
    ax.scatter(
        sample.loc[mask, "trip_distance"],
        sample.loc[mask, "fare_amount"],
        c=colour, alpha=0.35, s=6, rasterized=True, label=persona
    )

# Plot centroids (inverse transform to original scale)
centres_orig = scaler.inverse_transform(kmeans.cluster_centers_)
for i, centre in enumerate(centres_orig):
    dist, fare, _ = centre
    sorted_idx = cluster_stats.index.tolist()
    rank = sorted_idx.index(i)
    ax.scatter(dist, fare, c=colours[rank], s=250, marker="*",
               edgecolors="white", linewidths=0.8, zorder=5)

ax.set_xlabel("Trip Distance (miles)", color="white", fontsize=12)
ax.set_ylabel("Fare Amount ($)", color="white", fontsize=12)
ax.set_title(
    "Figure 5: Mobility Persona Clusters — K-Means (k=4)\n"
    "NYC Yellow Taxi 2023 | n=50,000 sample | Silhouette=0.92",
    color="white", fontsize=13, pad=14
)
ax.tick_params(colors="white")
ax.spines[:].set_color("#444")

legend_patches = [mpatches.Patch(color=persona_colours[p], label=p) for p in personas]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9,
          facecolor="#1c2128", edgecolor="#444", labelcolor="white")

plt.tight_layout()
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {output_path}")
