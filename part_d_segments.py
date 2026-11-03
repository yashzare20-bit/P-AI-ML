"""Part D — Hub Segments and Operations Playbook
Clusters 24 hubs into operational segments based on delivery performance, lane distance,
and delivery stops. Evaluates cluster separation, identifies tier-label mismatches, and generates profiles.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")

    # 1. Hub profile (24 rows, sorted by ascending hub_id)
    hp_rows = []
    for hid, sub in df.groupby("hub_id"):
        hp_rows.append({
            "hub_id": hid,
            "shipments": len(sub),
            "mean_delivery_hours": round(float(sub["delivery_hours"].mean()), 4),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_distance_km": round(float(sub["distance_km"].mean()), 4),
            "mean_n_stops": round(float(sub["n_stops"].mean()), 4),
        })

    hub_profile = pd.DataFrame(hp_rows).sort_values("hub_id").reset_index(drop=True)
    hub_profile.to_csv("outputs/hub_profile.csv", index=False)
    print("Wrote outputs/hub_profile.csv (24 rows)")

    # 2. Scaling
    feature_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hub_profile[feature_cols])

    # 3. Separation evidence for K in 2, 3, 4, 5
    dist_matrix = squareform(pdist(X_scaled, metric="euclidean"))
    n_hubs = len(hub_profile)

    sep_records = []
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        # Cluster sizes descending
        counts = pd.Series(labels).value_counts().sort_values(ascending=False).astype(str).tolist()
        cluster_sizes_str = "-".join(counts)

        # Pairwise distances
        across_dists = []
        within_dists = []
        for i in range(n_hubs):
            for j in range(i + 1, n_hubs):
                d = dist_matrix[i, j]
                if labels[i] == labels[j]:
                    within_dists.append(d)
                else:
                    across_dists.append(d)

        ratio = float(np.mean(across_dists) / np.mean(within_dists))
        sep_records.append({
            "K": k,
            "across_over_within": round(ratio, 4),
            "cluster_sizes": cluster_sizes_str,
        })

    pd.DataFrame(sep_records).to_csv("outputs/part_d_separation.csv", index=False)
    print("Wrote outputs/part_d_separation.csv")

    # 4. K=3 segmentation and naming
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    hub_profile["cluster"] = km3.fit_predict(X_scaled)

    # Order clusters by mean delay_rate: Standard (lowest), Watch (middle), Escalate (highest)
    cluster_delay_means = hub_profile.groupby("cluster")["delay_rate"].mean().sort_values()
    ordered_cluster_ids = cluster_delay_means.index.tolist()
    seg_map = {
        ordered_cluster_ids[0]: "Standard",
        ordered_cluster_ids[1]: "Watch",
        ordered_cluster_ids[2]: "Escalate",
    }
    hub_profile["segment"] = hub_profile["cluster"].map(seg_map)

    # Merge hub metadata
    merged_hub = hub_profile.merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id")
    hub_segments = merged_hub[["hub_id", "segment", "hub_city", "hub_tier"]].sort_values("hub_id")
    hub_segments.to_csv("outputs/hub_segments.csv", index=False)
    print("Wrote outputs/hub_segments.csv")

    # Segment profile
    seg_profile_rows = []
    for seg in ["Standard", "Watch", "Escalate"]:
        sub = hub_profile[hub_profile["segment"] == seg]
        seg_profile_rows.append({
            "segment": seg,
            "hubs": len(sub),
            "mean_delivery_hours": round(float(sub["mean_delivery_hours"].mean()), 4),
            "delay_rate": round(float(sub["delay_rate"].mean()), 4),
            "mean_distance_km": round(float(sub["mean_distance_km"].mean()), 4),
            "mean_n_stops": round(float(sub["mean_n_stops"].mean()), 4),
        })
    pd.DataFrame(seg_profile_rows).to_csv("outputs/segment_profile.csv", index=False)
    print("Wrote outputs/segment_profile.csv")

    # Tier vs segment mismatch
    # Implied: Tier 1 -> Standard, Tier 2 -> Watch, Tier 3 -> Escalate
    tier_implied = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatches = hub_segments[hub_segments["segment"] != hub_segments["hub_tier"].map(tier_implied)]
    mismatches.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
    print(f"Wrote outputs/tier_vs_segment_mismatch.csv ({len(mismatches)} mismatches)")


if __name__ == "__main__":
    main()
