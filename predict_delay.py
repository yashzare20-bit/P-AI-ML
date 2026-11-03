"""Part D — Production Scoring Script
Consumes new shipment batch, reuses shared preprocessing with stored medians,
generates delay probabilities with packaged classifier, applies alert threshold,
and routes shipments to operational playbooks based on hub segmentation.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.shipment_prep import clean_shipments, build_features


def main():
    os.makedirs("outputs", exist_ok=True)

    # 1. Load data and models artifacts
    new_raw = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    model = joblib.load("models/delay_model.pkl")
    threshold_df = pd.read_csv("models/delay_threshold.csv")
    threshold = float(threshold_df["threshold"].iloc[0])

    feature_order_df = pd.read_csv("models/feature_order.csv")
    feature_order = feature_order_df["feature"].tolist()

    impute_df = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(impute_df["column"], impute_df["median"]))

    hub_segments_df = pd.read_csv("outputs/hub_segments.csv")
    hub_seg_map = dict(zip(hub_segments_df["hub_id"], hub_segments_df["segment"]))

    # 2. Clean new shipments using stored medians
    clean_new, _ = clean_shipments(new_raw, medians=medians)

    # 3. Build features
    feat_new = build_features(clean_new, hubs)

    # 4. Arrange columns strictly according to feature_order
    X_new = feat_new[feature_order]

    # 5. Predict probabilities and flag
    probs = model.predict_proba(X_new)[:, 1].round(4)
    flagged = (probs >= threshold).astype(int)

    # 6. Map hub segments and determine operational actions
    segments = clean_new["hub_id"].map(hub_seg_map)

    actions = []
    for f, seg in zip(flagged, segments):
        if f == 0:
            actions.append("No action")
        elif seg == "Escalate":
            actions.append("Escalate to hub manager")
        else:
            actions.append("Proactive customer call")

    # 7. Construct action table in file order
    action_table = pd.DataFrame({
        "shipment_id": clean_new["shipment_id"],
        "hub_id": clean_new["hub_id"],
        "segment": segments,
        "delay_probability": probs,
        "flagged": flagged,
        "action": actions,
    })

    output_path = "outputs/new_shipment_actions.csv"
    action_table.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(action_table)} shipments)")


if __name__ == "__main__":
    main()
