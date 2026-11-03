"""Part A — Data Foundation and Delay Evidence
Cleans historical shipment data, engineers features, generates QA checks and delay evidence.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.shipment_prep import FEATURES, clean_shipments, build_features


def format_check_val(val):
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        rounded = round(val, 4)
        # Avoid trailing zeros if it is an exact simple decimal like 1121.5 or 314.95
        s = f"{rounded:.4f}"
        if s.endswith("00"):
            s = s[:-2]
        if s.endswith("0") and not s.endswith(".0"):
            s = s[:-1]
        return s
    return str(val)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    raw_path = "data/raw_shipments.csv"
    hubs_path = "data/hubs.csv"

    raw = pd.read_csv(raw_path)
    hubs = pd.read_csv(hubs_path)

    raw_rows = len(raw)
    raw_dedup = raw.drop_duplicates()
    duplicate_rows_removed = raw_rows - len(raw_dedup)
    rows_after_dedup = len(raw_dedup)

    missing_dist_before = int(raw_dedup["distance_km"].isna().sum())
    missing_weight_before = int(raw_dedup["parcel_weight_kg"].isna().sum())
    sentinel_weight_rows = int((raw_dedup["parcel_weight_kg"] > 100).sum())
    missing_val_before = int(raw_dedup["declared_value"].isna().sum())
    missing_pay_before = int(raw_dedup["payment_mode"].isna().sum())
    spellings_before = int(raw_dedup["service_type"].nunique())

    clean_df, medians = clean_shipments(raw)
    spellings_after = int(clean_df["service_type"].nunique())

    model_df = build_features(clean_df, hubs)
    rows_after_hub_merge = len(model_df)

    # Add outcome target
    model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

    # Save data/modelling_table.csv
    model_df.to_csv("data/modelling_table.csv", index=False)
    print("Wrote data/modelling_table.csv (rows:", len(model_df), ")")

    # Checks
    cols_to_check = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_features_target = int(model_df[cols_to_check].isna().sum().sum())
    delayed_count = int(model_df["is_delayed"].sum())
    delay_rate = float(model_df["is_delayed"].mean())

    q1 = float(model_df["delivery_hours"].quantile(0.25))
    q3 = float(model_df["delivery_hours"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((model_df["delivery_hours"] < lower_bound) | (model_df["delivery_hours"] > upper_bound)).sum())

    val_mean = float(model_df["declared_value_inr"].mean())
    val_median = float(model_df["declared_value_inr"].median())
    log_val_mean = float(model_df["log_declared_value"].mean())
    log_val_median = float(model_df["log_declared_value"].median())

    checks = [
        ("raw_rows", raw_rows),
        ("duplicate_rows_removed", duplicate_rows_removed),
        ("rows_after_dedup", rows_after_dedup),
        ("missing_distance_km_before", missing_dist_before),
        ("missing_parcel_weight_before", missing_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_val_before),
        ("missing_payment_mode_before", missing_pay_before),
        ("service_type_spellings_before", spellings_before),
        ("service_type_spellings_after", spellings_after),
        ("median_distance_km", medians["distance_km"]),
        ("median_parcel_weight_kg", medians["parcel_weight_kg"]),
        ("median_declared_value_inr", medians["declared_value_inr"]),
        ("rows_after_hub_merge", rows_after_hub_merge),
        ("nulls_in_features_and_target", nulls_features_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", q1),
        ("hours_q3", q3),
        ("hours_lower_bound", lower_bound),
        ("hours_upper_bound", upper_bound),
        ("hours_outlier_count", outlier_count),
        ("declared_value_mean_inr", val_mean),
        ("declared_value_median_inr", val_median),
        ("log_declared_value_mean", log_val_mean),
        ("log_declared_value_median", log_val_median),
    ]

    checks_df = pd.DataFrame([
        {"metric": k, "value": format_check_val(v)} for k, v in checks
    ])
    checks_df.to_csv("outputs/part_a_checks.csv", index=False)
    print("Wrote outputs/part_a_checks.csv")

    # Delay evidence table
    model_df["distance_band"] = pd.cut(
        model_df["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True,
    )

    group_rows = []
    # service_type (ordered as Economy, Express, Standard)
    for s_type in ["Economy", "Express", "Standard"]:
        sub = model_df[model_df["service_type"] == s_type]
        group_rows.append({
            "dimension": "service_type",
            "group": s_type,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # hub_tier (1, 2, 3)
    for tier in [1, 2, 3]:
        sub = model_df[model_df["hub_tier"] == tier]
        group_rows.append({
            "dimension": "hub_tier",
            "group": str(tier),
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    # distance_band
    for band in ["0-200", "201-400", "401-600", "601+"]:
        sub = model_df[model_df["distance_band"] == band]
        group_rows.append({
            "dimension": "distance_band",
            "group": band,
            "shipments": len(sub),
            "delayed": int(sub["is_delayed"].sum()),
            "delay_rate": round(float(sub["is_delayed"].mean()), 4),
            "mean_hours": round(float(sub["delivery_hours"].mean()), 4),
        })

    delay_group_df = pd.DataFrame(group_rows)
    delay_group_df.to_csv("outputs/part_a_delay_by_group.csv", index=False)
    print("Wrote outputs/part_a_delay_by_group.csv")

    # Correlations
    num_cols = model_df.select_dtypes(include=[np.number]).columns
    corr_matrix = model_df[num_cols].corr().round(4)
    corr_matrix.reset_index(names="feature").to_csv("outputs/part_a_correlations.csv", index=False)
    print("Wrote outputs/part_a_correlations.csv")


if __name__ == "__main__":
    main()
