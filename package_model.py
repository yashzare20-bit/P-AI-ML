"""Package the final delay classification pipeline and operational artifacts.
Saves model pickle, decision threshold, feature ordering, and imputation medians.
Verifies serializability via test reload check.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

sys.path.insert(0, os.path.abspath("."))
from src.shipment_prep import FEATURES, clean_shipments, build_features


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Read config and raw data
    config_df = pd.read_csv("outputs/part_c_final_config.csv")
    config = config_df.iloc[0]

    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")

    clean_df, medians = clean_shipments(raw)
    model_df = build_features(clean_df, hubs)
    model_df["is_delayed"] = (model_df["delivery_hours"] > model_df["promised_hours"]).astype(int)

    X = model_df[FEATURES]
    y = model_df["is_delayed"]

    # Rebuild Part C stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Parse parameters from config
    selector_k = int(config["selector_k"])
    n_estimators = int(config["n_estimators"])
    max_depth = None if str(config["max_depth"]).lower() in ["unlimited", "none", "nan"] else int(config["max_depth"])
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # Fit final pipeline on training rows
    pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=selector_k)),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Save models/delay_model.pkl
    model_pkl_path = "models/delay_model.pkl"
    joblib.dump(pipeline, model_pkl_path)
    print(f"Saved {model_pkl_path}")

    # Save models/delay_threshold.csv (one column threshold)
    threshold_df = pd.DataFrame([{"threshold": round(threshold, 4)}])
    threshold_df.to_csv("models/delay_threshold.csv", index=False)
    print("Saved models/delay_threshold.csv")

    # Save models/feature_order.csv (one column feature, the 17 names in FEATURES order)
    feature_order_df = pd.DataFrame({"feature": FEATURES})
    feature_order_df.to_csv("models/feature_order.csv", index=False)
    print("Saved models/feature_order.csv")

    # Save models/impute_values.csv (columns column, median — the three medians from clean_shipments)
    impute_df = pd.DataFrame([
        {"column": "distance_km", "median": round(medians["distance_km"], 4)},
        {"column": "parcel_weight_kg", "median": round(medians["parcel_weight_kg"], 4)},
        {"column": "declared_value_inr", "median": round(medians["declared_value_inr"], 4)},
    ])
    impute_df.to_csv("models/impute_values.csv", index=False)
    print("Saved models/impute_values.csv")

    # Reload check
    reloaded_pipeline = joblib.load(model_pkl_path)
    in_memory_probs = pipeline.predict_proba(X_test)[:, 1]
    reloaded_probs = reloaded_pipeline.predict_proba(X_test)[:, 1]

    max_prob_diff = float(np.max(np.abs(in_memory_probs - reloaded_probs)))
    print(f"Reload max probability difference: {max_prob_diff}")

    reload_checks = [
        {"metric": "test_rows_scored", "value": str(len(X_test))},
        {"metric": "reload_max_abs_probability_diff", "value": round(max_prob_diff, 4)},
        {"metric": "threshold", "value": round(threshold, 4)},
    ]
    pd.DataFrame(reload_checks).to_csv("outputs/part_d_reload_check.csv", index=False)
    print("Wrote outputs/part_d_reload_check.csv")


if __name__ == "__main__":
    main()
