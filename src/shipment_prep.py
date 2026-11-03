"""Shipment preprocessing, feature engineering, and shared constants.
Delhivery Last-Mile Delay Intelligence Pipeline.
"""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = [
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"
]


def clean_shipments(raw: pd.DataFrame, medians: dict = None):
    """Clean raw shipments export and handle missing values.

    Steps applied in strict order:
    1. Drop exact duplicate rows.
    2. Normalize service_type: strip whitespace and convert to title case.
    3. Impute missing payment_mode with 'Unknown'.
    4. Parse declared_value (strip rupee symbol and commas), cast to float,
       store in declared_value_inr, and drop the text column.
    5. Convert dispatch_time to datetime.
    6. Mask scanner-fault sentinel weights (> 100 kg) as missing (NaN).
    7. Fill missing distance_km, parcel_weight_kg, and declared_value_inr with
       medians. If medians is None, calculate from non-missing values;
       otherwise use provided medians dictionary.

    Returns:
        tuple: (clean_df, medians_dict)
    """
    # 1. Drop exact duplicate rows
    df = raw.drop_duplicates().copy()

    # 2. Service type normalization
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. Payment mode placeholder
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. Declared value parsing
    df["declared_value_inr"] = df["declared_value"].apply(
        lambda x: float(str(x).replace("₹", "").replace(",", "").strip()) if pd.notna(x) else np.nan
    )
    df = df.drop(columns=["declared_value"])

    # 5. Dispatch time parsing
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. Sentinel weight treatment (> 100 kg treated as scanner malfunction)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. Imputation medians
    if medians is None:
        medians = {
            "distance_km": float(df["distance_km"].median()),
            "parcel_weight_kg": float(df["parcel_weight_kg"].median()),
            "declared_value_inr": float(df["declared_value_inr"].median()),
        }

    df["distance_km"] = df["distance_km"].fillna(medians["distance_km"])
    df["parcel_weight_kg"] = df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
    df["declared_value_inr"] = df["declared_value_inr"].fillna(medians["declared_value_inr"])

    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub metadata and engineer model features without using delivery_hours.

    Appends:
    - hub_city, region, hub_tier, hub_daily_capacity (via left merge on hub_id)
    - promised_hours: mapped from PROMISED_HOURS based on service_type
    - is_cod: binary indicator (1 if payment_mode is 'COD', else 0)
    - is_weekend: binary indicator (1 if dispatch on Saturday/Sunday, else 0)
    - dispatch_hour: integer hour from dispatch_time (6 to 21)
    - log_declared_value: natural logarithm of declared_value_inr

    Returns:
        pd.DataFrame: Merged and feature-engineered DataFrame.
    """
    df = clean.merge(
        hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
        on="hub_id",
        how="left",
    )

    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.day_of_week >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"].astype(float))

    return df
