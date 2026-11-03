"""Part B — Delivery-Hours Regression
Trains point-estimate regression models for parcel delivery hours, conducts diagnostics,
learning-rate experiments, cross-validation, and final test evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.abspath("."))
from src.shipment_prep import FEATURES


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")

    X = df[FEATURES]
    y = df["delivery_hours"]

    # 1. Split into train, val, test (840 / 180 / 180)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # 2. Scaling without leakage
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    train_max_abs_mean = float(np.max(np.abs(X_tr_s.mean(axis=0))))
    test_max_abs_mean = float(np.max(np.abs(X_te_s.mean(axis=0))))

    b_checks = [
        {"metric": "train_rows", "value": str(len(X_train))},
        {"metric": "val_rows", "value": str(len(X_val))},
        {"metric": "test_rows", "value": str(len(X_test))},
        {"metric": "train_scaled_max_abs_mean", "value": f"{train_max_abs_mean:.4e}"},
        {"metric": "test_scaled_max_abs_mean", "value": f"{test_max_abs_mean:.4f}"},
    ]
    pd.DataFrame(b_checks).to_csv("outputs/part_b_checks.csv", index=False)
    print("Wrote outputs/part_b_checks.csv")

    # 3. Baseline
    base_pred = np.full(len(y_val), y_train.mean())
    base_mae = mean_absolute_error(y_val, base_pred)
    base_mse = mean_squared_error(y_val, base_pred)
    base_rmse = np.sqrt(base_mse)
    base_r2 = r2_score(y_val, base_pred)

    # 4. Tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_records = []
    best_depth = None
    best_val_rmse = float("inf")

    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        tr_pred = dt.predict(X_train)
        va_pred = dt.predict(X_val)

        tr_rmse = np.sqrt(mean_squared_error(y_train, tr_pred))
        va_rmse = np.sqrt(mean_squared_error(y_val, va_pred))

        d_label = "unlimited" if d is None else str(d)
        tree_records.append({
            "max_depth": d_label,
            "train_rmse": round(tr_rmse, 4),
            "val_rmse": round(va_rmse, 4),
        })

        if va_rmse < best_val_rmse:
            best_val_rmse = va_rmse
            best_depth = d

    pd.DataFrame(tree_records).to_csv("outputs/part_b_tree_depth.csv", index=False)
    print(f"Wrote outputs/part_b_tree_depth.csv (best depth: {best_depth})")

    # 5. Four candidate models
    candidates = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    metrics_rows = [
        {
            "model": "mean_baseline",
            "split": "val",
            "MAE": round(base_mae, 4),
            "MSE": round(base_mse, 4),
            "RMSE": round(base_rmse, 4),
            "R2": round(base_r2, 4),
        }
    ]

    val_performance = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        mse = mean_squared_error(y_val, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, preds)

        metrics_rows.append({
            "model": name,
            "split": "val",
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })
        val_performance[name] = {"val_rmse": rmse, "model": model}

    # 6. Learning-rate experiment
    lr_rows = []
    for lr in [0.01, 0.1, 1.0]:
        gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb.fit(X_train, y_train)
        preds = gb.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        lr_rows.append({
            "learning_rate": lr,
            "val_rmse": round(rmse, 4),
            "val_r2": round(r2, 4),
        })
    pd.DataFrame(lr_rows).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    print("Wrote outputs/part_b_gb_learning_rate.csv")

    # 7. Cross-validation on top 2 models & final test evaluation
    # Rank candidates by validation RMSE
    ranked_candidates = sorted(val_performance.items(), key=lambda item: item[1]["val_rmse"])
    top2 = ranked_candidates[:2]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    for name, data in top2:
        scores = cross_val_score(data["model"], X_train_val, y_train_val, cv=kf)
        cv_rows.append({
            "model": name,
            "mean_r2": round(float(scores.mean()), 4),
            "std_r2": round(float(scores.std()), 4),
        })
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)
    print("Wrote outputs/part_b_cv.csv")

    # Final model is the candidate with lowest validation RMSE
    final_name = ranked_candidates[0][0]
    final_model = ranked_candidates[0][1]["model"]

    # Final test evaluation (scored once)
    test_preds = final_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_preds)

    metrics_rows.append({
        "model": final_name,
        "split": "test",
        "MAE": round(test_mae, 4),
        "MSE": round(test_mse, 4),
        "RMSE": round(test_rmse, 4),
        "R2": round(test_r2, 4),
    })

    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
    print(f"Wrote outputs/part_b_metrics.csv (final model on test: {final_name})")


if __name__ == "__main__":
    main()
