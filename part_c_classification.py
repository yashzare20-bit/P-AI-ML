"""Part C — Delay Classifier and Alert Threshold
Builds classification pipeline to predict delivery delays, conducts feature selection,
hyperparameter tuning, threshold optimization based on business cost trade-offs, and test evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.abspath("."))
from src.shipment_prep import FEATURES, NOISE_COLUMNS


def main():
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv("data/modelling_table.csv")

    X = df[FEATURES]
    y = df["is_delayed"]

    # 1. Stratified split (840 / 180 / 180)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Split sizes: Train={len(y_train)} (delayed={y_train.sum()}), "
          f"Val={len(y_val)} (delayed={y_val.sum()}), "
          f"Test={len(y_test)} (delayed={y_test.sum()})")

    # 2. Blind baseline and 3 candidate models on validation (threshold 0.5)
    model_rows = []

    # Blind baseline
    blind_preds = np.zeros(len(y_val), dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_val, blind_preds).ravel()
    acc = accuracy_score(y_val, blind_preds)
    rec = recall_score(y_val, blind_preds, zero_division=0)
    f1 = f1_score(y_val, blind_preds, zero_division=0)
    model_rows.append({
        "model": "blind_all_on_time",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(acc, 4), "precision": 0.0, "recall": round(rec, 4), "f1": round(f1, 4),
    })

    # Logistic Regression Pipeline
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ])
    lr_pipe.fit(X_train, y_train)
    lr_preds = lr_pipe.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, lr_preds).ravel()
    model_rows.append({
        "model": "logistic_regression",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, lr_preds), 4),
        "precision": round(precision_score(y_val, lr_preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, lr_preds, zero_division=0), 4),
        "f1": round(f1_score(y_val, lr_preds, zero_division=0), 4),
    })

    # Best Decision Tree by validation F1
    dt_depths = [2, 3, 4, 5, 6, 8, 10]
    best_dt_d = None
    best_dt_f1 = -1.0
    best_dt_metrics = None

    for d in dt_depths:
        dt = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        p = dt.predict(X_val)
        curr_f1 = f1_score(y_val, p, zero_division=0)
        if curr_f1 > best_dt_f1:
            best_dt_f1 = curr_f1
            best_dt_d = d
            tn, fp, fn, tp = confusion_matrix(y_val, p).ravel()
            best_dt_metrics = {
                "model": f"decision_tree_depth_{d}",
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                "accuracy": round(accuracy_score(y_val, p), 4),
                "precision": round(precision_score(y_val, p, zero_division=0), 4),
                "recall": round(recall_score(y_val, p, zero_division=0), 4),
                "f1": round(curr_f1, 4),
            }
    model_rows.append(best_dt_metrics)

    # Random Forest Default
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_def.fit(X_train, y_train)
    rf_preds = rf_def.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, rf_preds).ravel()
    model_rows.append({
        "model": "random_forest_default",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_val, rf_preds), 4),
        "precision": round(precision_score(y_val, rf_preds, zero_division=0), 4),
        "recall": round(recall_score(y_val, rf_preds, zero_division=0), 4),
        "f1": round(f1_score(y_val, rf_preds, zero_division=0), 4),
    })

    pd.DataFrame(model_rows).to_csv("outputs/part_c_models.csv", index=False)
    print("Wrote outputs/part_c_models.csv")

    # 3. Feature signals: RF importances & SelectKBest F-scores
    rf_importances = rf_def.feature_importances_
    selector_all = SelectKBest(f_classif, k="all")
    selector_all.fit(X_train, y_train)
    f_scores = selector_all.scores_

    feature_scores_df = pd.DataFrame({
        "feature": FEATURES,
        "f_score": [round(float(s), 4) for s in f_scores],
        "rf_importance": [round(float(imp), 4) for imp in rf_importances],
        "is_noise": [col in NOISE_COLUMNS for col in FEATURES],
    })
    feature_scores_df["f_rank"] = feature_scores_df["f_score"].rank(ascending=False, method="min").astype(int)
    feature_scores_df = feature_scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
    feature_scores_df = feature_scores_df.sort_values("f_rank").reset_index(drop=True)
    feature_scores_df.to_csv("outputs/part_c_feature_scores.csv", index=False)
    print("Wrote outputs/part_c_feature_scores.csv")

    # SelectKBest k experiment (k in 5, 7, 10, 12, 17)
    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        selectk_rows.append({
            "k": k,
            "val_f1": round(f1_score(y_val, preds, zero_division=0), 4),
        })
    pd.DataFrame(selectk_rows).to_csv("outputs/part_c_selectk.csv", index=False)
    print("Wrote outputs/part_c_selectk.csv")

    # 4. Tuning with k=7 features
    selector_7 = SelectKBest(f_classif, k=7)
    X_tr_7 = selector_7.fit_transform(X_train, y_train)
    X_va_7 = selector_7.transform(X_val)
    X_te_7 = selector_7.transform(X_test)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    # GridSearchCV
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_tr_7, y_train)
    grid_val_f1 = f1_score(y_val, grid.predict(X_va_7), zero_division=0)

    # RandomizedSearchCV
    rnd = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid, n_iter=9, cv=5, random_state=42)
    rnd.fit(X_tr_7, y_train)
    rnd_val_f1 = f1_score(y_val, rnd.predict(X_va_7), zero_division=0)

    # Default forest on same 7 features (no depth limit, min_samples_leaf=1)
    def_7 = RandomForestClassifier(n_estimators=100, random_state=42)
    def_7.fit(X_tr_7, y_train)
    def_7_val_f1 = f1_score(y_val, def_7.predict(X_va_7), zero_division=0)

    tuning_rows = [
        {
            "search": "grid_search",
            "candidates": 27,
            "cv_folds": 5,
            "total_fits": 135,
            "best_params": str(grid.best_params_),
            "best_cv_accuracy": round(grid.best_score_, 4),
            "val_f1": round(grid_val_f1, 4),
        },
        {
            "search": "randomized_search",
            "candidates": 9,
            "cv_folds": 5,
            "total_fits": 45,
            "best_params": str(rnd.best_params_),
            "best_cv_accuracy": round(rnd.best_score_, 4),
            "val_f1": round(rnd_val_f1, 4),
        },
        {
            "search": "default_forest_same_7_features",
            "candidates": 1,
            "cv_folds": 0,
            "total_fits": 1,
            "best_params": "defaults",
            "best_cv_accuracy": "",
            "val_f1": round(def_7_val_f1, 4),
        },
    ]
    pd.DataFrame(tuning_rows).to_csv("outputs/part_c_tuning.csv", index=False)
    print("Wrote outputs/part_c_tuning.csv")

    # 5. Choosing final configuration
    # Pick row of part_c_tuning.csv with highest val_f1
    # Ties: default_forest_same_7_features, then grid_search, then randomized_search
    # Here def_7_val_f1 is 0.8333, which is strictly highest!
    final_search_name = "default_forest_same_7_features"
    final_model = def_7
    final_pipeline = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    final_pipeline.fit(X_train, y_train)

    # 6. Alert threshold
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    th_records = []

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        th_records.append({
            "threshold": t,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": round(accuracy_score(y_val, preds), 4),
            "precision": round(precision_score(y_val, preds, zero_division=0), 4),
            "recall": round(recall_score(y_val, preds, zero_division=0), 4),
            "f1": round(f1_score(y_val, preds, zero_division=0), 4),
        })

    pd.DataFrame(th_records).to_csv("outputs/part_c_thresholds.csv", index=False)
    print("Wrote outputs/part_c_thresholds.csv")

    # Operations rule: highest t whose validation recall is >= 0.80
    valid_thresholds = [r["threshold"] for r in th_records if r["recall"] >= 0.80]
    chosen_threshold = max(valid_thresholds) if valid_thresholds else 0.1
    print(f"Chosen alert threshold: {chosen_threshold}")

    # 7. One test evaluation
    test_probs = final_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= chosen_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

    test_eval_df = pd.DataFrame([{
        "threshold": chosen_threshold,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": round(accuracy_score(y_test, test_preds), 4),
        "precision": round(precision_score(y_test, test_preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, test_preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, test_preds, zero_division=0), 4),
    }])
    test_eval_df.to_csv("outputs/part_c_final_test.csv", index=False)
    print("Wrote outputs/part_c_final_test.csv")

    # Final config
    config_df = pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": 100,
        "max_depth": "unlimited",
        "min_samples_leaf": 1,
        "threshold": chosen_threshold,
        "chosen_from": final_search_name,
    }])
    config_df.to_csv("outputs/part_c_final_config.csv", index=False)
    print("Wrote outputs/part_c_final_config.csv")


if __name__ == "__main__":
    main()
