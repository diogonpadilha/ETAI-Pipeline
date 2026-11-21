"""
Compares numeric imputers (median, knn, iterative, random_forest) using Nadeau & Bengio's corrected paired t-test.
"""
import copy
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import load_data
from src.preprocessing import load_config, preprocess
from src.model import build_pipeline
from src.evaluate import corrected_paired_test


def main():
    config = load_config("config.yaml")
    df = load_data(config["data"]["path"])
    strategies = ["median", "knn", "iterative", "random_forest"]

    cv_cfg = config.get("cv", {})
    n_splits = cv_cfg.get("n_splits", 5)
    seed = cv_cfg.get("random_state", 42)
    metric = cv_cfg.get("selection_metric", "roc_auc")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores_by_strategy = {}
    times_by_strategy = {}

    for strat in strategies:
        cfg = copy.deepcopy(config)
        cfg["preprocessing"]["imputation"]["numeric_strategy"] = strat
        X_dev, _, y_dev, _, _, preprocessor = preprocess(df, config=cfg)
        pipe = build_pipeline(preprocessor, cfg["model"], cfg.get("feature_selection", {}))

        t0 = time.time()
        fold_scores = cross_val_score(pipe, X_dev, y_dev, cv=cv, scoring=metric)
        times_by_strategy[strat] = time.time() - t0
        scores_by_strategy[strat] = fold_scores

    n_total = len(y_dev)
    n_test = n_total // n_splits
    n_train = n_total - n_test
    baseline = scores_by_strategy["median"]

    print(f"\n{'Imputer':<15} | {'Mean ' + metric:<14} | {'Std':<8} | {'Time (s)':<9} | {'t vs median':<12} | {'p-value':<8}")
    print("-" * 78)
    for strat in strategies:
        sc = scores_by_strategy[strat]
        if strat == "median":
            t_str, p_str = "—", "—"
        else:
            t_val, p_val = corrected_paired_test(sc, baseline, n_train=n_train, n_test=n_test)
            t_str, p_str = f"{t_val:+.3f}", f"{p_val:.4f}"
        print(f"{strat:<15} | {sc.mean():<14.4f} | {sc.std():<8.4f} | {times_by_strategy[strat]:<9.2f} | {t_str:<12} | {p_str:<8}")


if __name__ == "__main__":
    main()