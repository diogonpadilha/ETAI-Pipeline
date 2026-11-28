"""
Evaluates AnomalyScore methods (none, isolation_forest, lof) on identical CV folds
and checks recidivism rates among top 5% most anomalous individuals (Challenge 4).
"""
import copy
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import load_data
from src.preprocessing import AnomalyScore, load_config, preprocess
from src.model import build_pipeline
from src.evaluate import corrected_paired_test


def main():
    config = load_config("config.yaml")
    df = load_data(config["data"]["path"])

    X_dev, _, y_dev, _, _, preprocessor = preprocess(df, config=config)
    Z = preprocessor.fit_transform(X_dev, y_dev)

    # 1. Taxa de reincidência: Top 5% mais anómalos vs restantes 95%
    print("=== Recidivism Rate: Top 5% Most Unusual vs Rest (Dev Set) ===")
    for method in ["isolation_forest", "lof"]:
        scorer = AnomalyScore(method=method, n_neighbors=20, random_state=42)
        Z_plus = scorer.fit_transform(Z, y_dev)
        scores = np.asarray(Z_plus)[:, -1]
        cutoff = np.percentile(scores, 95)
        is_top5 = scores >= cutoff
        rate_top5 = y_dev.values[is_top5].mean()
        rate_rest = y_dev.values[~is_top5].mean()
        print(
            f"  {method:<18s} | cutoff={cutoff:.3f} | "
            f"Top 5% recid rate = {rate_top5:.3f} (n={is_top5.sum()}) | "
            f"Rest 95% recid rate = {rate_rest:.3f} (n={(~is_top5).sum()})"
        )

    # 2. Comparação em CV (none vs isolation_forest vs lof)
    methods = ["none", "isolation_forest", "lof"]
    cv_cfg = config.get("cv", {})
    n_splits = cv_cfg.get("n_splits", 5)
    seed = cv_cfg.get("random_state", 42)
    metric = cv_cfg.get("selection_metric", "roc_auc")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores_by_method = {}
    times_by_method = {}

    for m in methods:
        cfg = copy.deepcopy(config)
        cfg.setdefault("anomaly", {})["method"] = m
        cfg["anomaly"].setdefault("n_neighbors", 20)
        cfg["anomaly"].setdefault("random_state", 42)

        X_d, _, y_d, _, _, prep = preprocess(df, config=cfg)
        pipe = build_pipeline(
            prep,
            cfg["model"],
            fs_config=cfg.get("feature_selection", {}),
            anomaly_config=cfg["anomaly"],
        )

        t0 = time.time()
        fold_scores = cross_val_score(pipe, X_d, y_d, cv=cv, scoring=metric)
        times_by_method[m] = time.time() - t0
        scores_by_method[m] = fold_scores

    n_total = len(y_dev)
    n_test = n_total // n_splits
    n_train = n_total - n_test
    baseline = scores_by_method["none"]

    print(f"\n{'Anomaly Method':<18} | {'Mean ' + metric:<14} | {'Std':<8} | {'Time (s)':<9} | {'t vs none':<10} | {'p-value':<8}")
    print("-" * 80)
    for m in methods:
        sc = scores_by_method[m]
        if m == "none":
            t_str, p_str = "—", "—"
        else:
            t_val, p_val = corrected_paired_test(sc, baseline, n_train=n_train, n_test=n_test)
            t_str, p_str = f"{t_val:+.3f}", f"{p_val:.4f}"
        print(f"{m:<18} | {sc.mean():<14.4f} | {sc.std():<8.4f} | {times_by_method[m]:<9.2f} | {t_str:<10} | {p_str:<8}")


if __name__ == "__main__":
    main()