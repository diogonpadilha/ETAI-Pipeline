"""
Entry point for the predictive pipeline (supports multiple datasets and model serialization).
"""
import os
import sys
import joblib
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_data
from src.preprocessing import preprocess, prepare_inference_features
from src.model import build_pipeline
from src.tuning import tune_and_evaluate
from src.evaluate import oof_classification_report, fairness_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)
    df = load_data(config["data"]["path"])

    imputer_name = config.get("preprocessing", {}).get("imputation", {}).get("numeric_strategy", "median")
    print(f"Running pipeline | config={config_path} | model={config['model']['type']} | imputer={imputer_name}")

    X_dev, X_test, y_dev, y_test, extras_test, preprocessor = preprocess(df, config=config)

    fs_config = config.get("feature_selection", {})
    base_pipeline = build_pipeline(preprocessor, config["model"], fs_config)

    tuning_report = ""
    cv_score = 0.0
    best_params = (config["model"].get("params") or {}).copy()

    if config.get("tuning", {}).get("enabled", False):
        if fs_config.get("enabled", False) and "search_space" in fs_config:
            model_type = config["model"]["type"]
            if model_type in config["tuning"]["search_spaces"]:
                config["tuning"]["search_spaces"][model_type].update(fs_config["search_space"])

        pipeline, y_dev_pred, y_dev_prob, tuning_report, cv_score, best_params = tune_and_evaluate(
            base_pipeline, X_dev, y_dev, config
        )
        print(tuning_report)
    else:
        pipeline = base_pipeline
        cv_cfg = config.get("cv", {})
        cv_splitter = StratifiedKFold(
            n_splits=cv_cfg.get("n_splits", 5),
            shuffle=True,
            random_state=cv_cfg.get("random_state", 42),
        )
        sel_metric = cv_cfg.get("selection_metric", "roc_auc")
        cv_res = cross_validate(pipeline, X_dev, y_dev, cv=cv_splitter, scoring=["roc_auc", "accuracy", "f1"])
        cv_score = float(cv_res[f"test_{sel_metric}"].mean())
        tuning_report = (
            "=== Cross-Validation (Default / No Tuning) ===\n"
            f"CV ROC-AUC:  {cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}\n"
            f"CV Accuracy: {cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}\n"
            f"CV F1:       {cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}\n"
            "==============================================\n\n"
        )
        print(tuning_report)
        pipeline.fit(X_dev, y_dev)

    # Registar features mantidas pelo seletor
    features_text = ""
    if "selector" in pipeline.named_steps:
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            mask = pipeline.named_steps["selector"].get_support()
            features_kept = np.array(feature_names)[mask]
            features_dropped = np.array(feature_names)[~mask]
            features_text = (
                f"\nFeatures kept ({len(features_kept)}/{len(feature_names)}): {list(features_kept)}\n"
                f"Features dropped ({len(features_dropped)}): {list(features_dropped)}\n"
            )
        except Exception:
            mask = pipeline.named_steps["selector"].get_support()
            features_text = f"\nFeatures kept: {mask.sum()} out of {len(mask)}\n"
        print(features_text)

    # Avaliação no conjunto de teste
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    report = tuning_report + features_text + oof_classification_report(y_test, y_test_pred, y_test_prob)
    report += "\n\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"\nRelatório completo gravado em: {path}")

    scoreboard_path = config.get("output", {}).get("scoreboard", "results/scoreboard.csv")
    selection_metric = config.get("cv", {}).get("selection_metric", "roc_auc")
    model_name = config["model"]["type"] + ("_with_fs" if fs_config.get("enabled", False) else "")
    save_scores(scoreboard_path, model_name, selection_metric, cv_score, best_params, imputer=imputer_name)
    print(f"Resultado guardado no scoreboard central: {scoreboard_path}")

    # Guardar a pipeline final ajustada para inferência (predict.py)
    model_path = config.get("output", {}).get("model_path", "models/model.joblib")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    pipeline.set_params(memory=None)
    joblib.dump(pipeline, model_path)
    print(f"Modelo final guardado em: {model_path}")

    # Previsão em dados novos se configurado (ex: adult_new.csv)
    new_data_path = config["data"].get("new_data_path")
    predictions_output = config["output"].get("predictions_path")
    if new_data_path and predictions_output:
        df_new = load_data(new_data_path)
        X_new, ids = prepare_inference_features(df_new, config)
        new_probs = pipeline.predict_proba(X_new)[:, 1] if hasattr(pipeline, "predict_proba") else np.zeros(len(X_new))
        new_preds = (new_probs >= 0.5).astype(int) if hasattr(pipeline, "predict_proba") else pipeline.predict(X_new)

        out_dir = os.path.dirname(predictions_output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df = pd.DataFrame({
            "id": ids,
            "probability": np.round(new_probs, 6),
            "prediction": new_preds.astype(int),
        })
        out_df.to_csv(predictions_output, index=False)
        print(f"Previsões para {len(out_df)} linhas gravadas em: {predictions_output}")


if __name__ == "__main__":
    main()