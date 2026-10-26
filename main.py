"""
Entry point for the baseline predictive pipeline.
"""
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import load_data
from src.preprocessing import preprocess
from src.model import build_pipeline
from src.tuning import tune_and_evaluate
from src.evaluate import oof_classification_report, fairness_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    df = load_data(config["data"]["path"])

    X_dev, X_test, y_dev, y_test, extras_test, preprocessor = preprocess(df, config=config)
    base_pipeline = build_pipeline(preprocessor, config["model"])

    tuning_report = ""
    cv_score = 0.0
    best_params = config["model"].get("params", {})
    
    if config.get("tuning", {}).get("enabled", False):
        pipeline, y_dev_pred, y_dev_prob, tuning_report, cv_score, best_params = tune_and_evaluate(
            base_pipeline, X_dev, y_dev, config
        )
        print(tuning_report)
    else:
        pipeline = base_pipeline
        pipeline.fit(X_dev, y_dev)
        y_dev_pred = pipeline.predict(X_dev)
        y_dev_prob = pipeline.predict_proba(X_dev)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        cv_cfg = config.get("cv", {})
        cv_splitter = StratifiedKFold(n_splits=cv_cfg.get("n_splits", 5), shuffle=True, random_state=cv_cfg.get("random_state", 42))
        cv_score = cross_val_score(pipeline, X_dev, y_dev, cv=cv_splitter, scoring=cv_cfg.get("selection_metric", "accuracy")).mean()

    # Previsões no conjunto de teste
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Gerar os relatórios com as novas métricas e probabilidades
    report = tuning_report + oof_classification_report(y_test, y_test_pred, y_test_prob)
    report += "\n\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # Guardar os resultados no ficheiro individual de teste
    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"\nRelatório completo gravado em: {path}")

    # Guardar a pontuação centralizada no scoreboard.csv
    scoreboard_path = config.get("output", {}).get("scoreboard", "results/scoreboard.csv")
    selection_metric = config.get("cv", {}).get("selection_metric", "accuracy")
    save_scores(scoreboard_path, config["model"]["type"], selection_metric, cv_score, best_params)
    print(f"Resultado guardado no scoreboard central: {scoreboard_path}")

if __name__ == "__main__":
    main()