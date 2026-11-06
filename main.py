"""
Entry point for the predictive pipeline with feature selection and XGBoost.
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
    
    # A2: Passar as definições de feature selection para o construtor da pipeline
    fs_config = config.get("feature_selection", {})
    base_pipeline = build_pipeline(preprocessor, config["model"], fs_config)

    tuning_report = ""
    cv_score = 0.0
    best_params = config["model"].get("params", {}).copy()
    
    if config.get("tuning", {}).get("enabled", False):
        # A2: Juntar o espaço de pesquisa do seletor com o do modelo durante o tuning
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
        pipeline.fit(X_dev, y_dev)
        y_dev_pred = pipeline.predict(X_dev)
        y_dev_prob = pipeline.predict_proba(X_dev)[:, 1] if hasattr(pipeline, "predict_proba") else None
        
        cv_cfg = config.get("cv", {})
        cv_splitter = StratifiedKFold(n_splits=cv_cfg.get("n_splits", 5), shuffle=True, random_state=cv_cfg.get("random_state", 42))
        cv_score = cross_val_score(pipeline, X_dev, y_dev, cv=cv_splitter, scoring=cv_cfg.get("selection_metric", "accuracy")).mean()

    # A2: Imprimir e anexar ao relatório o número/lista de features mantidas
    features_text = ""
    if "selector" in pipeline.named_steps:
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            mask = pipeline.named_steps["selector"].get_support()
            features_kept = np.array(feature_names)[mask]
            features_text = f"\nFeatures kept ({len(features_kept)}/{len(feature_names)}): {list(features_kept)}\n"
        except Exception:
            mask = pipeline.named_steps["selector"].get_support()
            features_text = f"\nFeatures kept: {mask.sum()} out of {len(mask)}\n"
            
        print(features_text)

    # Previsões no conjunto de teste
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    # Gerar os relatórios
    report = tuning_report + features_text + oof_classification_report(y_test, y_test_pred, y_test_prob)
    report += "\n\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # Guardar o relatório completo
    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"\nRelatório completo gravado em: {path}")

    # Guardar no scoreboard.csv (A3)
    scoreboard_path = config.get("output", {}).get("scoreboard", "results/scoreboard.csv")
    selection_metric = config.get("cv", {}).get("selection_metric", "accuracy")
    
    # Identificar se a run usou seleção de features para diferenciação no scoreboard
    model_name = config["model"]["type"]
    if fs_config.get("enabled", False):
        model_name += "_with_fs"

    save_scores(scoreboard_path, model_name, selection_metric, cv_score, best_params)
    print(f"Resultado guardado no scoreboard central: {scoreboard_path}")


if __name__ == "__main__":
    main()