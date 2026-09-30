"""
Entry point for the baseline predictive pipeline.
"""
import yaml
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_data
from src.preprocessing import preprocess
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])

    # Receber o X_dev puro e o preprocessor
    X_dev, X_test, y_dev, y_test, extras_test, preprocessor = preprocess(
        df, 
        config=config
    )

    # Construir o modelo estático
    model = build_model(config["model"])
    
    # Criar o Pipeline (liga as transformações diretamente ao modelo)
    pipeline = make_pipeline(preprocessor, model)

    # ---------------------------------------------------------
    # 1. Stratified 5-Fold Cross-Validation (Apenas no conjunto Dev)
    # ---------------------------------------------------------
    print("A executar Stratified 5-Fold Cross-Validation...")
    skf = StratifiedKFold(
        n_splits=5, 
        shuffle=True, 
        random_state=config["split"]["random_state"]
    )

    cv_results = cross_validate(
        pipeline,
        X_dev,
        y_dev,
        cv=skf,
        scoring=["accuracy"], 
        return_train_score=True
    )

    train_acc_mean = np.mean(cv_results['train_accuracy'])
    train_acc_std = np.std(cv_results['train_accuracy'])
    val_acc_mean = np.mean(cv_results['test_accuracy'])
    val_acc_std = np.std(cv_results['test_accuracy'])

    cv_report = (
        "=== Stratified 5-Fold Cross-Validation (Dev Set) ===\n"
        f"Train Accuracy: {train_acc_mean:.3f} (± {train_acc_std:.3f})\n"
        f"Val Accuracy:   {val_acc_mean:.3f} (± {val_acc_std:.3f})\n"
        f"Gap (Train - Val): {train_acc_mean - val_acc_mean:+.3f}\n"
        "============================================================\n\n"
    )
    
    print(cv_report)

    # ---------------------------------------------------------
    # 2. Treino Final e Avaliação no Test Set (Holdout)
    # ---------------------------------------------------------
    # Treinar o pipeline final com TODOS os dados de desenvolvimento
    pipeline.fit(X_dev, y_dev)

    # Fazer as previsões usando a pipeline completa
    y_dev_pred = pipeline.predict(X_dev)
    y_test_pred = pipeline.predict(X_test)

    # Avaliar (usando dev vs test)
    report = cv_report + evaluate(y_dev, y_dev_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()