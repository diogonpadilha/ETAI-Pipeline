"""
Entry point for the baseline predictive pipeline.
"""
import yaml
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import load_data
from src.preprocessing import preprocess
from src.model import build_pipeline
from src.tuning import tune_and_evaluate
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    df = load_data(config["data"]["path"])

    X_dev, X_test, y_dev, y_test, extras_test, preprocessor = preprocess(df, config=config)
    base_pipeline = build_pipeline(preprocessor, config["model"])

    tuning_report = ""
    
    if config.get("tuning", {}).get("enabled", False):
        pipeline, y_dev_pred, tuning_report = tune_and_evaluate(base_pipeline, X_dev, y_dev, config)
    else:
        pipeline = base_pipeline
        pipeline.fit(X_dev, y_dev)
        y_dev_pred = pipeline.predict(X_dev)

    y_test_pred = pipeline.predict(X_test)

    report = tuning_report + evaluate(y_dev, y_dev_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")

if __name__ == "__main__":
    main()