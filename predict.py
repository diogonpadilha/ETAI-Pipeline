"""
Inference CLI: loads a saved pipeline (.joblib) and predicts a new CSV without retraining.
"""
import argparse
import joblib
import pandas as pd
import yaml

from src.preprocessing import prepare_inference_features


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """
    Returns id, probability of the positive class, and binary prediction (at 0.5)
    for every input row without dropping duplicates.
    """
    X, ids = prepare_inference_features(df, config)
    probability = pipeline.predict_proba(X)[:, 1]
    return pd.DataFrame({
        "id": ids,
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="path to YAML config file (default: config.yaml)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = joblib.load(config["output"]["model_path"])
    df_input = pd.read_csv(args.input)
    predictions = predict(df_input, pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")