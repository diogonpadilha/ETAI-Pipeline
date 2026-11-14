"""
Saving each run's results and centralized scoreboard tracking.
"""
import csv
import json
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside `results_dir`.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model']['params']}\n"
        f"Test size: {config['split']['test_size']}  "
        f"random_state: {config['split']['random_state']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path


def save_scores(scoreboard_path: str, model_name: str, metric: str, score: float, params: dict) -> None:
    """
    Appends one model evaluation row to the centralized CSV scoreboard.
    """
    out_dir = os.path.dirname(scoreboard_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    file_exists = os.path.exists(scoreboard_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(scoreboard_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "metric", "cv_score", "params"])
        writer.writerow([timestamp, model_name, metric, f"{score:.4f}", json.dumps(params)])