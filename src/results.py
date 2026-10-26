"""Saving each run's results to disk and maintaining a global scoreboard."""
import os
import csv
from datetime import datetime

def save_scores(scoreboard_path: str, model_name: str, selection_metric: str, cv_score: float, params: dict):
    """
    Acrescenta uma linha ao scoreboard.csv com o resultado do modelo e hiperparâmetros.
    """
    os.makedirs(os.path.dirname(scoreboard_path), exist_ok=True)
    
    file_exists = os.path.isfile(scoreboard_path)
    
    with open(scoreboard_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Selection_Metric", "Score", "Params"])
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, model_name, selection_metric, f"{cv_score:.4f}", str(params)])

def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Escreve o relatório completo num ficheiro de texto individual[cite: 5].
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    # Usar get() para aceder a 'test_size' e 'random_state', prevenindo erros caso as chaves não existam no yaml antigo
    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", "N/A")
    random_state = split_cfg.get("random_state", "N/A")

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model']['params']}\n"
        f"Test size: {test_size}  random_state: {random_state}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path