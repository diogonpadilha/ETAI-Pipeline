"""Evaluation -- multi-metric evaluation, out-of-fold predictions, and fairness."""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

def build_scorers(metrics: list) -> dict:
    """Mapeia os nomes das métricas do config.yaml para as funções do scikit-learn."""
    available = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": lambda y, p: precision_score(y, p, zero_division=0),
        "recall": lambda y, p: recall_score(y, p, zero_division=0),
        "f1": lambda y, p: f1_score(y, p, zero_division=0),
        "f_beta": lambda y, p: fbeta_score(y, p, beta=2.0, zero_division=0), # Penaliza mais os falsos negativos
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score
    }
    return {m: available[m] for m in metrics if m in available}

def oof_classification_report(y_true, y_pred, y_prob=None) -> str:
    """Gera o relatório de classificação e a matriz de confusão."""
    lines = ["=== Classification Report & Confusion Matrix ==="]
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    lines.append("Confusion Matrix (Predicted 0 | Predicted 1):")
    lines.append(f"Actual 0:  {cm[0, 0]:<5d} | {cm[0, 1]:<5d}")
    lines.append(f"Actual 1:  {cm[1, 0]:<5d} | {cm[1, 1]:<5d}")
    lines.append("")
    
    # Relatório padrão
    lines.append(classification_report(y_true, y_pred, zero_division=0))
    
    # Métricas de probabilidade se fornecidas
    if y_prob is not None:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        lines.append(f"ROC-AUC: {roc_auc:.3f}")
        lines.append(f"PR-AUC:  {pr_auc:.3f}")
        
    return "\n".join(lines)

def fairness_report(y_test, y_pred, extras_test: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """Calcula a Taxa de Falsos Positivos por grupo demográfico[cite: 4]."""
    df = extras_test.copy()
    df["y_true"] = y_test.values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "=== Fairness Report ===",
        "False positive rate by race",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    return "\n".join(lines)