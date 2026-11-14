"""Evaluation utilities: classification report, multi-metric scorers, and fairness audit."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    roc_auc_score,
)


def build_scorers(metric_names: list) -> dict:
    """Constrói um dicionário de scorers compatíveis com o scikit-learn."""
    available = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f_beta": make_scorer(fbeta_score, beta=2, zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    return {m: available[m] for m in metric_names if m in available}


def oof_classification_report(y_true, y_pred, y_prob=None) -> str:
    """Gera matriz de confusão, classification report e métricas ROC-AUC / PR-AUC."""
    cm = confusion_matrix(y_true, y_pred)
    lines = [
        "=== Classification Report & Confusion Matrix ===",
        "Confusion Matrix (Predicted 0 | Predicted 1):",
        f"Actual 0:  {cm[0, 0]:<5d} | {cm[0, 1]:<5d}",
        f"Actual 1:  {cm[1, 0]:<5d} | {cm[1, 1]:<5d}",
        "",
        classification_report(y_true, y_pred, zero_division=0),
    ]

    if y_prob is not None:
        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
        lines.append(f"ROC-AUC: {roc:.3f}")
        lines.append(f"PR-AUC:  {pr:.3f}")

    text = "\n".join(lines)
    print(text)
    return text


def fairness_report(y_test, y_pred, extras_test: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """Calcula a Taxa de Falsos Positivos por grupo demográfico (funciona com qualquer dataset)."""
    df = extras_test.copy()
    df["y_true"] = y_test.values
    df["y_pred_model"] = y_pred

    models_to_audit = [("Our model", "y_pred_model")]
    if "score_text" in df.columns:
        df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)
        models_to_audit.append(("COMPAS's own score", "y_pred_compas"))

    lines = [
        "=== Fairness Report ===",
        f"False positive rate by {sensitive_attr}",
        "(share of negative cases that were predicted positive)",
        "",
    ]

    for label, col in models_to_audit:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {str(group):<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text