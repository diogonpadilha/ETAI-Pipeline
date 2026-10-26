"""Hyperparameter tuning with Optuna and Nested Cross-Validation."""
import optuna
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

def _parse_search_space(trial, space_config):
    """Lê o YAML e pede ao Optuna os parâmetros dinamicamente[cite: 6]."""
    params = {}
    for param_name, rules in space_config.items():
        p_type = rules.get("type")
        if p_type == "int":
            params[param_name] = trial.suggest_int(param_name, rules["low"], rules["high"])
        elif p_type == "float":
            params[param_name] = trial.suggest_float(param_name, rules["low"], rules["high"], log=rules.get("log", False))
        elif p_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, rules["choices"])
    return params

def tune_and_evaluate(pipeline, X_dev, y_dev, config):
    """Executa Nested CV para estimativa honesta e devolve a pipeline final afinada[cite: 6]."""
    tune_cfg = config["tuning"]
    cv_cfg = config.get("cv", {"n_splits": 5, "random_state": 42, "selection_metric": "roc_auc"})
    
    model_type = config["model"]["type"]
    search_space = tune_cfg["search_spaces"].get(model_type, {})
    
    n_splits = cv_cfg["n_splits"]
    seed = cv_cfg["random_state"]
    scoring = cv_cfg["selection_metric"]
    n_trials = tune_cfg["n_trials"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    print(f"A executar Nested CV (Estimativa Honesta a maximizar {scoring})...")
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_oof_pred = np.empty(len(y_dev), dtype=y_dev.dtype)
    y_oof_prob = np.empty(len(y_dev), dtype=float)
    
    X = X_dev.reset_index(drop=True)
    y = y_dev.reset_index(drop=True)
    inner_scores = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        def inner_obj(trial):
            pipe = clone(pipeline).set_params(**_parse_search_space(trial, search_space))
            inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            return cross_val_score(pipe, X_tr, y_tr, cv=inner_cv, scoring=scoring).mean()
            
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed + fold))
        study.optimize(inner_obj, n_trials=n_trials)
        inner_scores.append(study.best_value)
        
        best_fold_pipe = clone(pipeline).set_params(**study.best_params)
        best_fold_pipe.fit(X_tr, y_tr)
        
        y_oof_pred[val_idx] = best_fold_pipe.predict(X_val)
        if hasattr(best_fold_pipe, "predict_proba"):
            y_oof_prob[val_idx] = best_fold_pipe.predict_proba(X_val)[:, 1]
        
    # Recalcular score no outer loop de acordo com a métrica selecionada
    if scoring == "roc_auc" and hasattr(pipeline, "predict_proba"):
        from sklearn.metrics import roc_auc_score
        nested_score = roc_auc_score(y, y_oof_prob)
    else:
        from sklearn.metrics import accuracy_score
        nested_score = accuracy_score(y, y_oof_pred)
        
    mean_inner = np.mean(inner_scores)

    print("A afinar a pipeline final em todo o conjunto de desenvolvimento...")
    def final_obj(trial):
        pipe = clone(pipeline).set_params(**_parse_search_space(trial, search_space))
        cv_final = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return cross_val_score(pipe, X, y, cv=cv_final, scoring=scoring).mean()

    final_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    final_study.optimize(final_obj, n_trials=n_trials)
    
    final_pipe = clone(pipeline).set_params(**final_study.best_params)
    final_pipe.fit(X, y)
    
    report = (
        "=== Hyperparameter Tuning (Optuna) ===\n"
        f"Melhores parâmetros: {final_study.best_params}\n"
        f"Score Otimista (Inner best médio - {scoring}): {mean_inner:.3f}\n"
        f"Score Honesto (Nested CV - {scoring}):         {nested_score:.3f}\n"
        f"Viés de Otimismo:                              {mean_inner - nested_score:+.3f}\n"
        "======================================\n\n"
    )
    
    return final_pipe, y_oof_pred, y_oof_prob, report, nested_score, final_study.best_params