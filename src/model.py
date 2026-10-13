"""Model construction."""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline # Usar Pipeline explícito

_MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "dummy": DummyClassifier,
}

def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {})

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)

def build_pipeline(preprocessor, model_config: dict):
    """Constrói a pipeline com o nome estático 'model' para facilitar o tuning."""
    model = build_model(model_config)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])