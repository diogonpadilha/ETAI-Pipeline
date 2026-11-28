"""Model construction and feature selection."""
import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.random_projection import SparseRandomProjection
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
import inspect
from src.preprocessing import AnomalyScore

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class VoteSelector(BaseEstimator, SelectorMixin):
    """
    Seletor de features baseado no voto cumulativo de múltiplos critérios estatísticos.
    """
    def __init__(self, threshold=2):
        self.threshold = threshold

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if hasattr(X, "shape"):
            self.n_features_in_ = X.shape[1]

        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        votes = np.zeros(n_features, dtype=int)

        # Voto 1: Variância superior a zero
        variances = np.var(X_arr, axis=0)
        votes += (variances > 1e-5).astype(int)

        # Voto 2: Desvio padrão
        std_devs = np.std(X_arr, axis=0)
        votes += (std_devs > 1e-3).astype(int)

        if y is not None:
            y_arr = np.asarray(y)
            # Voto 3: Correlação absoluta com o target
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for i in range(n_features):
                    if std_devs[i] > 1e-5:
                        corr = np.corrcoef(X_arr[:, i], y_arr)[0, 1]
                        if not np.isnan(corr) and abs(corr) > 0.01:
                            votes[i] += 1

        self.votes_ = votes
        return self

    def _get_support_mask(self):
        return np.asarray(self.votes_) >= self.threshold


def adaboost(**kwargs):
    """Constrói um classificador AdaBoost."""
    return AdaBoostClassifier(**kwargs)


def oblique_random_forest(
    n_estimators: int = 100,
    n_components: int = 10,
    density="auto",
    random_state=None,
    n_jobs=None,
    **tree_params
):
    """
    A forest of oblique trees using SparseRandomProjection + DecisionTreeClassifier.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(
        estimator=oblique_tree,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )


_MODELS = {
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "naive_bayes": GaussianNB,
    "dummy": DummyClassifier,
    "adaboost": adaboost,
    "oblique_random_forest": oblique_random_forest,
}

if XGBClassifier is not None:
    _MODELS["xgboost"] = XGBClassifier


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = (model_config.get("params") or {}).copy()

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    estimator_cls = _MODELS[model_type]
    sig = inspect.signature(estimator_cls)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not accepts_kwargs:
        params = {k: v for k, v in params.items() if k in sig.parameters}

    return estimator_cls(**params)


def build_pipeline(
    preprocessor,
    model_config: dict,
    fs_config: dict = None,
    anomaly_config: dict = None,
    memory=None,
) -> Pipeline:
    """
    Constrói a pipeline completa: preprocessor -> [anomaly] -> [selector] -> model.
    """
    steps = [("preprocessor", preprocessor)]

    if anomaly_config and anomaly_config.get("method", "none") != "none":
        steps.append((
            "anomaly",
            AnomalyScore(
                method=anomaly_config["method"],
                n_neighbors=anomaly_config.get("n_neighbors", 20),
                random_state=anomaly_config.get("random_state"),
            ),
        ))

    if fs_config and fs_config.get("enabled", False):
        # Mantém o teu VoteSelector exatamente como estava configurado
        from src.features import VoteSelector
        steps.append((
            "selector",
            VoteSelector(**fs_config.get("params", {})),
        ))

    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)