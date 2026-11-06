"""Model construction and feature selection pipeline."""
import tempfile
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
    BaggingClassifier
)
from sklearn.random_projection import SparseRandomProjection
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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
    """A forest of oblique trees."""
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


class VoteSelector(BaseEstimator, SelectorMixin):
    """
    Seletor de features baseado no voto cumulativo de múltiplos métodos de seleção.
    """
    def __init__(self, threshold=1):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        n_features = X_arr.shape[1]
        
        # Exemplo de lógica de voto: variância não-nula e correlação com a classe target
        votes = np.zeros(n_features, dtype=int)
        
        # Voto 1: Variância superior a zero (elimina colunas constantes)
        variances = np.var(X_arr, axis=0)
        votes += (variances > 1e-5).astype(int)
        
        # Voto 2: Desvio padrão / presença de sinal
        std_devs = np.std(X_arr, axis=0)
        votes += (std_devs > 1e-3).astype(int)
        
        if y is not None:
            y_arr = np.asarray(y)
            # Voto 3: Correlação com a variável target (se y for numérico/binário)
            for i in range(n_features):
                if std_devs[i] > 1e-5:
                    corr = np.corrcoef(X_arr[:, i], y_arr)[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.01:
                        votes[i] += 1

        self.votes_ = votes
        return self

    def _get_support_mask(self):
        return np.asarray(self.votes_) >= self.threshold


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
    "xgboost": XGBClassifier,
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params", {}).copy()

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessor, model_config: dict, feature_selection_config: dict = None):
    """
    Constrói a pipeline com cache temporária, seleção de features opcional (VoteSelector) e modelo final.
    """
    model = build_model(model_config)
    cachedir = tempfile.mkdtemp()
    
    steps = [("preprocessor", preprocessor)]
    
    if feature_selection_config and feature_selection_config.get("enabled", False):
        params = feature_selection_config.get("params", {})
        steps.append(("selector", VoteSelector(**params)))
        
    steps.append(("model", model))
    
    return Pipeline(steps, memory=cachedir)