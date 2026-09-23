"""
Preprocessing -- raw data in, model-ready train/test split out, driven by config.yaml.
"""
import logging
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder, # Adicionado para suportar o target encoder do YAML
)

logger = logging.getLogger(__name__)

COMPAS_OUTPUT_COLUMNS = ["decile_score", "score_text"]

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------------
# 1. Cleaning (target-agnostic, split-independent)
# ---------------------------------------------------------------------------

def _drop_duplicates(df: pd.DataFrame, id_column: str = "id") -> pd.DataFrame:
    n_start = len(df)
    df = df.drop_duplicates()
    n_exact = n_start - len(df)
    n_id = 0
    if id_column in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=id_column, keep="first")
        n_id = n_before - len(df)
    logger.info("Duplicates dropped: %d exact rows, %d extra repeated ids", n_exact, n_id)
    return df

def _placeholders_to_nan(df: pd.DataFrame, tokens) -> pd.DataFrame:
    tokens = set(tokens)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]) or pd.api.types.is_bool_dtype(out[col]):
            continue
        stripped = out[col].str.strip()
        out[col] = stripped.mask(stripped.isin(tokens))
    return out

def _canonicalize_categories(df: pd.DataFrame, canonical_maps: dict) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in canonical_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].str.strip()
        mapped = cleaned.str.lower().map(mapping)
        out[col] = mapped.where(mapped.notna(), cleaned)
    return out

def _apply_domain_rules(df: pd.DataFrame, validity_rules: dict) -> pd.DataFrame:
    """Adaptado para ler dicionários min/max do YAML."""
    out = df.copy()
    for col, rule in validity_rules.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        bad = pd.Series(False, index=values.index)
        
        if "min" in rule:
            bad |= values < rule["min"]
        if "max" in rule:
            bad |= values > rule["max"]
            
        logger.info("Domain rule '%s': %d violations -> NaN", rule, int(bad.sum()))
        out[col] = values.mask(bad)
    return out


def clean_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    diag_cfg = config["diagnostics"]
    
    out = _drop_duplicates(df, diag_cfg.get("id_column", "id"))
    out = _placeholders_to_nan(out, diag_cfg["placeholder_tokens"])
    out = _canonicalize_categories(out, diag_cfg["canonical_categories"])
    out = _apply_domain_rules(out, diag_cfg["validity_rules"])
    
    # Eliminar APENAS as redundâncias do EDA nesta fase.
    # A lista 'drop_columns' será tratada mais à frente na separação das features.
    cols_to_drop = diag_cfg.get("redundant_columns", [])
    return out.drop(columns=[c for c in cols_to_drop if c in out.columns])

# ---------------------------------------------------------------------------
# 2. Indicators and X / y / extras
# ---------------------------------------------------------------------------

def add_missingness_indicators(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out

def split_features_target(df, target, sensitive_attr, drop_columns, indicator_sources):
    df = add_missingness_indicators(df, [c for c in indicator_sources if c != sensitive_attr])
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr] + COMPAS_OUTPUT_COLUMNS if c in df.columns]
    extras = df[extras_cols].copy()

    always_drop = {target, sensitive_attr, *COMPAS_OUTPUT_COLUMNS, *drop_columns}
    X = df[[c for c in df.columns if c not in always_drop]]
    return X, y, extras

def split_train_test(X, y, extras, test_size: float, random_state: int):
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)

# ---------------------------------------------------------------------------
# 3. Leak-safe preprocessor
# ---------------------------------------------------------------------------
_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "target": lambda: TargetEncoder(), # Adicionado target encoder
}

def build_preprocessor(numeric_features, categorical_features, indicator_features,
                       config: dict) -> ColumnTransformer:
    prep_cfg = config["preprocessing"]
    
    scaler_factory = _SCALERS[prep_cfg.get("scaler", "none")]
    encoder_factory = _ENCODERS[prep_cfg.get("encoder", "onehot")]
    
    num_strategy = prep_cfg["imputation"].get("numeric_strategy", "median")
    cat_strategy = prep_cfg["imputation"].get("categorical_strategy", "most_frequent")

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=num_strategy)),
        ("scale", scaler_factory() if callable(scaler_factory) else scaler_factory),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy=cat_strategy)),
        ("encode", encoder_factory()),
    ])
    transformer = ColumnTransformer(
        [
            ("numeric", numeric, list(numeric_features)),
            ("categorical", categorical, list(categorical_features)),
            ("indicators", "passthrough", list(indicator_features)),
        ],
        verbose_feature_names_out=False,
    )
    return transformer.set_output(transform="pandas")

# ---------------------------------------------------------------------------
# 4. Everything together
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    config: dict = None,
    return_preprocessor: bool = False,
):
    if config is None:
        config = load_config()

    target = config["data"]["target"]
    sensitive_attr = config["data"]["sensitive_attr"]
    
    # Agregar colunas a eliminar
    drop_columns = config["data"].get("drop_columns", []) + config["diagnostics"].get("redundant_columns", [])
    
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]

    df = clean_dataset(df, config)
    df = df.dropna(subset=[target]).reset_index(drop=True)

    indicator_sources = config["preprocessing"]["mnar_indicator_sources"]
    X, y, extras = split_features_target(df, target, sensitive_attr, drop_columns, indicator_sources)

    indicator_features = [f"{c}_was_missing" for c in indicator_sources if f"{c}_was_missing" in X.columns]
    feature_cols = [c for c in X.columns if c not in indicator_features]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X_train, X_test, y_train, y_test, extras_train, extras_test = split_train_test(
        X, y, extras, test_size, random_state
    )

    X_train = X_train.astype({c: object for c in categorical_features})
    X_test = X_test.astype({c: object for c in categorical_features})

    preprocessor = build_preprocessor(numeric_features, categorical_features, indicator_features, config)
    
    if config["preprocessing"]["encoder"] == "target":
        X_train_t = preprocessor.fit_transform(X_train, y_train)
    else:
        X_train_t = preprocessor.fit_transform(X_train)
        
    X_test_t = preprocessor.transform(X_test)

    if sensitive_attr in extras_train.columns and extras_train[sensitive_attr].notna().any():
        extras_test[sensitive_attr] = extras_test[sensitive_attr].fillna(extras_train[sensitive_attr].mode().iloc[0])

    outputs = (X_train_t, X_test_t, y_train, y_test, extras_test)
    return outputs + (preprocessor,) if return_preprocessor else outputs