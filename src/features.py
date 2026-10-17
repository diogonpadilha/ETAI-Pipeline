"""Feature engineering transformations for COMPAS dataset."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class COMPASFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador de Feature Engineering para o dataset COMPAS.
    Gera novas variáveis com base no histórico criminal juvenil, histórico adulto,
    rácios de idade e transformações de escala.
    """

    def __init__(
        self,
        add_aggregates: bool = True,
        add_ratios: bool = True,
        add_flags: bool = True,
        add_log_transforms: bool = True,
    ):
        self.add_aggregates = add_aggregates
        self.add_ratios = add_ratios
        self.add_flags = add_flags
        self.add_log_transforms = add_log_transforms

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # 1. Agregações: Total de antecedentes juvenis e total global
        if self.add_aggregates:
            juv_cols = [
                c for c in ["juv_fel_count", "juv_misd_count", "juv_other_count"] if c in X_out.columns
            ]
            if juv_cols:
                X_out["total_juv_count"] = X_out[juv_cols].sum(axis=1)

            if "priors_count" in X_out.columns and "total_juv_count" in X_out.columns:
                X_out["total_prior_records"] = X_out["priors_count"] + X_out["total_juv_count"]

        # 2. Rácios: Densidade de antecedentes por ano de vida adulta
        if self.add_ratios and "age" in X_out.columns and "priors_count" in X_out.columns:
            adult_years = (X_out["age"] - 18).clip(lower=1)
            X_out["priors_per_adult_year"] = X_out["priors_count"] / adult_years

        # 3. Indicadores Binários (Flags)
        if self.add_flags:
            if "priors_count" in X_out.columns:
                X_out["has_priors"] = (X_out["priors_count"] > 0).astype(int)
            if "total_juv_count" in X_out.columns:
                X_out["has_juv_record"] = (X_out["total_juv_count"] > 0).astype(int)

        # 4. Transformação Logarítmica
        if self.add_log_transforms:
            if "priors_count" in X_out.columns:
                X_out["log_priors_count"] = np.log1p(X_out["priors_count"].clip(lower=0))

        return X_out