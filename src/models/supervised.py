"""
Denetimli Öğrenme Modelleri
============================
Random Forest, XGBoost, SVM ve MLP tabanlı anomali sınıflandırma.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


class SupervisedAnomalyDetector:
    """Denetimli öğrenme ile anomali tespiti."""

    AVAILABLE_MODELS = {
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "mlp": MLPClassifier,
    }

    DEFAULT_PARAMS = {
        "random_forest": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1},
        "svm": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (128, 64, 32), "max_iter": 500, "random_state": 42, "early_stopping": True},
    }

    def __init__(self, model_type: str = "random_forest", params: Optional[Dict[str, Any]] = None):
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Geçersiz model: {model_type}. Seçenekler: {list(self.AVAILABLE_MODELS.keys())}")
        self.model_type = model_type
        model_params = self.DEFAULT_PARAMS[model_type].copy()
        if params:
            model_params.update(params)
        self.model = self.AVAILABLE_MODELS[model_type](**model_params)
        self.is_fitted = False

    def fit(self, X, y):
        """Modeli eğitir."""
        self.model.fit(X, y)
        self.is_fitted = True
        print(f"✅ {self.model_type} modeli eğitildi.")
        return self

    def predict(self, X):
        """Tahmin yapar."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmedi.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Olasılık tahmini yapar."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmedi.")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise AttributeError(f"{self.model_type} olasılık tahmini desteklemiyor.")

    def get_feature_importance(self, feature_names=None):
        """Özellik önem skorlarını döndürür (destekleyen modeller için)."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            if feature_names is not None:
                return pd.Series(importances, index=feature_names).sort_values(ascending=False)
            return importances
        return None
