"""
Denetimsiz Öğrenme Modelleri
==============================
Isolation Forest, One-Class SVM, DBSCAN ve Autoencoder tabanlı anomali tespiti.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN


class UnsupervisedAnomalyDetector:
    """Denetimsiz öğrenme ile anomali tespiti."""

    AVAILABLE_MODELS = {
        "isolation_forest": IsolationForest,
        "one_class_svm": OneClassSVM,
        "dbscan": DBSCAN,
    }

    DEFAULT_PARAMS = {
        "isolation_forest": {"n_estimators": 100, "contamination": 0.05, "random_state": 42, "n_jobs": -1},
        "one_class_svm": {"kernel": "rbf", "gamma": "scale", "nu": 0.05},
        "dbscan": {"eps": 0.5, "min_samples": 5},
    }

    def __init__(self, model_type: str = "isolation_forest", params: Optional[Dict[str, Any]] = None):
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Geçersiz model: {model_type}. Seçenekler: {list(self.AVAILABLE_MODELS.keys())}")
        self.model_type = model_type
        model_params = self.DEFAULT_PARAMS[model_type].copy()
        if params:
            model_params.update(params)
        self.model = self.AVAILABLE_MODELS[model_type](**model_params)
        self.is_fitted = False

    def fit(self, X):
        """Modeli eğitir."""
        self.model.fit(X)
        self.is_fitted = True
        print(f"✅ {self.model_type} modeli eğitildi.")
        return self

    def predict(self, X):
        """Anomali tahmini yapar. 1=normal, -1=anomali."""
        if not self.is_fitted:
            raise RuntimeError("Model henüz eğitilmedi.")
        if self.model_type == "dbscan":
            labels = self.model.fit_predict(X)
            return np.where(labels == -1, -1, 1)
        return self.model.predict(X)

    def get_anomaly_scores(self, X):
        """Anomali skorlarını döndürür (destekleyen modeller için)."""
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        if hasattr(self.model, "score_samples"):
            return self.model.score_samples(X)
        return None

    def fit_predict(self, X):
        """Fit ve predict işlemlerini birleştirir."""
        self.fit(X)
        return self.predict(X)
