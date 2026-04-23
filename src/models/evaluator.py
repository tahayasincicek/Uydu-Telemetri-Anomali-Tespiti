"""
Model Değerlendirme Modülü
============================
Model performansını ölçmek ve karşılaştırmak için metrikler ve görselleştirmeler.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)


class ModelEvaluator:
    """Model performansını değerlendiren ve raporlayan sınıf."""

    def __init__(self):
        self.results: Dict[str, dict] = {}

    def evaluate(self, y_true, y_pred, model_name: str, y_proba=None) -> dict:
        """
        Model performansını hesaplar.

        Args:
            y_true: Gerçek etiketler.
            y_pred: Tahmin edilen etiketler.
            model_name: Model adı.
            y_proba: Olasılık tahminleri (ROC-AUC için).

        Returns:
            dict: Performans metrikleri.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        if y_proba is not None:
            try:
                if y_proba.ndim == 2:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = None

        self.results[model_name] = metrics
        print(f"📊 {model_name} Sonuçları:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        if "roc_auc" in metrics and metrics["roc_auc"]:
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")

        return metrics

    def compare_models(self) -> pd.DataFrame:
        """Tüm modellerin performanslarını karşılaştırır."""
        if not self.results:
            print("⚠️  Henüz değerlendirilmiş model yok.")
            return pd.DataFrame()

        rows = []
        for name, m in self.results.items():
            rows.append({
                "Model": name,
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1-Score": m.get("f1_score"),
                "ROC-AUC": m.get("roc_auc"),
            })

        df = pd.DataFrame(rows).set_index("Model")
        return df.sort_values("F1-Score", ascending=False)

    def get_best_model(self, metric: str = "f1_score") -> str:
        """Belirtilen metriğe göre en iyi modeli döndürür."""
        if not self.results:
            raise ValueError("Henüz değerlendirilmiş model yok.")
        best = max(self.results.items(), key=lambda x: x[1].get(metric, 0))
        print(f"🏆 En iyi model ({metric}): {best[0]} = {best[1][metric]:.4f}")
        return best[0]
