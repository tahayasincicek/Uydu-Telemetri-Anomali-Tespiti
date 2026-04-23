"""
Gözetimli Öğrenme Modelleri Modülü (Supervised Learning)
========================================================

Uydu telemetrisi (Reaction Wheels) anomali tespiti için gözetimli makine öğrenmesi
ve derin öğrenme (LSTM) modellerini eğitme, değerlendirme ve tahmin etme sınıfı.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    Sequential = None


class SupervisedAnomalyDetector:
    """
    Gözetimli öğrenme algoritmalarını (Random Forest, SVM, XGBoost, LSTM)
    kullanarak anomali tespiti yapan yönetici sınıf.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.best_model_name: Optional[str] = None
        self.metrics: Dict[str, Dict[str, float]] = {}

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series, tune: bool = False) -> RandomForestClassifier:
        """
        Random Forest modelini eğitir ve kaydeder.
        
        Args:
            X_train (pd.DataFrame): Eğitim özellikleri.
            y_train (pd.Series): Eğitim etiketleri.
            tune (bool): Hiperparametre optimizasyonu yapılıp yapılmayacağı.
            
        Returns:
            RandomForestClassifier: Eğitilmiş model.
        """
        print("🌲 Random Forest eğitiliyor...")
        if tune:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            rf = RandomForestClassifier(class_weight='balanced', random_state=self.random_state, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            grid = GridSearchCV(rf, param_grid, cv=tscv, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"En iyi parametreler: {grid.best_params_}")
        else:
            model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', 
                                         random_state=self.random_state, n_jobs=-1)
            model.fit(X_train, y_train)

        self.models['RandomForest'] = model
        return model

    def train_svm(self, X_train: pd.DataFrame, y_train: pd.Series, kernel: str = 'rbf') -> CalibratedClassifierCV:
        """
        Support Vector Machine modelini eğitir (Olasılık kalibrasyonu ile).
        
        Args:
            X_train (pd.DataFrame): Eğitim özellikleri.
            y_train (pd.Series): Eğitim etiketleri.
            kernel (str): SVM çekirdeği ('rbf', 'linear', 'poly').
            
        Returns:
            CalibratedClassifierCV: Olasılık çıktıları verebilen eğitilmiş SVM modeli.
        """
        print(f"⚔️ SVM ({kernel} kernel) eğitiliyor...")
        # SVC probability=True çok yavaştır, bu yüzden CalibratedClassifierCV kullanılır.
        base_svm = SVC(kernel=kernel, class_weight='balanced', random_state=self.random_state, max_iter=5000)
        model = CalibratedClassifierCV(base_svm, cv=3)
        model.fit(X_train, y_train)
        
        self.models['SVM'] = model
        return model

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        XGBoost modelini eğitir.
        """
        if xgb is None:
            raise ImportError("XGBoost kütüphanesi bulunamadı.")
            
        print("🚀 XGBoost eğitiliyor...")
        
        # Dengesiz veri için scale_pos_weight
        neg_count = sum(y_train == 0)
        pos_count = sum(y_train == 1)
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_weight,
            early_stopping_rounds=50,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='auc'
        )

        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            # early_stopping_rounds cannot be used if eval_set is not provided
            model.set_params(early_stopping_rounds=None)
            model.fit(X_train, y_train)

        self.models['XGBoost'] = model
        return model

    def train_mlp(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                   epochs: int = 50, batch_size: int = 64):
        """Derin Öğrenme (MLP) modelini eğitir."""
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
            
        print("🧠 MLP eğitiliyor...")
        
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        self.models['MLP'] = model
        return model, history

    def evaluate_model(self, name: str, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Belirli bir modeli test seti üzerinde değerlendirir."""
        if name not in self.models:
            raise ValueError(f"Model bulunamadı: {name}")
            
        model = self.models[name]
        
        # MLP prediction şekli farklıdır
        if name == 'MLP':
            y_pred_prob = model.predict(X_test).flatten()
        elif hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = model.predict(X_test)

        y_pred = (y_pred_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1score = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_pred_prob)
        except ValueError:
            auc = 0.5 # Sadece tek sınıf varsa
            
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        metrics = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1_Score': f1,
            'AUC': auc,
            'FAR': far
        }
        
        self.metrics[name] = metrics
        return metrics

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Kayıtlı tüm modelleri değerlendirip tablo olarak döndürür."""
        for name in self.models.keys():
            self.evaluate_model(name, X_test, y_test)
                
        df_metrics = pd.DataFrame(self.metrics).T
        if not df_metrics.empty:
            self.best_model_name = df_metrics['F1_Score'].idxmax()
        return df_metrics

    def save_model(self, name: str, filepath: str):
        """Eğitilmiş modeli kaydeder."""
        if name not in self.models:
            raise ValueError(f"Model bulunamadı: {name}")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if name == 'MLP':
            self.models[name].save(filepath)
        else:
            joblib.dump(self.models[name], filepath)
        print(f"✅ Model kaydedildi: {filepath}")

    def save_metadata(self, filepath: str):
        """Eğitim metriklerini JSON olarak kaydeder."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "best_model": self.best_model_name,
            "metrics": self.metrics
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"📝 Metadata kaydedildi: {filepath}")
