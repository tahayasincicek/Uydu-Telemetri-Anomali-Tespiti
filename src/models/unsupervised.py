"""
Gözetimsiz Anomali Tespiti Modülü (Unsupervised Learning)
===========================================================

Uydu telemetrisi (Reaction Wheels) verilerinde etiket kullanmadan
(unsupervised) anomali tespiti yapan modelleri içerir.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, RepeatVector, TimeDistributed
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    Sequential, Model = None, None


class UnsupervisedAnomalyDetector:
    """
    Etiket gerektirmeyen gözetimsiz algoritmaları (Isolation Forest, Autoencoder,
    LSTM Autoencoder, One-Class SVM, K-Means) kullanarak anomali tespiti yapan sınıf.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.thresholds: Dict[str, float] = {}

    def train_isolation_forest(self, X_train: pd.DataFrame, contamination: float = 0.05) -> IsolationForest:
        """
        Isolation Forest modelini eğitir.
        """
        print("🌲 Isolation Forest eğitiliyor...")
        model = IsolationForest(
            n_estimators=200, 
            max_features=1.0, 
            contamination=contamination, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        model.fit(X_train)
        
        # Skorlar (ne kadar küçükse o kadar anomali, ama biz pozitife çeviriyoruz)
        scores = -model.score_samples(X_train)
        # Eşik değeri olarak ortalama + 3 standart sapma
        threshold = np.mean(scores) + 3 * np.std(scores)
        
        self.models['IsolationForest'] = model
        self.thresholds['IsolationForest'] = threshold
        return model

    def train_autoencoder(self, X_train: np.ndarray, X_val: np.ndarray, epochs: int = 50, batch_size: int = 64):
        """
        Derin Öğrenme tabanlı Tabular Autoencoder modelini eğitir.
        """
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
            
        print("🧠 Autoencoder eğitiliyor...")
        input_dim = X_train.shape[1]
        
        model = Sequential([
            # Encoder
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu', name='latent_space'),
            
            # Decoder
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(input_dim, activation='sigmoid') # X_train 0-1 arası normalize edilmiş olmalı veya linear
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, X_train, # Autoencoder'da X ve y aynıdır
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Train seti üzerinde reconstruction error (MSE) hesapla
        reconstructions = model.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        threshold = np.mean(mse) + 3 * np.std(mse)
        
        self.models['Autoencoder'] = model
        self.thresholds['Autoencoder'] = float(threshold)
        return model, history

    def train_lstm_autoencoder(self, X_train_seq: np.ndarray, X_val_seq: np.ndarray, seq_len: int, features: int, epochs: int = 20, batch_size: int = 64):
        """
        Zaman serisi anomalileri için LSTM Autoencoder modelini eğitir.
        """
        if Sequential is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
            
        print("⏳ LSTM Autoencoder eğitiliyor...")
        
        model = Sequential([
            # Encoder
            Input(shape=(seq_len, features)),
            LSTM(64, return_sequences=True),
            LSTM(32, return_sequences=False),
            
            # Reconstruct sequence
            RepeatVector(seq_len),
            
            # Decoder
            LSTM(32, return_sequences=True),
            LSTM(64, return_sequences=True),
            TimeDistributed(Dense(features))
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train_seq, X_train_seq,
            validation_data=(X_val_seq, X_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        reconstructions = model.predict(X_train_seq)
        mse = np.mean(np.power(X_train_seq - reconstructions, 2), axis=(1,2))
        threshold = np.mean(mse) + 3 * np.std(mse)
        
        self.models['LSTM_Autoencoder'] = model
        self.thresholds['LSTM_Autoencoder'] = float(threshold)
        return model, history

    def train_one_class_svm(self, X_train: pd.DataFrame, nu: float = 0.05) -> OneClassSVM:
        """
        One-Class SVM modelini eğitir. (Sadece normal verilerle eğitilmesi tavsiye edilir).
        """
        print("⚔️ One-Class SVM eğitiliyor...")
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
        model.fit(X_train)
        
        # Skorlar
        scores = -model.decision_function(X_train)
        threshold = np.percentile(scores, 100 * (1 - nu))
        
        self.models['OneClassSVM'] = model
        self.thresholds['OneClassSVM'] = threshold
        return model

    def train_kmeans(self, X_train: pd.DataFrame, n_clusters: int = 3) -> KMeans:
        """
        K-Means Clustering modelini eğitir.
        Küme merkezine uzaklık anomali skoru olarak kullanılır.
        """
        print(f"🎯 K-Means (K={n_clusters}) eğitiliyor...")
        model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
        model.fit(X_train)
        
        # Uzaklık skorlarını hesapla
        distances = model.transform(X_train)
        min_distances = np.min(distances, axis=1)
        threshold = np.mean(min_distances) + 3 * np.std(min_distances)
        
        self.models['KMeans'] = model
        self.thresholds['KMeans'] = threshold
        return model
        
    def train_lof(self, X_train: pd.DataFrame, n_neighbors: int = 20) -> LocalOutlierFactor:
        """
        Local Outlier Factor (LOF) modelini eğitir. Novelty modu açık.
        """
        print("🔍 Local Outlier Factor (LOF) eğitiliyor...")
        model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        model.fit(X_train)
        
        scores = -model.score_samples(X_train)
        threshold = np.mean(scores) + 3 * np.std(scores)
        
        self.models['LOF'] = model
        self.thresholds['LOF'] = threshold
        return model

    def compute_ensemble_score(self, X_test: np.ndarray, active_models: List[str] = None) -> np.ndarray:
        """
        Kayıtlı modellerin (aktif olanların) anomali skorlarını normalize edip birleştirir.
        """
        if active_models is None:
            active_models = ['IsolationForest', 'Autoencoder', 'OneClassSVM', 'KMeans', 'LOF']
            
        print(f"🤝 Ensemble Anomali Skoru Hesaplanıyor ({len(active_models)} model)...")
        scores_matrix = []
        
        for name in active_models:
            if name not in self.models:
                continue
                
            model = self.models[name]
            if name == 'IsolationForest' or name == 'LOF':
                scores = -model.score_samples(X_test)
            elif name == 'OneClassSVM':
                scores = -model.decision_function(X_test)
            elif name == 'Autoencoder':
                recon = model.predict(X_test, verbose=0)
                scores = np.mean(np.power(X_test - recon, 2), axis=1)
            elif name == 'KMeans':
                dist = model.transform(X_test)
                scores = np.min(dist, axis=1)
            else:
                continue
                
            # Skaler olarak Normalize Et (Min-Max Scaling)
            scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
            scores_matrix.append(scores_norm)
            
        # Modellerin ortalama skoru (Majority/Ağırlıklı)
        ensemble_score = np.mean(np.array(scores_matrix), axis=0)
        return ensemble_score

    def detect_anomalies(self, ensemble_score: np.ndarray, global_threshold: float = 0.5) -> np.ndarray:
        """
        Ensemble skoru belirli bir threshold'u geçenleri anomali (1) olarak işaretler.
        """
        return (ensemble_score > global_threshold).astype(int)

    def save_models(self, path: str):
        """Tüm modelleri ve threshold değerlerini kaydeder."""
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            filepath = os.path.join(path, f"{name.lower()}_model")
            if name in ['Autoencoder', 'LSTM_Autoencoder']:
                model.save(filepath + ".keras")
            else:
                joblib.dump(model, filepath + ".joblib")
                
        # Thresholds JSON kaydı
        with open(os.path.join(path, "unsupervised_thresholds.json"), "w", encoding='utf-8') as f:
            json.dump(self.thresholds, f, indent=4)
        print("✅ Gözetimsiz modeller başarıyla kaydedildi.")
