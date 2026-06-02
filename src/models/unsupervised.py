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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Input, BatchNormalization, RepeatVector,
        TimeDistributed, Lambda,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    import keras
except ImportError:
    Sequential, Model = None, None

# PyOD anomali dedektörleri (opsiyonel bağımlılık)
try:
    from pyod.models.ecod import ECOD
    from pyod.models.copod import COPOD
    from pyod.models.hbos import HBOS
    from pyod.models.cblof import CBLOF
    _PYOD_AVAILABLE = True
except ImportError:
    _PYOD_AVAILABLE = False

# Tek tip API'ye (fit / decision_function / predict) sahip PyOD modelleri
PYOD_MODELS = {"ECOD", "COPOD", "HBOS", "CBLOF"}


class UnsupervisedAnomalyDetector:
    """
    Etiket gerektirmeyen gözetimsiz algoritmaları kullanarak anomali tespiti yapan sınıf.

    sklearn/derin: Isolation Forest, One-Class SVM, K-Means, LOF, Autoencoder,
        LSTM Autoencoder, GMM, Elliptic Envelope, PCA (recon), DBSCAN, VAE.
    PyOD: ECOD, COPOD, HBOS, CBLOF.
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

    # ==================================================================
    #  Ek Gözetimsiz Modeller (sklearn tabanlı)
    # ==================================================================

    def train_gmm(self, X_train: np.ndarray, n_components: int = 3) -> GaussianMixture:
        """Gaussian Mixture Model (GMM). Düşük olabilirlik = anomali."""
        print("🔔 Gaussian Mixture Model eğitiliyor...")
        model = GaussianMixture(n_components=n_components, covariance_type='full',
                                random_state=self.random_state)
        model.fit(X_train)
        scores = -model.score_samples(X_train)  # negatif log-olabilirlik
        threshold = np.mean(scores) + 3 * np.std(scores)
        self.models['GMM'] = model
        self.thresholds['GMM'] = float(threshold)
        return model

    def train_elliptic_envelope(self, X_train: np.ndarray, contamination: float = 0.05) -> EllipticEnvelope:
        """Elliptic Envelope (robust kovaryans tabanlı Mahalanobis anomali tespiti)."""
        print("🥚 Elliptic Envelope eğitiliyor...")
        model = EllipticEnvelope(contamination=contamination, random_state=self.random_state)
        model.fit(X_train)
        scores = -model.score_samples(X_train)
        threshold = np.percentile(scores, 100 * (1 - contamination))
        self.models['EllipticEnvelope'] = model
        self.thresholds['EllipticEnvelope'] = float(threshold)
        return model

    def train_pca(self, X_train: np.ndarray, n_components: float = 0.95) -> PCA:
        """PCA tabanlı yeniden yapılandırma hatası ile anomali tespiti."""
        print("📉 PCA (reconstruction error) eğitiliyor...")
        model = PCA(n_components=n_components, random_state=self.random_state)
        model.fit(X_train)
        recon = model.inverse_transform(model.transform(X_train))
        scores = np.mean(np.power(X_train - recon, 2), axis=1)
        threshold = np.mean(scores) + 3 * np.std(scores)
        self.models['PCA'] = model
        self.thresholds['PCA'] = float(threshold)
        return model

    def train_dbscan(self, X_train: np.ndarray, eps: float = 1.5, min_samples: int = 5):
        """DBSCAN ile çekirdek (core) noktaları bul; yeni nokta skoru = en yakın
        çekirdek noktaya uzaklık. Kaydedilen model bir NearestNeighbors nesnesidir
        (DBSCAN doğrudan yeni veri üzerinde predict desteklemez).
        """
        print("🌌 DBSCAN (core-distance novelty) eğitiliyor...")
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X_train)
        # Çekirdek (gürültü olmayan) noktalar
        core = X_train[labels != -1]
        if len(core) == 0:
            core = X_train  # tümü gürültüyse geri düş
        nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(core)
        # Eşik kalibrasyonu: eğitim noktasının kendine 0 mesafesini saymamak için
        # 2-en-yakın komşu sorgusu yapıp ikinci komşuyu (kendisi hariç) kullanırız.
        nbrs2 = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(core)
        dist2, _ = nbrs2.kneighbors(X_train)
        scores = dist2[:, 1]
        threshold = float(np.mean(scores) + 3 * np.std(scores))
        self.models['DBSCAN'] = nbrs
        self.thresholds['DBSCAN'] = threshold
        return nbrs

    def train_vae(self, X_train: np.ndarray, X_val: np.ndarray, latent_dim: int = 8,
                  epochs: int = 50, batch_size: int = 64, beta: float = 1.0):
        """Variational Autoencoder (VAE). Anomali skoru = yeniden yapılandırma hatası."""
        if Model is None:
            raise ImportError("TensorFlow/Keras bulunamadı.")
        print("🧠 Variational Autoencoder (VAE) eğitiliyor...")
        input_dim = X_train.shape[1]

        # Keras 3'te Functional model üzerinde add_loss() kaldırıldı;
        # KL kaybı örnekleme katmanının call() içinde add_loss() ile eklenmeli.
        class SamplingLayer(keras.layers.Layer):
            def __init__(self, beta=1.0, **kwargs):
                super().__init__(**kwargs)
                self._beta = beta

            def call(self, inputs):
                z_mean, z_log_var = inputs
                eps = keras.random.normal(shape=keras.ops.shape(z_mean))
                kl = -0.5 * self._beta * keras.ops.mean(
                    keras.ops.sum(
                        1 + z_log_var - keras.ops.square(z_mean) - keras.ops.exp(z_log_var),
                        axis=1,
                    )
                )
                self.add_loss(kl)
                return z_mean + keras.ops.exp(0.5 * z_log_var) * eps

        inputs = Input(shape=(input_dim,))
        h = Dense(64, activation='relu')(inputs)
        h = BatchNormalization()(h)
        h = Dense(32, activation='relu')(h)
        z_mean = Dense(latent_dim, name='z_mean')(h)
        z_log_var = Dense(latent_dim, name='z_log_var')(h)
        z = SamplingLayer(beta=beta, name='z')([z_mean, z_log_var])

        d = Dense(32, activation='relu')(z)
        d = Dense(64, activation='relu')(d)
        outputs = Dense(input_dim, activation='linear')(d)

        vae = Model(inputs, outputs, name='VAE')
        vae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        vae.fit(X_train, X_train, validation_data=(X_val, X_val),
                epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)

        recon = vae.predict(X_train, verbose=0)
        scores = np.mean(np.power(X_train - recon, 2), axis=1)
        threshold = np.mean(scores) + 3 * np.std(scores)
        self.models['VAE'] = vae
        self.thresholds['VAE'] = float(threshold)
        return vae

    # ==================================================================
    #  PyOD Anomali Dedektörleri (ECOD, COPOD, HBOS, CBLOF)
    # ==================================================================

    def train_pyod(self, X_train: np.ndarray, contamination: float = 0.05) -> Dict[str, Any]:
        """PyOD tabanlı dedektörleri (ECOD, COPOD, HBOS, CBLOF) eğitir.

        Her dedektör kendi ``threshold_`` değerini taşır; tahmin için
        ``predict`` (0/1) ve skor için ``decision_function`` kullanılır.
        """
        if not _PYOD_AVAILABLE:
            print("⚠️ PyOD kurulu değil — atlanıyor (pip install pyod).")
            return {}
        print("🧪 PyOD dedektörleri (ECOD, COPOD, HBOS, CBLOF) eğitiliyor...")
        detectors = {
            'ECOD':  ECOD(contamination=contamination),
            'COPOD': COPOD(contamination=contamination),
            'HBOS':  HBOS(contamination=contamination),
            'CBLOF': CBLOF(contamination=contamination, random_state=self.random_state),
        }
        trained = {}
        for name, det in detectors.items():
            det.fit(X_train)
            self.models[name] = det
            self.thresholds[name] = float(det.threshold_)
            trained[name] = det
        return trained

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
            if name in ['Autoencoder', 'LSTM_Autoencoder', 'VAE']:
                model.save(filepath + ".keras")
            else:
                joblib.dump(model, filepath + ".joblib")
                
        # Thresholds JSON kaydı
        with open(os.path.join(path, "unsupervised_thresholds.json"), "w", encoding='utf-8') as f:
            json.dump(self.thresholds, f, indent=4)
        print("✅ Gözetimsiz modeller başarıyla kaydedildi.")
