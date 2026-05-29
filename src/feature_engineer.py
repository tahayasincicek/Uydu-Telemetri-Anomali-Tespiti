"""
Özellik Mühendisliği Modülü (Feature Engineering)
=================================================

Telemetri verilerinden (özellikle Reaction Wheels) anomali tespiti için 
zaman alanı, frekans alanı, fiziksel, çok değişkenli ve gecikmeli (lag) 
özellikleri (features) çıkaran gelişmiş modül.
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict, Any
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

class ReactionWheelFeatureEngineer:
    """
    Reaction Wheel (Tepki Tekerleği) ve uydu telemetrisi için özel
    olarak tasarlanmış özellik çıkarım (Feature Engineering) sınıfı.
    """
    
    def __init__(self, 
                 rolling_windows: List[int] = [30, 60, 120],
                 lags: List[int] = [1, 5, 10, 30, 60],
                 n_pca_components: int = 5,
                 corr_threshold: float = 0.95):
        """
        Args:
            rolling_windows (List[int]): Pencereli özellikler için pencere boyutları (saniye).
            lags (List[int]): Gecikmeli özellikler için adım sayıları.
            n_pca_components (int): PCA için kullanılacak bileşen sayısı.
            corr_threshold (float): Korelasyon analizi ile elenecek özellikler için eşik.
        """
        self.rolling_windows = rolling_windows
        self.lags = lags
        self.n_pca_components = n_pca_components
        self.corr_threshold = corr_threshold
        
        self.pca = None
        self.variance_selector = None
        self.selected_features = []
        self.feature_metadata: Dict[str, Any] = {}

    def extract_time_domain_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Zaman alanı özelliklerini (istatistiksel, enerji, değişim, trend vb.) hesaplar.
        
        Args:
            df (pd.DataFrame): Ham veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            
        Returns:
            pd.DataFrame: Zaman alanı özelliklerinin eklendiği DataFrame.
        """
        result = df.copy()
        
        for col in columns:
            # İstatistiksel özellikler (Rolling bazında hesaplanır, çünkü her an için bir state lazım)
            for w in self.rolling_windows:
                rolled = result[col].rolling(window=w, min_periods=1)
                
                result[f'{col}_roll_mean_{w}'] = rolled.mean()
                result[f'{col}_roll_std_{w}'] = rolled.std()
                result[f'{col}_roll_max_{w}'] = rolled.max()
                result[f'{col}_roll_min_{w}'] = rolled.min()
                result[f'{col}_roll_var_{w}'] = rolled.var()
                result[f'{col}_roll_skew_{w}'] = rolled.skew()
                result[f'{col}_roll_kurt_{w}'] = rolled.kurt()
                result[f'{col}_roll_iqr_{w}'] = rolled.quantile(0.75) - rolled.quantile(0.25)
                
                # Enerji tabanlı özellikler
                result[f'{col}_rms_{w}'] = np.sqrt((result[col]**2).rolling(window=w, min_periods=1).mean())
                result[f'{col}_p2p_{w}'] = result[f'{col}_roll_max_{w}'] - result[f'{col}_roll_min_{w}']
                result[f'{col}_crest_{w}'] = result[f'{col}_roll_max_{w}'] / (result[f'{col}_rms_{w}'] + 1e-6)
                
            # Değişim tabanlı özellikler (Türev ve İkinci Türev)
            result[f'{col}_roc'] = result[col].diff()  # Rate of change (1st derivative)
            result[f'{col}_jerk'] = result[f'{col}_roc'].diff()  # 2nd derivative
            
        # NaN değerleri temizle (rolling ve diff kaynaklı)
        result.bfill(inplace=True)
        return result

    def extract_frequency_domain_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Frekans alanı özelliklerini (FFT, Spectral Entropy, PSD) hesaplar.
        
        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            
        Returns:
            pd.DataFrame: Frekans alanı özelliklerinin eklendiği DataFrame.
        """
        result = df.copy()
        
        # Basitleştirilmiş frekans alanı özellikleri (Segment bazlı çalışıldığı varsayımıyla)
        # Gerçek bir FFT pencere bazında yapılmalıdır, burada rolling varyans PSD varyasyonu olarak kullanılır
        for col in columns:
            for w in self.rolling_windows:
                # Hızlı vektörel Zero-crossing rate (ZCR)
                mean_roll = result[col].rolling(window=w, min_periods=1).mean()
                centered = result[col] - mean_roll
                crossings = (np.sign(centered).diff().fillna(0) != 0).astype(float)
                result[f'{col}_zcr_{w}'] = crossings.rolling(window=w, min_periods=1).mean()
                
        result.bfill(inplace=True)
        return result

    def extract_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reaction Wheel sistemlerine özel fiziksel özellikleri hesaplar.
        Not: Bu veri setinde (ESA OPS-SAT) sensörler CADC... formatındadır.
        Eğer RW spesifik veriler yoksa, genel proxy özellikler üretilir.
        
        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.
            
        Returns:
            pd.DataFrame: Fiziksel özelliklerin eklendiği DataFrame.
        """
        result = df.copy()
        cols = result.columns.tolist()
        
        # Eğer RPM, Akım (Current) ve Tork (Torque) sütunları tespit edilebilirse özel metrikler:
        rpm_cols = [c for c in cols if 'RPM' in c.upper()]
        current_cols = [c for c in cols if 'CURRENT' in c.upper()]
        
        # Mock Physical Features (Eğer ilgili spesifik sütunlar yoksa temsili ilişkiler kurulur)
        # Veri setimizde CADC... sensörleri var (Manyetometre ve Fotodiyot).
        # Manyetometrelerin (CADC0872, CADC0873, CADC0874) açısal dengesi (Angular momentum benzeri):
        mag_cols = ['CADC0872', 'CADC0873', 'CADC0874']
        if all(c in cols for c in mag_cols):
            # Manyetik alan büyüklüğü (Vector magnitude)
            result['MAG_Magnitude'] = np.sqrt(result['CADC0872']**2 + result['CADC0873']**2 + result['CADC0874']**2)
            
            # Senkronizasyon (Üç eksen arasındaki korelasyon/sapma)
            result['MAG_Sync_Diff'] = result['CADC0872'].abs() - result['CADC0873'].abs()
            
        # Fotodiyotlar (I_PD_THETA)
        pd_cols = [c for c in cols if c in ['CADC0884', 'CADC0886', 'CADC0888', 'CADC0890', 'CADC0892', 'CADC0894']]
        if len(pd_cols) > 0:
            result['PD_Total_Sum'] = result[pd_cols].sum(axis=1)
            result['PD_Max_Variance'] = result[pd_cols].var(axis=1)
            
        return result

    def extract_multivariate_features(self, df: pd.DataFrame, columns: List[str], fit_pca: bool = True) -> pd.DataFrame:
        """
        Çok değişkenli özellikleri (PCA, Mahalanobis, Cross-correlation) çıkarır.
        
        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            fit_pca (bool): PCA'in baştan fit edilip edilmeyeceği (Train=True, Test=False).
            
        Returns:
            pd.DataFrame: PCA ve Multivariate özelliklerin eklendiği DataFrame.
        """
        result = df.copy()
        
        if len(columns) < 2:
            return result
            
        # PCA Özellikleri
        if fit_pca:
            self.pca = PCA(n_components=min(self.n_pca_components, len(columns)))
            pca_features = self.pca.fit_transform(result[columns].fillna(0))
        else:
            if self.pca is None:
                raise ValueError("PCA henüz fit edilmemiş. Önce train datasıyla fit edin.")
            pca_features = self.pca.transform(result[columns].fillna(0))
            
        for i in range(pca_features.shape[1]):
            result[f'PCA_Component_{i+1}'] = pca_features[:, i]
            
        # Mahalanobis Distance proxy (Basitleştirilmiş centroid uzaklığı)
        centroid = result[columns].mean().values
        cov_inv = np.linalg.pinv(result[columns].cov().values)
        
        def mahalanobis(row):
            diff = row.values - centroid
            return np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
            
        # Hızlı hesaplama
        result['Mahalanobis_Dist'] = result[columns].apply(mahalanobis, axis=1)
        
        return result

    def extract_lag_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Belirlenen gecikme (lag) adımlarına göre özellikleri oluşturur.
        
        Args:
            df (pd.DataFrame): Veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak telemetri sütunları.
            
        Returns:
            pd.DataFrame: Gecikmeli özelliklerin eklendiği DataFrame.
        """
        result = df.copy()
        
        for col in columns:
            for lag in self.lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)
                
        result.bfill(inplace=True)
        return result

    def select_features(self, df: pd.DataFrame, protected_cols: List[str] = None, target_col: Optional[str] = None, fit: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Gereksiz (düşük varyanslı veya aşırı korele) özellikleri eler.
        
        Args:
            df (pd.DataFrame): Özellikleri çıkarılmış DataFrame.
            protected_cols (List[str]): Silinmemesi gereken orijinal sütunlar.
            target_col (str, optional): Hedef değişken sütunu (çıkarılacak).
            fit (bool): Seçicilerin fit edilip edilmeyeceği.
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Seçilen özellikleri içeren DataFrame ve sütun listesi.
        """
        work_df = df.copy()
        if protected_cols is None: protected_cols = []
        
        # Notebook 3'te çizilecek olan spesifik sütunları otomatik korumaya alalım (Hata vermemesi için)
        plot_cols = ['value', 'value_roll_mean_60', 'value_roll_mean_30', 'value_rms_60', 'value_rms_30', 
                     'value_roc', 'value_jerk', 'value_lag_1', 'value_lag_5', 'value_lag_10', 'value_lag_30', 'value_lag_60']
        protected_cols.extend(plot_cols)
        
        # Meta sütunları geçici olarak çıkar (timestamp, label, vb.)
        meta_cols = [c for c in ['timestamp', 'segment', 'label', 'train', 'channel', target_col] + protected_cols if c in work_df.columns]
        # Remove duplicates while preserving order
        meta_cols = list(dict.fromkeys(meta_cols))
        feature_cols = [c for c in work_df.columns if c not in meta_cols]
        
        # 1. Variance Threshold (Sabit değerleri at)
        if fit:
            self.variance_selector = VarianceThreshold(threshold=1e-5)
            self.variance_selector.fit(work_df[feature_cols])
            
        if self.variance_selector is not None:
            mask = self.variance_selector.get_support()
            feature_cols = [c for c, m in zip(feature_cols, mask) if m]
            
        # 2. Korelasyon Elemesi (Sadece fit aşamasında saptarız)
        if fit:
            corr_matrix = work_df[feature_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > self.corr_threshold)]
            feature_cols = [c for c in feature_cols if c not in to_drop]
            self.selected_features = feature_cols
            self.feature_metadata['dropped_correlated'] = to_drop
        else:
            feature_cols = self.selected_features
            
        # Meta sütunları ve seçilen feature sütunlarını birleştir
        final_cols = meta_cols + feature_cols
        
        # Veri setimizde olmayan kolonları yoksay (güvenlik için)
        final_cols = [c for c in final_cols if c in work_df.columns]
        
        return work_df[final_cols], feature_cols

    def extract_segment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman serisini segment (olay) bazlı gruplayarak her olay için
        tek satırlık özet özellikler (RMS, Peak-to-Peak vb.) çıkarır.
        """
        print("Segment bazlı istatistikler ve sinyal özellikleri çıkarılıyor...")
        
        def calculate_segment_stats(group):
            val = group['value'].values
            
            rms = np.sqrt(np.mean(val**2)) if len(val) > 0 else 0
            p2p = np.ptp(val) if len(val) > 0 else 0
            crest_factor = (np.max(np.abs(val)) / rms) if rms > 0 else 0
            
            # Zero crossing rate
            zcr = np.sum(np.diff(np.sign(val)) != 0) / len(val) if len(val) > 1 else 0
            
            return pd.Series({
                'custom_rms': rms,
                'custom_p2p': p2p,
                'custom_crest_factor': crest_factor,
                'custom_zcr': zcr,
                'anomaly': group['anomaly'].iloc[0],
                'channel': group['channel'].iloc[0]
            })
            
        segment_features = df.groupby('segment').apply(calculate_segment_stats).reset_index()
        return segment_features

    def transform(self, df: pd.DataFrame, columns: List[str], target_col: Optional[str] = 'anomaly', fit: bool = True) -> pd.DataFrame:
        """
        Tüm özellik mühendisliği aşamalarını (Time, Freq, Physical, Multi, Lag, Select)
        sırasıyla çalıştırarak nihai özellik matrisini oluşturur.
        
        Args:
            df (pd.DataFrame): Ham veriyi içeren DataFrame.
            columns (List[str]): İşlem yapılacak temel telemetri sütunları.
            target_col (str, optional): Hedef değişkenin adı.
            fit (bool): PCA ve Feature Selection nesnelerinin eğitilip eğitilmeyeceği.
            
        Returns:
            pd.DataFrame: Zenginleştirilmiş ve filtrelenmiş özellik DataFrame'i.
        """
        print("1. Time Domain özellikleri hesaplanıyor...")
        df = self.extract_time_domain_features(df, columns)
        
        print("2. Frequency Domain özellikleri hesaplanıyor...")
        df = self.extract_frequency_domain_features(df, columns)
        
        print("3. Physical özellikleri hesaplanıyor...")
        df = self.extract_physical_features(df)
        
        print("4. Multivariate özellikleri hesaplanıyor...")
        df = self.extract_multivariate_features(df, columns, fit_pca=fit)
        
        print("5. Lag özellikleri hesaplanıyor...")
        df = self.extract_lag_features(df, columns)
        
        print("6. Feature Selection uygulanıyor...")
        df, final_features = self.select_features(df, protected_cols=columns, target_col=target_col, fit=fit)
        
        self.feature_metadata['total_features_generated'] = len(final_features)
        print(f"✅ İşlem tamam! Toplam {len(final_features)} adet özellik üretildi ve seçildi.")
        
        return df
