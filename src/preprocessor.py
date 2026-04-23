"""
Veri Ön İşleme Modülü
======================

Uydu telemetri verilerinin temizlenmesi, dönüştürülmesi ve
model eğitimine hazırlanması işlemlerini gerçekleştirir.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer


class TelemetryPreprocessor:
    """
    Uydu telemetri verilerini ön işleme tabi tutan sınıf.

    İşlem adımları:
        1. Eksik veri doldurma
        2. Aykırı değer tespiti ve işleme
        3. Normalizasyon / Standartlaştırma
        4. Zaman damgası işleme

    Attributes:
        scaler: Kullanılan ölçeklendirici.
        imputer: Kullanılan eksik veri doldurma yöntemi.
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        impute_method: str = "median",
        outlier_method: str = "iqr",
    ):
        """
        Args:
            scaling_method (str): Ölçeklendirme yöntemi
                ('standard', 'minmax', 'robust').
            impute_method (str): Eksik veri doldurma yöntemi
                ('mean', 'median', 'knn', 'forward_fill').
            outlier_method (str): Aykırı değer tespiti yöntemi
                ('iqr', 'zscore', 'isolation_forest').
        """
        self.scaling_method = scaling_method
        self.impute_method = impute_method
        self.outlier_method = outlier_method
        self.scaler = None
        self.imputer = None
        self.is_fitted = False

        self._init_scaler()
        self._init_imputer()

    def _init_scaler(self):
        """Ölçeklendiriciyi başlatır."""
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        self.scaler = scalers.get(self.scaling_method)
        if self.scaler is None:
            raise ValueError(f"Geçersiz ölçeklendirme yöntemi: {self.scaling_method}")

    def _init_imputer(self):
        """Eksik veri doldurucu başlatır."""
        if self.impute_method == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
        elif self.impute_method in ["mean", "median"]:
            self.imputer = SimpleImputer(strategy=self.impute_method)
        elif self.impute_method == "forward_fill":
            self.imputer = None  # pandas ffill kullanılacak
        else:
            raise ValueError(f"Geçersiz doldurma yöntemi: {self.impute_method}")

    def fit(self, data: pd.DataFrame, numeric_columns: Optional[List[str]] = None):
        """
        Ön işleme parametrelerini veriye göre öğrenir.

        Args:
            data (pd.DataFrame): Eğitim verisi.
            numeric_columns (list, optional): Sayısal sütunlar.
        """
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        self.numeric_columns = numeric_columns

        # Imputer fit
        if self.imputer is not None:
            self.imputer.fit(data[numeric_columns])

        # Scaler fit
        self.scaler.fit(data[numeric_columns])

        self.is_fitted = True
        print("✅ Preprocessor fit işlemi tamamlandı.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Veriye ön işleme uygular.

        Args:
            data (pd.DataFrame): Dönüştürülecek veri.

        Returns:
            pd.DataFrame: Ön işlenmiş veri.
        """
        if not self.is_fitted:
            raise RuntimeError("Önce fit() çağırılmalı.")

        df = data.copy()

        # Eksik veri doldurma
        df = self._handle_missing(df)

        # Aykırı değer işleme
        df = self._handle_outliers(df)

        # Ölçeklendirme
        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])

        print("✅ Veri dönüşümü tamamlandı.")
        return df

    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit ve transform işlemlerini sırasıyla uygular."""
        self.fit(data, **kwargs)
        return self.transform(data)

    def _handle_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Eksik verileri doldurur."""
        df = data.copy()

        if self.impute_method == "forward_fill":
            df[self.numeric_columns] = df[self.numeric_columns].ffill().bfill()
        elif self.imputer is not None:
            df[self.numeric_columns] = self.imputer.transform(
                df[self.numeric_columns]
            )

        remaining = df[self.numeric_columns].isnull().sum().sum()
        if remaining > 0:
            print(f"⚠️  {remaining} eksik değer hâlâ mevcut.")

        return df

    def _handle_outliers(
        self, data: pd.DataFrame, threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Aykırı değerleri tespit edip kırpar (clip).

        Args:
            data (pd.DataFrame): Veri.
            threshold (float): IQR çarpanı (varsayılan 1.5).

        Returns:
            pd.DataFrame: Aykırı değerleri kırpılmış veri.
        """
        df = data.copy()

        if self.outlier_method == "iqr":
            for col in self.numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df[col] = df[col].clip(lower=lower, upper=upper)

        elif self.outlier_method == "zscore":
            for col in self.numeric_columns:
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - 3 * std
                upper = mean + 3 * std
                df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    @staticmethod
    def parse_timestamps(
        data: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Zaman damgası sütununu ayrıştırır ve ek özellikler çıkarır.

        Args:
            data (pd.DataFrame): Veri.
            timestamp_col (str): Zaman damgası sütun adı.

        Returns:
            pd.DataFrame: Zaman özellikleri eklenmiş veri.
        """
        df = data.copy()

        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df["saat"] = df[timestamp_col].dt.hour
            df["gun"] = df[timestamp_col].dt.day
            df["hafta_gunu"] = df[timestamp_col].dt.dayofweek
            df["ay"] = df[timestamp_col].dt.month
            print("✅ Zaman damgası özellikleri çıkarıldı.")
        else:
            print(f"⚠️  '{timestamp_col}' sütunu bulunamadı.")

        return df
