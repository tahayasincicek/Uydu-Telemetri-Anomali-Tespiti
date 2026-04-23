"""
Özellik Mühendisliği Modülü
============================
Telemetri verileri üzerinde istatistiksel, zaman serisi ve frekans tabanlı özellikler oluşturur.
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from scipy import stats


class FeatureEngineer:
    """Telemetri verilerinden anlamlı özellikler çıkaran sınıf."""

    def __init__(self, window_sizes: Optional[List[int]] = None):
        self.window_sizes = window_sizes or [5, 10, 30]
        self.feature_names: List[str] = []

    def create_features(self, data: pd.DataFrame, numeric_columns: Optional[List[str]] = None,
                        include_stats=True, include_rolling=True, include_lags=True, include_diffs=True) -> pd.DataFrame:
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        df = data.copy()
        if include_stats:
            df = self._add_statistical_features(df, numeric_columns)
        if include_rolling:
            df = self._add_rolling_features(df, numeric_columns)
        if include_lags:
            df = self._add_lag_features(df, numeric_columns)
        if include_diffs:
            df = self._add_diff_features(df, numeric_columns)
        df = df.dropna().reset_index(drop=True)
        self.feature_names = list(df.columns)
        print(f"✅ Özellik mühendisliği tamamlandı. Toplam sütun: {len(df.columns)}")
        return df

    def _add_statistical_features(self, data, columns):
        df = data.copy()
        for col in columns:
            df[f"{col}_zscore"] = stats.zscore(df[col].fillna(0))
        return df

    def _add_rolling_features(self, data, columns):
        df = data.copy()
        for w in self.window_sizes:
            for col in columns:
                df[f"{col}_rmean_{w}"] = df[col].rolling(window=w, min_periods=1).mean()
                df[f"{col}_rstd_{w}"] = df[col].rolling(window=w, min_periods=1).std()
        return df

    def _add_lag_features(self, data, columns, lags=None):
        df = data.copy()
        for lag in (lags or [1, 3, 5]):
            for col in columns:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        return df

    def _add_diff_features(self, data, columns, orders=None):
        df = data.copy()
        for order in (orders or [1, 2]):
            for col in columns:
                df[f"{col}_diff_{order}"] = df[col].diff(order)
        return df
