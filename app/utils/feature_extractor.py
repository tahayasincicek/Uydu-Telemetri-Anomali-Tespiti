import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks

def extract_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw telemetry signals grouped by segment and calculates the 24 required
    statistical and signal processing features for the ML models.
    """
    # Group by segment to calculate features per segment
    features = []
    
    for segment_id, group in df.groupby('segment'):
        val = group['value'].values
        
        # If no values, skip
        if len(val) == 0:
            continue
            
        f_dict = {}
        f_dict['segment'] = segment_id
        
        # Meta info
        f_dict['channel'] = group['channel'].iloc[0] if 'channel' in group.columns else 'UNKNOWN'
        f_dict['anomaly'] = group['anomaly'].iloc[0] if 'anomaly' in group.columns else 0
        f_dict['train'] = group['train'].iloc[0] if 'train' in group.columns else 0
        f_dict['sampling'] = group['sampling'].iloc[0] if 'sampling' in group.columns else 1
        
        # Length & Duration
        n_len = len(val)
        f_dict['len'] = n_len
        f_dict['duration'] = n_len - 1 if n_len > 1 else 1 # Rough estimate of duration
        
        # Basic Stats
        f_dict['mean'] = np.mean(val)
        f_dict['var'] = np.var(val)
        f_dict['std'] = np.std(val)
        
        # Kurtosis & Skew
        f_dict['kurtosis'] = stats.kurtosis(val) if n_len > 3 else 0
        f_dict['skew'] = stats.skew(val) if n_len > 2 else 0
        
        # Peaks
        peaks, _ = find_peaks(val)
        f_dict['n_peaks'] = len(peaks)
        
        # Smooth peaks (simple rolling mean simulation)
        s10 = pd.Series(val).rolling(10, min_periods=1).mean().values
        p10, _ = find_peaks(s10)
        f_dict['smooth10_n_peaks'] = len(p10)
        
        s20 = pd.Series(val).rolling(20, min_periods=1).mean().values
        p20, _ = find_peaks(s20)
        f_dict['smooth20_n_peaks'] = len(p20)
        
        # Diff features
        diff1 = np.diff(val)
        p_diff, _ = find_peaks(diff1)
        f_dict['diff_peaks'] = len(p_diff)
        f_dict['diff_var'] = np.var(diff1) if len(diff1) > 0 else 0
        
        diff2 = np.diff(diff1)
        p_diff2, _ = find_peaks(diff2)
        f_dict['diff2_peaks'] = len(p_diff2)
        f_dict['diff2_var'] = np.var(diff2) if len(diff2) > 0 else 0
        
        # Gaps & Weights (approximations for specific telemetry attributes)
        # gaps_squared: rough approx
        f_dict['gaps_squared'] = np.sum(diff1**2)
        f_dict['len_weighted'] = n_len
        
        # Div features
        f_dict['var_div_duration'] = f_dict['var'] / f_dict['duration'] if f_dict['duration'] > 0 else 0
        f_dict['var_div_len'] = f_dict['var'] / f_dict['len'] if f_dict['len'] > 0 else 0
        
        # Custom signal features (RMS, P2P, Crest, ZCR)
        rms = np.sqrt(np.mean(val**2)) if len(val) > 0 else 0
        p2p = np.ptp(val) if len(val) > 0 else 0
        crest_factor = (np.max(np.abs(val)) / rms) if rms > 0 else 0
        zcr = np.sum(np.diff(np.sign(val)) != 0) / n_len if n_len > 1 else 0
        
        f_dict['custom_rms'] = rms
        f_dict['custom_p2p'] = p2p
        f_dict['custom_crest_factor'] = crest_factor
        f_dict['custom_zcr'] = zcr
        
        features.append(f_dict)
        
    df_features = pd.DataFrame(features)
    
    # Handle channel_id
    if not df_features.empty:
        # Give unique int for each channel. If CADC0872 is present, try to keep it 0
        df_features['channel_id'] = pd.factorize(df_features['channel'])[0]
        
    # Fill any NaNs
    df_features = df_features.fillna(0)
    
    return df_features
