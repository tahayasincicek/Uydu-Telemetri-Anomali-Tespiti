"""Model loading and prediction utilities."""
import os, json, joblib, warnings
import numpy as np
import pandas as pd

# ── TensorFlow environment fixes (must be set BEFORE importing tf) ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"           # suppress TF info/warning logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"           # disable oneDNN to avoid numerical noise
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"            # force CPU-only (avoids CUDA init crash)
os.environ["TF_NUM_INTEROP_THREADS"] = "1"           # limit inter-op parallelism
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"           # limit intra-op parallelism

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(ROOT, "models")
UNSUP_DIR = os.path.join(MODEL_DIR, "unsupervised")


def _safe_load(path):
    try:
        return joblib.load(path) if os.path.exists(path) else None
    except Exception:
        return None


def load_all():
    """Return (models_dict, thresholds_dict, scaler, test_data)."""
    models = {}
    for name, fname in [("RandomForest", "rf_model.joblib"), ("XGBoost", "xgb_model.joblib"),
                        ("SVM", "svm_model.joblib"), ("LightGBM", "lightgbm_model.joblib"),
                        ("CatBoost", "catboost_model.joblib"), ("Stacking Ensemble", "stacking_ensemble_model.joblib")]:
        m = _safe_load(os.path.join(MODEL_DIR, fname))
        if m: models[name] = m

    for name, fname in [("IsolationForest", "isolationforest_model.joblib"),
                        ("OneClassSVM", "oneclasssvm_model.joblib"),
                        ("KMeans", "kmeans_model.joblib"), ("LOF", "lof_model.joblib")]:
        m = _safe_load(os.path.join(UNSUP_DIR, fname))
        if m: models[name] = m

    try:
        from tensorflow.keras.models import load_model as kload
        p = os.path.join(MODEL_DIR, "mlp_model.keras")
        if os.path.exists(p): models["MLP"] = kload(p)
        p = os.path.join(UNSUP_DIR, "autoencoder_model.keras")
        if os.path.exists(p): models["Autoencoder"] = kload(p)
    except Exception:
        pass

    thresholds = {}
    tp = os.path.join(UNSUP_DIR, "unsupervised_thresholds.json")
    if os.path.exists(tp):
        with open(tp) as f:
            thresholds = json.load(f)

    scaler = _safe_load(os.path.join(MODEL_DIR, "scaler.joblib"))
    test_data = _safe_load(os.path.join(MODEL_DIR, "test_data.joblib"))

    return models, thresholds, scaler, test_data


def predict(model, name, X, thresholds, threshold_mult=1.0):
    """Return (predictions, scores) for a single model."""
    if name == "MLP":
        sc = model.predict(X, verbose=0).flatten()
        pr = (sc >= 0.5).astype(int)
    elif name == "Autoencoder":
        recon = model.predict(X, verbose=0)
        sc = np.mean(np.power(X - recon, 2), axis=1)
        t = thresholds.get("Autoencoder", np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name in ("IsolationForest", "LOF"):
        sc = -model.score_samples(X)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name == "OneClassSVM":
        sc = -model.decision_function(X)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name == "KMeans":
        sc = np.min(model.transform(X), axis=1)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    else:
        pr = model.predict(X)
        sc = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pr.astype(float)
    return pr, sc


def load_metrics():
    metrics = {}
    p1 = os.path.join(ROOT, "reports", "metrics", "final_comparison.json")
    p2 = os.path.join(ROOT, "reports", "metrics", "adv_metrics.json")
    if os.path.exists(p1):
        with open(p1) as f:
            metrics.update(json.load(f))
    if os.path.exists(p2):
        with open(p2) as f:
            metrics.update(json.load(f))
    return metrics
