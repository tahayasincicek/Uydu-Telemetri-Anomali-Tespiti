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

# Sıralı (sequence) Keras modelleri 3B girdi bekler: (örnek, özellik, 1)
SEQUENCE_MODELS = {"LSTM", "BiLSTM", "GRU", "BiGRU", "CNN1D", "CNN_LSTM", "CNN_BiLSTM",
                   "CNN_GRU", "Transformer", "TCN", "Attention_BiLSTM",
                   "FCN", "ResNet1D", "InceptionTime", "LSTM_FCN"}
# Tek tip API'li (decision_function + predict) PyOD dedektörleri
PYOD_MODELS = {"ECOD", "COPOD", "HBOS", "CBLOF",
               "ABOD", "COF", "SOD", "SOS", "LODA", "INNE", "LMDD",
               "SO_GAAL", "MO_GAAL", "DeepSVDD", "LUNAR", "DIF", "XGBOD"}


def _safe_load(path):
    try:
        return joblib.load(path) if os.path.exists(path) else None
    except Exception:
        return None


def _kload(path):
    """Keras modelini güvenli şekilde yükler (VAE'nin Lambda katmanı için safe_mode=False)."""
    from tensorflow.keras.models import load_model
    return load_model(path, compile=False, safe_mode=False)


def load_all():
    """Return (models_dict, thresholds_dict, scaler, test_data)."""
    models = {}
    for name, fname in [("RandomForest", "rf_model.joblib"), ("XGBoost", "xgb_model.joblib"),
                        ("SVM", "svm_model.joblib"), ("LightGBM", "lightgbm_model.joblib"),
                        ("CatBoost", "catboost_model.joblib"), ("Stacking Ensemble", "stacking_ensemble_model.joblib"),
                        ("ExtraTrees", "extratrees_model.joblib"),
                        ("GradientBoosting", "gradientboosting_model.joblib"),
                        ("HistGradientBoosting", "histgradientboosting_model.joblib"),
                        ("AdaBoost", "adaboost_model.joblib"), ("KNN", "knn_model.joblib"),
                        ("LogisticRegression", "logisticregression_model.joblib"),
                        ("DecisionTree", "decisiontree_model.joblib"),
                        ("NaiveBayes", "naivebayes_model.joblib"),
                        ("Voting Ensemble", "voting_ensemble_model.joblib"),
                        ("LDA", "lda_model.joblib"),
                        ("QDA", "qda_model.joblib"),
                        ("Bagging", "bagging_model.joblib"),
                        ("Ridge", "ridge_model.joblib"),
                        ("SGD", "sgd_model.joblib"),
                        ("LSVC", "lsvc_model.joblib"),
                        ("XGBOD", "xgbod_model.joblib"),
                        ("MLP", "mlp_sklearn_model.joblib")]:
        m = _safe_load(os.path.join(MODEL_DIR, fname))
        if m: models[name] = m

    for name, fname in [("IsolationForest", "isolationforest_model.joblib"),
                        ("OneClassSVM", "oneclasssvm_model.joblib"),
                        ("KMeans", "kmeans_model.joblib"), ("LOF", "lof_model.joblib"),
                        ("GMM", "gmm_model.joblib"), ("EllipticEnvelope", "ellipticenvelope_model.joblib"),
                        ("PCA", "pca_model.joblib"), ("DBSCAN", "dbscan_model.joblib"),
                        ("ECOD", "ecod_model.joblib"), ("COPOD", "copod_model.joblib"),
                        ("HBOS", "hbos_model.joblib"), ("CBLOF", "cblof_model.joblib"),
                        ("ABOD", "abod_model.joblib"), ("COF", "cof_model.joblib"),
                        ("SOD", "sod_model.joblib"), ("SOS", "sos_model.joblib"),
                        ("LODA", "loda_model.joblib"), ("INNE", "inne_model.joblib"),
                        ("LMDD", "lmdd_model.joblib"),
                        ("SO_GAAL", "so_gaal_model.joblib"), ("MO_GAAL", "mo_gaal_model.joblib"),
                        ("DeepSVDD", "deepsvdd_model.joblib"),
                        ("LUNAR", "lunar_model.joblib"), ("DIF", "dif_model.joblib")]:
        m = _safe_load(os.path.join(UNSUP_DIR, fname))
        if m is not None: models[name] = m

    try:
        # MLP + derin sıralı/hibrit ağlar
        for name, fname in [("MLP", "mlp_model.keras"), ("LSTM", "lstm_model.keras"),
                            ("BiLSTM", "bilstm_model.keras"), ("GRU", "gru_model.keras"),
                            ("BiGRU", "bigru_model.keras"), ("CNN1D", "cnn1d_model.keras"),
                            ("CNN_LSTM", "cnn_lstm_model.keras"),
                            ("CNN_BiLSTM", "cnn_bilstm_model.keras"), ("CNN_GRU", "cnn_gru_model.keras"),
                            ("Transformer", "transformer_model.keras"), ("TCN", "tcn_model.keras"),
                            ("Attention_BiLSTM", "attention_bilstm_model.keras"),
                            ("FCN", "fcn_model.keras"), ("ResNet1D", "resnet1d_model.keras"),
                            ("InceptionTime", "inceptiontime_model.keras"), ("LSTM_FCN", "lstm_fcn_model.keras")]:
            p = os.path.join(MODEL_DIR, fname)
            if os.path.exists(p): models[name] = _kload(p)
        for name, fname in [("Autoencoder", "autoencoder_model.keras"), ("VAE", "vae_model.keras"),
                            ("AnoGAN", "anogan_model.keras"), ("ALAD", "alad_model.keras")]:
            p = os.path.join(UNSUP_DIR, fname)
            if os.path.exists(p): models[name] = _kload(p)
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
    if name in SEQUENCE_MODELS:
        # Sıralı modeller 3B girdi bekler: (örnek, özellik, 1)
        X_seq = np.asarray(X, dtype="float32").reshape((X.shape[0], X.shape[1], 1))
        sc = model.predict(X_seq, verbose=0).flatten()
        pr = (sc >= 0.5).astype(int)
    elif name == "MLP":
        if hasattr(model, "predict_proba"):
            # sklearn MLPClassifier
            pr = model.predict(X)
            sc = model.predict_proba(X)[:, 1]
        else:
            # Keras MLP
            sc = model.predict(X, verbose=0).flatten()
            pr = (sc >= 0.5).astype(int)
    elif name in PYOD_MODELS:
        sc = model.decision_function(X)
        pr = model.predict(X)
    elif name in ("Autoencoder", "VAE", "AnoGAN", "ALAD"):
        recon = model.predict(X, verbose=0)
        sc = np.mean(np.power(X - recon, 2), axis=1)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name == "PCA":
        recon = model.inverse_transform(model.transform(X))
        sc = np.mean(np.power(X - recon, 2), axis=1)
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif name in ("IsolationForest", "LOF", "GMM", "EllipticEnvelope"):
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
    elif name == "DBSCAN":
        sc = model.kneighbors(X)[0].ravel()  # en yakın çekirdek noktaya uzaklık
        t = thresholds.get(name, np.percentile(sc, 90)) * threshold_mult
        pr = (sc > t).astype(int)
    elif hasattr(model, "predict_proba"):
        pr = model.predict(X)
        sc = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        pr = model.predict(X)            # Ridge gibi olasılık vermeyen modeller
        sc = model.decision_function(X)
    else:
        pr = model.predict(X)
        sc = pr.astype(float)
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
