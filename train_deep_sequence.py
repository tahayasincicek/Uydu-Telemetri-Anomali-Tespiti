"""Derin sıralı modelleri HAM telemetri sinyali üzerinde eğitir (öncelikli alt küme).

18 elle-üretilmiş ESA özelliği yerine, her segmentin ham `value` dizisini
(data/raw/segments.csv) kullanır. Böylece LSTM/GRU/Transformer/TCN/CNN1D
gerçek zamansal dinamiği modeller (metodolojik olarak doğru girdi).

Akış:
  - Per-kanal standardizasyon (ortalama/std yalnız train=1 örneklerinden).
  - Her segment zaman-sıralı, L=256'ya pad/truncate edilir -> (n, L).
  - Resmi split: train=1 -> T (eğitim havuzu), train=0 -> Ψ (resmi test).
  - T'den 85/15 stratified train/val (seed 42, train_all_models.py ile aynı).
  - SupervisedAnomalyDetector'ın mevcut train_* metotları (girdi uzunluğunu
    veriden alır) yeniden kullanılır; model kodu değişmez.

Çıktı:
  - reports/metrics/deep_sequence_comparison.json  (5 model, 7 zorunlu metrik)
  - models/deep_sequence/<ad>_model.keras
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from metrics import compute_metrics, metrics_table, BENCHMARK_METRICS  # noqa: E402
from models.supervised import SupervisedAnomalyDetector  # noqa: E402

SEG_PATH = os.path.join(ROOT, "data", "raw", "segments.csv")
OUT_DIR = os.path.join(ROOT, "models", "deep_sequence")
METRICS_PATH = os.path.join(ROOT, "reports", "metrics", "deep_sequence_comparison.json")
SEQ_LEN = 256
RANDOM_STATE = 42
EPOCHS = 40
BATCH = 32

# Eğitilecek öncelikli alt küme (mimari aileleri temsil eder)
PRIORITY = ["LSTM", "GRU", "Transformer", "CNN1D", "TCN"]

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)


def build_sequences():
    """segments.csv -> (X, y, train_mask). X: (n_seg, SEQ_LEN) per-kanal std + pad."""
    df = pd.read_csv(SEG_PATH)
    df = df.sort_values(["segment", "timestamp"], kind="stable")

    # Per-kanal standardizasyon: ortalama/std yalnız train=1 örneklerinden.
    stats = (df[df["train"] == 1]
             .groupby("channel")["value"].agg(["mean", "std"]))
    g_mean = df.loc[df["train"] == 1, "value"].mean()
    g_std = df.loc[df["train"] == 1, "value"].std() or 1.0

    def zstd(row_ch, vals):
        if row_ch in stats.index:
            mu = stats.loc[row_ch, "mean"]; sd = stats.loc[row_ch, "std"]
        else:
            mu, sd = g_mean, g_std
        sd = sd if sd and sd > 1e-12 else 1.0
        return (vals - mu) / sd

    seg_meta = (df.groupby("segment")
                  .agg(channel=("channel", "first"),
                       anomaly=("anomaly", "first"),
                       train=("train", "first"))
                  .reset_index()
                  .sort_values("segment"))

    n = len(seg_meta)
    X = np.zeros((n, SEQ_LEN), dtype="float32")
    grouped = {sid: grp["value"].to_numpy() for sid, grp in df.groupby("segment")}

    for i, r in enumerate(seg_meta.itertuples(index=False)):
        vals = zstd(r.channel, grouped[r.segment].astype("float64"))
        vals = vals[:SEQ_LEN]                     # uzunları kes (ilk SEQ_LEN)
        X[i, :len(vals)] = vals                   # kısaları 0 ile post-pad et

    y = seg_meta["anomaly"].to_numpy().astype(int)
    train_mask = seg_meta["train"].to_numpy() == 1
    return X, y, train_mask


def main():
    print("=" * 64)
    print("  DERIN SIRALI MODELLER — HAM SINYAL (resmi split, segments.csv)")
    print("=" * 64)
    X, y, tr_mask = build_sequences()
    X_pool, y_pool = X[tr_mask], y[tr_mask]      # T = 1594
    X_test, y_test = X[~tr_mask], y[~tr_mask]    # Ψ = 529
    print(f"Diziler: L={SEQ_LEN} | T={X_pool.shape} Ψ={X_test.shape} "
          f"| Ψ anomali oranı={y_test.mean():.3f}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_pool, y_pool, test_size=0.15, random_state=RANDOM_STATE, stratify=y_pool)

    det = SupervisedAnomalyDetector(random_state=RANDOM_STATE)
    trainers = {
        "LSTM": det.train_lstm, "GRU": det.train_gru,
        "Transformer": det.train_transformer, "CNN1D": det.train_cnn1d,
        "TCN": det.train_tcn,
    }

    all_metrics = {}
    for name in PRIORITY:
        print("\n" + "-" * 64 + f"\n  {name} (ham sinyal)\n" + "-" * 64)
        try:
            t0 = time.time()
            trainers[name](X_tr, y_tr, X_val, y_val, epochs=EPOCHS, batch_size=BATCH)
            model = det.models[name]
            prob = model.predict(det._reshape_seq(X_test), verbose=0).ravel()
            pred = (prob >= 0.5).astype(int)
            inf_ms = (time.time() - t0) * 1000 / len(X_test)
            m = compute_metrics(y_test, pred, prob, inf_time_ms=inf_ms)
            all_metrics[name] = m
            det.save_model(name, os.path.join(OUT_DIR, f"{name.lower()}_model.keras"))
            print(f"  {name}: AUC_PR={m['AUC_PR']:.4f}  F1={m['F1']:.4f}  "
                  f"MCC={m['MCC']:.4f}  AUC_ROC={m['AUC_ROC']:.4f}")
        except Exception as e:
            print(f"  {name}: ATLANDI ({type(e).__name__}: {e})")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\n" + "=" * 64)
    print(f"  {len(all_metrics)} derin sıralı model (ham sinyal) eğitildi.")
    print(f"  Metrikler: {METRICS_PATH}")
    print(f"  Modeller : {OUT_DIR}")
    if all_metrics:
        tbl = metrics_table(all_metrics, sort_by="AUC_PR")
        print("\n" + tbl[BENCHMARK_METRICS].round(4).to_string())


if __name__ == "__main__":
    main()
