"""
Sentetik Uydu Telemetri Verisi Ureticisi
=========================================

ESA OPS-SAT manyetometre ve fotodiyot kanallarini taklit eden
gercekci sentetik telemetri segmentleri uretir.

Referans kanal profilleri gercek OPSSAT-AD verisinden cikarilmistir.

Kullanim:
    from src.synthetic_generator import SyntheticTelemetryGenerator
    gen = SyntheticTelemetryGenerator(seed=42)
    segments_df = gen.generate(n_segments=500, anomaly_ratio=0.20)
    segments_df.to_csv("sentetik_segments.csv", index=False)

    # Ardindan feature extraction:
    from src.feature_engineer import extract_esa_features
    dataset_df = extract_esa_features(segments_df)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════
#  Gercek veriden cikarilmis kanal profilleri
# ═══════════════════════════════════════════════════════════

CHANNEL_PROFILES = {
    # ── Manyetometre kanallari (cok kucuk genlikli sinyal) ──
    "CADC0872": {
        "type": "magnetometer",
        "signal_mean": 9.84e-07, "signal_std": 2.15e-05,
        "signal_min": -8.80e-05, "signal_max": 6.52e-05,
        "sampling_options": [1, 5], "sampling_weights": [0.78, 0.22],
        "len_mean": 122, "len_std": 115, "len_min": 14, "len_max": 687,
        "weight": 0.26,  # gercek veri orani
    },
    "CADC0873": {
        "type": "magnetometer",
        "signal_mean": -1.13e-06, "signal_std": 2.09e-05,
        "signal_min": -1.01e-04, "signal_max": 5.95e-05,
        "sampling_options": [1, 5], "sampling_weights": [0.77, 0.23],
        "len_mean": 115, "len_std": 111, "len_min": 12, "len_max": 673,
        "weight": 0.28,
    },
    "CADC0874": {
        "type": "magnetometer",
        "signal_mean": 3.23e-06, "signal_std": 2.01e-05,
        "signal_min": -4.83e-05, "signal_max": 1.01e-04,
        "sampling_options": [1, 5], "sampling_weights": [0.88, 0.12],
        "len_mean": 303, "len_std": 256, "len_min": 37, "len_max": 1040,
        "weight": 0.09,
    },
    # ── Fotodiyot kanallari (0 − pi/2 araligi) ──
    "CADC0884": {
        "type": "photodiode",
        "signal_mean": 4.71e-01, "signal_std": 3.91e-01,
        "signal_min": 0.0, "signal_max": 1.5708,
        "sampling_options": [5], "sampling_weights": [1.0],
        "len_mean": 75, "len_std": 39, "len_min": 17, "len_max": 173,
        "weight": 0.07,
    },
    "CADC0888": {
        "type": "photodiode",
        "signal_mean": 2.65e-01, "signal_std": 3.38e-01,
        "signal_min": 0.0, "signal_max": 1.3019,
        "sampling_options": [5], "sampling_weights": [1.0],
        "len_mean": 46, "len_std": 22, "len_min": 8, "len_max": 97,
        "weight": 0.12,
    },
    "CADC0892": {
        "type": "photodiode",
        "signal_mean": 2.55e-01, "signal_std": 3.31e-01,
        "signal_min": 0.0, "signal_max": 1.5708,
        "sampling_options": [1, 5], "sampling_weights": [0.98, 0.02],
        "len_mean": 236, "len_std": 93, "len_min": 27, "len_max": 470,
        "weight": 0.10,
    },
    "CADC0894": {
        "type": "photodiode",
        "signal_mean": 1.14e-01, "signal_std": 2.09e-01,
        "signal_min": 0.0, "signal_max": 1.4003,
        "sampling_options": [1, 5], "sampling_weights": [0.90, 0.10],
        "len_mean": 247, "len_std": 230, "len_min": 21, "len_max": 827,
        "weight": 0.07,
    },
}

# Dusuk ornekli kanallar (gercek veride az segment) dahil edilebilir
# CADC0886 (11 seg), CADC0890 (14 seg) — cok az, weight ~0.01

ANOMALY_TYPES = ["spike", "shift", "noise", "gap", "flat", "deformation"]


class SyntheticTelemetryGenerator:
    """ESA OPS-SAT benzeri sentetik uydu telemetri verisi uretir.

    Iki kanal ailesi desteklenir:
      - **Manyetometre** (CADC0872/0873/0874): ~1e-5 genlikli dusuk frekanslı
        sinyal, yavas rastgele yuruyus (random walk) + kucuk titresim.
      - **Fotodiyot** (CADC0884/0888/0892/0894): 0 − pi/2 arasinda duzgun
        yukselen/alcalan egriler, genellikle sifirdan baslar.

    Anomali turleri:
      - **spike**       : Ani sivri tepe (hem yukari hem asagi)
      - **shift**       : Sinyal seviyesinde kalici kayma
      - **noise**       : Belirli bolge icin gurultu varyansinda artis
      - **gap**         : Zaman damgasinda bosluk (eksik okumalar)
      - **flat**        : Sinyalin sabit/duz kalmasi
      - **deformation** : Sinusoidal bozulma enjeksiyonu
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.base_time = datetime(2023, 1, 1, 0, 0, 0)

    # ───────────────────────────────────────────────
    #  Normal sinyal ureticileri
    # ───────────────────────────────────────────────

    def _gen_magnetometer(self, n: int, profile: dict) -> np.ndarray:
        """Manyetometre kanalı icin gercekci sinyal uret.

        Gercek OPS-SAT manyetometre sinyali cok duzgun bir egridir:
        yavas monoton trend + cok kucuk olcum gurultusu.
        Cogu segmentte sadece 1 peak bulunur (10% prominence ile).
        """
        # Gercek sinyal neredeyse monoton bir egridir (lineer/kuadratik)
        t = np.linspace(0, 1, n)
        slope = self.rng.normal(0, profile["signal_std"] * 2)
        curve = self.rng.normal(0, profile["signal_std"] * 1)
        start = self.rng.normal(profile["signal_mean"], profile["signal_std"] * 0.5)
        trend = start + slope * t + curve * t ** 2

        # Cok kucuk olcum gurultusu (gercek veride diff_std ~ 1e-6)
        noise_std = profile["signal_std"] * 0.02
        noise = self.rng.normal(0, noise_std, size=n)
        signal = trend + noise

        signal = np.clip(signal, profile["signal_min"] * 1.2, profile["signal_max"] * 1.2)
        return signal

    def _gen_photodiode(self, n: int, profile: dict) -> np.ndarray:
        """Fotodiyot kanalı icin gercekci sinyal uret.

        Tipik desen: sifirdan baslar, monoton yukselir (veya tersi),
        ust sinir ~pi/2. Cok az gurultu — gercek veride sinyal
        neredeyse tamamen duzgun.
        """
        pattern = self.rng.choice(["rise", "fall", "plateau", "rise_fall"])

        t = np.linspace(0, 1, n)
        max_val = profile["signal_max"]
        # Gercek PD sinyali cok duzgun, gurultu cok dusuk
        noise_std = max_val * 0.005

        if pattern == "rise":
            base = max_val * t
        elif pattern == "fall":
            base = max_val * (1 - t)
        elif pattern == "plateau":
            level = self.rng.uniform(0, max_val)
            base = np.full(n, level)
        else:  # rise_fall
            base = max_val * np.sin(np.pi * t)

        signal = base + self.rng.normal(0, noise_std, size=n)
        signal = np.clip(signal, 0, max_val * 1.05)
        return signal

    def _generate_signal(self, n: int, profile: dict) -> np.ndarray:
        """Kanal tipine gore sinyal uret."""
        if profile["type"] == "magnetometer":
            return self._gen_magnetometer(n, profile)
        else:
            return self._gen_photodiode(n, profile)

    # ───────────────────────────────────────────────
    #  Anomali enjeksiyonu
    # ───────────────────────────────────────────────

    def _inject_spike(self, signal: np.ndarray, profile: dict) -> np.ndarray:
        """Ani sivri tepe enjekte et."""
        s = signal.copy()
        n_spikes = self.rng.integers(1, 4)
        for _ in range(n_spikes):
            pos = self.rng.integers(0, len(s))
            amplitude = profile["signal_std"] * self.rng.uniform(5, 15)
            direction = self.rng.choice([-1, 1])
            s[pos] += direction * amplitude
            # Yakin noktalari da hafifce etkilendir
            for offset in [-1, 1]:
                if 0 <= pos + offset < len(s):
                    s[pos + offset] += direction * amplitude * 0.3
        return s

    def _inject_shift(self, signal: np.ndarray, profile: dict) -> np.ndarray:
        """Sinyal seviyesinde kalici kayma."""
        s = signal.copy()
        shift_point = self.rng.integers(len(s) // 4, 3 * len(s) // 4)
        shift_amount = profile["signal_std"] * self.rng.uniform(3, 8) * self.rng.choice([-1, 1])
        s[shift_point:] += shift_amount
        return s

    def _inject_noise(self, signal: np.ndarray, profile: dict) -> np.ndarray:
        """Belirli bolge icin gurultu artisi."""
        s = signal.copy()
        start = self.rng.integers(0, max(1, len(s) // 2))
        end = self.rng.integers(start + len(s) // 4, len(s))
        noise_mult = self.rng.uniform(3, 10)
        s[start:end] += self.rng.normal(0, profile["signal_std"] * noise_mult, size=end - start)
        return s

    def _inject_flat(self, signal: np.ndarray, profile: dict) -> np.ndarray:
        """Sinyalin sabit kalmasi (sensor donmasi)."""
        s = signal.copy()
        start = self.rng.integers(0, max(1, len(s) // 2))
        length = self.rng.integers(len(s) // 5, len(s) // 2)
        end = min(start + length, len(s))
        flat_val = s[start]
        s[start:end] = flat_val
        return s

    def _inject_deformation(self, signal: np.ndarray, profile: dict) -> np.ndarray:
        """Sinusoidal bozulma enjeksiyonu."""
        s = signal.copy()
        freq = self.rng.uniform(0.5, 5.0)
        amp = profile["signal_std"] * self.rng.uniform(3, 8)
        t = np.linspace(0, 2 * np.pi * freq, len(s))
        s += amp * np.sin(t)
        return s

    def _inject_anomaly(self, signal: np.ndarray, profile: dict) -> Tuple[np.ndarray, str]:
        """Rastgele bir anomali turu sec ve enjekte et."""
        anomaly_type = self.rng.choice(ANOMALY_TYPES)
        injectors = {
            "spike": self._inject_spike,
            "shift": self._inject_shift,
            "noise": self._inject_noise,
            "flat": self._inject_flat,
            "deformation": self._inject_deformation,
        }
        # gap anomalisi timestamps'te yapilir, sinyal degismez
        if anomaly_type == "gap":
            return signal, "gap"
        return injectors[anomaly_type](signal, profile), anomaly_type

    # ───────────────────────────────────────────────
    #  Onboard artefakt enjeksiyonu
    # ───────────────────────────────────────────────

    def _apply_onboard_artifacts(self, signal: np.ndarray, timestamps: List[str],
                                 sampling: int, profile: dict,
                                 is_anomaly: bool) -> Tuple[np.ndarray, List[str]]:
        """Gercek uydu operasyonunda olusan sinyal artefaktlarini uygula.

        Bu artefaktlar anomali DEGILDIR — normal operasyon sirasinda da olusur.
        Anomali segmentlerinde daha yogun gorulur (gercek OPS-SAT verisinde oldugu gibi).

        Artefaktlar:
          1. Mikro-bosluk (2x expected interval): Tek okuma kaybi
          2. Sifir-zaman farki: Duplike timestamp
          3. Sifir-deger noktasi: Sensor sinyal kaybi
          4. Sabit-deger tekrari: Last-value-hold (sensor donmasi)
          5. Buyuk bosluk (nadir): Ciddi iletisim kesintisi
        """
        s = signal.copy()
        n = len(s)
        expected_sec = float(sampling)
        fmt = "%Y-%m-%dT%H:%M:%S.000Z"

        # ── Zaman damgasi artefaktlari: satirdan yeniden insa ──
        # Gercek OPS-SAT oranlari:
        #   Normal:  %22 bosluklu, %21 sifir-td
        #   Anomali: %51 bosluklu, %42 sifir-td
        apply_micro_gaps = False
        apply_zero_td = False

        if is_anomaly:
            apply_micro_gaps = self.rng.random() < 0.51
            apply_zero_td = self.rng.random() < 0.42
        else:
            apply_micro_gaps = self.rng.random() < 0.22
            apply_zero_td = self.rng.random() < 0.21

        # Mikro-bosluk ve sifir-td oranlari (segment ici)
        micro_gap_rate = self.rng.uniform(0.05, 0.25) if apply_micro_gaps else 0.0
        zero_td_rate = self.rng.uniform(0.01, 0.08) if apply_zero_td else 0.0

        # Buyuk bosluk pozisyonu (sadece anomali, %6 olasilik)
        big_gap_pos = -1
        big_gap_sec = 0.0
        if is_anomaly and self.rng.random() < 0.06:
            big_gap_pos = int(self.rng.integers(n // 4, max(n // 4 + 1, 3 * n // 4)))
            big_gap_sec = float(self.rng.choice(
                [self.rng.uniform(10, 30), self.rng.uniform(30, 130)],
                p=[0.7, 0.3]))

        # Timestamp dizisini yeniden olustur
        new_ts = [timestamps[0]]
        for i in range(1, n):
            prev_t = datetime.strptime(new_ts[-1], fmt)
            r = self.rng.random()

            if i == big_gap_pos:
                # Buyuk bosluk
                step = expected_sec + big_gap_sec
            elif r < zero_td_rate:
                # Duplike timestamp (td = 0)
                step = 0.0
            elif r < zero_td_rate + micro_gap_rate:
                # Mikro-bosluk (td = 2x beklenen, tek okuma kaybi)
                step = expected_sec * 2.0
            else:
                # Normal aralik
                step = expected_sec

            new_ts.append((prev_t + timedelta(seconds=step)).strftime(fmt))

        # ── Sinyal degeri artefaktlari ──

        # 3. Sifir-deger noktasi
        if profile["type"] == "magnetometer":
            # Manyetometre: arada sifir okumalar (%15 olasilik, anomalide %22)
            zero_prob = 0.22 if is_anomaly else 0.15
            if self.rng.random() < zero_prob:
                n_zeros = self.rng.integers(1, max(2, n // 8))
                zero_pos = self.rng.choice(n, size=min(n_zeros, n), replace=False)
                s[zero_pos] = 0.0
        else:
            # Fotodiyot: sifir bloklari cok yaygin (sensor golgede)
            if self.rng.random() < 0.40:
                block_start = self.rng.integers(0, max(1, n // 2))
                block_len = self.rng.integers(n // 5, max(n // 5 + 1, n // 2))
                block_end = min(block_start + block_len, n)
                s[block_start:block_end] = 0.0

        # 4. Sabit-deger tekrari (last-value-hold, sensor donmasi)
        stuck_prob = 0.18 if is_anomaly else 0.10
        if self.rng.random() < stuck_prob:
            run_start = self.rng.integers(0, max(1, n - 5))
            run_len = self.rng.integers(5, min(35, max(6, n - run_start)))
            s[run_start:run_start + run_len] = s[run_start]

        return s, new_ts

    # ───────────────────────────────────────────────
    #  Zaman damgasi ureticisi
    # ───────────────────────────────────────────────

    def _generate_timestamps(self, n: int, sampling: int,
                             inject_gap: bool = False) -> List[str]:
        """ISO formatinda zaman damgalari uret.

        inject_gap=True ise rastgele anomali bosluklari eklenir.
        """
        # sampling = zaman araligi (saniye), orn: 1 = 1Hz, 5 = her 5 saniyede bir
        interval = float(sampling)
        times = []
        current = self.base_time + timedelta(seconds=int(self.rng.integers(0, 86400)))

        for i in range(n):
            times.append(current.strftime("%Y-%m-%dT%H:%M:%S.000Z"))
            step = interval
            if inject_gap and self.rng.random() < 0.08:
                # Anomali kaynaklı bosluk (5-200 saniye)
                step = self.rng.uniform(5, 200)
            current += timedelta(seconds=step)

        return times

    # ───────────────────────────────────────────────
    #  Ana uretici
    # ───────────────────────────────────────────────

    def generate(self,
                 n_segments: int = 500,
                 anomaly_ratio: float = 0.20,
                 channels: Optional[List[str]] = None) -> pd.DataFrame:
        """Sentetik segments.csv formati uret.

        Args:
            n_segments: Uretilecek segment sayisi.
            anomaly_ratio: Anomali orani (0-1).
            channels: Kullanilacak kanal listesi (None = tum kanallar).

        Returns:
            DataFrame — segments.csv formatinda (channel, timestamp, value,
                         label, sampling, anomaly, segment, train).
        """
        if channels is None:
            channels = list(CHANNEL_PROFILES.keys())

        # Kanal basina segment sayisi (agirlikli)
        weights = np.array([CHANNEL_PROFILES[c]["weight"] for c in channels])
        weights /= weights.sum()
        n_per_channel = self.rng.multinomial(n_segments, weights)

        n_anomaly_total = int(n_segments * anomaly_ratio)
        anomaly_flags = np.zeros(n_segments, dtype=int)
        anomaly_idx = self.rng.choice(n_segments, size=n_anomaly_total, replace=False)
        anomaly_flags[anomaly_idx] = 1

        # Train/test split (%70 train)
        train_flags = np.zeros(n_segments, dtype=int)
        train_idx = self.rng.choice(n_segments, size=int(n_segments * 0.70), replace=False)
        train_flags[train_idx] = 1

        all_rows = []
        seg_counter = 0

        for ch_idx, ch_name in enumerate(channels):
            profile = CHANNEL_PROFILES[ch_name]
            n_ch = n_per_channel[ch_idx]

            for _ in range(n_ch):
                seg_id = seg_counter + 1
                is_anomaly = anomaly_flags[seg_counter]
                is_train = train_flags[seg_counter]

                # Segment uzunlugu
                seg_len = max(
                    profile["len_min"],
                    min(
                        profile["len_max"],
                        int(self.rng.normal(profile["len_mean"], profile["len_std"]))
                    )
                )

                # Sampling
                sampling = int(self.rng.choice(
                    profile["sampling_options"],
                    p=profile["sampling_weights"]
                ))

                # Sinyal uret
                signal = self._generate_signal(seg_len, profile)

                # Anomali enjekte et
                anomaly_type = None
                inject_gap = False
                if is_anomaly:
                    signal, anomaly_type = self._inject_anomaly(signal, profile)
                    if anomaly_type == "gap":
                        inject_gap = True

                # Zaman damgalari
                timestamps = self._generate_timestamps(seg_len, sampling,
                                                       inject_gap=inject_gap)

                # Onboard artefaktlar (anomali olmayan operasyonel bozulmalar)
                signal, timestamps = self._apply_onboard_artifacts(
                    signal, timestamps, sampling, profile, bool(is_anomaly))

                # Label
                label = "anomaly" if is_anomaly else "nominal"

                # Satirlari olustur
                for t_idx in range(seg_len):
                    all_rows.append({
                        "channel": ch_name,
                        "timestamp": timestamps[t_idx],
                        "value": signal[t_idx],
                        "label": label,
                        "sampling": sampling,
                        "anomaly": is_anomaly,
                        "segment": seg_id,
                        "train": is_train,
                    })

                seg_counter += 1

                if seg_counter % 100 == 0:
                    print(f"  [{seg_counter}/{n_segments}] segment uretildi...")

        df = pd.DataFrame(all_rows)
        print(f"Tamamlandi: {n_segments} segment, {n_anomaly_total} anomali, "
              f"{len(df)} satir uretildi.")
        return df

    def generate_and_extract(self,
                             n_segments: int = 500,
                             anomaly_ratio: float = 0.20,
                             channels: Optional[List[str]] = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sentetik veri uret VE ESA feature extraction uygula.

        Returns:
            (segments_df, dataset_df) — ham sinyal + 18 ozellikli tablo.
        """
        from feature_engineer import extract_esa_features

        segments_df = self.generate(n_segments, anomaly_ratio, channels)
        dataset_df = extract_esa_features(segments_df)
        return segments_df, dataset_df


# ═══════════════════════════════════════════════════════════
#  CLI Kullanim
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser(
        description="Sentetik uydu telemetri verisi uret")
    parser.add_argument("-n", "--n_segments", type=int, default=500,
                        help="Segment sayisi (default: 500)")
    parser.add_argument("-a", "--anomaly_ratio", type=float, default=0.20,
                        help="Anomali orani (default: 0.20)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Rastgelelik tohumu (default: 42)")
    parser.add_argument("-o", "--output_dir", type=str, default="data/synthetic",
                        help="Cikti klasoru (default: data/synthetic)")
    parser.add_argument("--extract", action="store_true",
                        help="Feature extraction da uygula")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gen = SyntheticTelemetryGenerator(seed=args.seed)
    print(f"Sentetik veri uretiliyor: {args.n_segments} segment, "
          f"anomali orani={args.anomaly_ratio}")

    if args.extract:
        seg_df, ds_df = gen.generate_and_extract(
            args.n_segments, args.anomaly_ratio)
        ds_path = os.path.join(args.output_dir, "synthetic_dataset.csv")
        ds_df.to_csv(ds_path, index=False)
        print(f"Feature tablosu: {ds_path}")
    else:
        seg_df = gen.generate(args.n_segments, args.anomaly_ratio)

    seg_path = os.path.join(args.output_dir, "synthetic_segments.csv")
    seg_df.to_csv(seg_path, index=False)
    print(f"Ham veri: {seg_path}")
    print(f"Toplam satir: {len(seg_df)}")
