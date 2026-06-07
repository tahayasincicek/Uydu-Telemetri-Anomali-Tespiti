# Uydu Telemetrisinde Anomali Tespiti: ESA OPS-SAT Kıyaslaması Üzerinde Sızıntısız, Tekrarüretilebilir Bir Makine Öğrenmesi Çalışması ve Profil-Temelli Sentetik Veri Katkısı

**Lisansüstü Tez Çalışması**

---

## Özet

Uydu telemetrisinde anomali tespiti, uzay araçlarının sağlık izleme (health monitoring)
sistemlerinin temel bileşenlerinden biridir; nadir, çeşitli ve önceden tam tanımlanamayan
anomalilerin sınırlı yer-istasyonu kaynaklarıyla yakalanmasını gerektirir. Bu tez, Avrupa Uzay
Ajansı'nın (ESA) OPS-SAT uydusundan toplanan ve kamuya açık hâle getirilen **OPSSAT-AD** kıyaslama
veri seti üzerinde, makine öğrenmesi temelli anomali tespitini **metodolojik olarak titiz,
tekrarüretilebilir ve veri sızıntısından (data leakage) arınmış** bir çerçevede ele alır.

Çalışmanın birincil amacı, literatürdeki tekrarüretilebilirlik krizine yanıt olarak, resmi
train/test bölmesine sadık (T = 1594, Ψ = 529 segment) ve yedi zorunlu metrikle (Accuracy,
Precision, Recall, F1, MCC, AUC-ROC, **AUC-PR**) raporlanan bir referans çizgisi (baseline)
kurmaktır. **42 model**lik bir havuz, tek bir kanonik eğitim motoruyla resmi test seti (Ψ)
üzerinde değerlendirilmiş; en iyi sonuç **ExtraTrees** ile **AUC-PR = 0.983** (AUC-ROC 0.994,
F1 0.931, MCC 0.914) olarak elde edilmiştir. Sonuçların, kıyaslamayı tanımlayan Ruszczak ve ark.
(2024) çalışmasının Tablo 3 değerleriyle gözetimli modellerde ortalama |ΔAUC-PR| = **0.004**
sapmayla yeniden üretildiği gösterilmiştir.

İkincil olarak, **profil-temelli sentetik telemetri üretimi** için üç aşamalı (ham akış →
segmentasyon → özellik çıkarımı) özgün bir boru hattı önerilmiş ve sentetik verinin gerçek test
setinde tespiti iyileştirip iyileştirmediği sistematik olarak ablasyonla incelenmiştir. Bulgu,
zengin-etiket rejiminde augmentasyonun **AUC-PR'ı anlamlı düzeyde değiştirmediği**, asıl etkisinin
**kesinlik–duyarlılık (precision–recall) dengesini kaydırmak** olduğu yönündedir. Bu nötr sonuç,
sentetik ve gerçek dağılımlar arasındaki orta düzey Kolmogorov–Smirnov mesafesiyle (ortalama
KS = 0.29) ve özellikle en ayırt edici özelliğin (`n_peaks`) sentetik tarafta en zayıf taklit
edilen özellik olmasıyla (KS = 0.52) açıklanmıştır. Ek olarak, 64 algoritmanın hesaplama
maliyeti/enerji profili çıkarılmış ve SHAP ile karar yorumlanabilirliği sağlanmıştır.

**Anahtar Kelimeler:** Uydu telemetrisi, anomali tespiti, OPS-SAT, makine öğrenmesi, veri
sızıntısı, tekrarüretilebilirlik, dengesiz sınıflandırma, AUC-PR, sentetik veri üretimi, veri
augmentasyonu, SHAP, yorumlanabilirlik.

---

## Abstract

Anomaly detection in satellite telemetry is a cornerstone of spacecraft health monitoring,
requiring the detection of rare, diverse and ill-defined anomalies under limited ground-station
resources. This thesis studies machine-learning–based anomaly detection on the publicly released
**OPSSAT-AD** benchmark — collected from the European Space Agency's OPS-SAT spacecraft — within a
**methodologically rigorous, reproducible and leakage-free** framework.

The primary goal is to establish a reference baseline that is faithful to the official train/test
split (T = 1594, Ψ = 529 segments) and reported with seven mandatory metrics (Accuracy, Precision,
Recall, F1, MCC, AUC-ROC, **AUC-PR**), responding to the reproducibility crisis in the literature.
A pool of **42 models** is evaluated on the held-out test set (Ψ) via a single canonical training
engine; the best result is achieved by **ExtraTrees** with **AUC-PR = 0.983** (AUC-ROC 0.994,
F1 0.931, MCC 0.914). The results reproduce the baseline of Ruszczak et al. (2024, Table 3) with a
mean |ΔAUC-PR| of **0.004** on supervised models.

Secondarily, a novel three-stage (raw stream → segmentation → feature extraction)
**profile-grounded synthetic telemetry generation** pipeline is proposed, and an ablation study
quantifies whether synthetic data improves detection on the real test set. The finding is that, in
a label-rich regime, augmentation does **not** significantly change AUC-PR; its main effect is to
**shift the precision–recall trade-off**. This neutral result is explained by a moderate
Kolmogorov–Smirnov distance between synthetic and real distributions (mean KS = 0.29), notably with
the most discriminative feature (`n_peaks`) being the worst-reproduced one (KS = 0.52). A
power/efficiency profile of 64 algorithms and SHAP-based interpretability complete the study.

**Keywords:** Satellite telemetry, anomaly detection, OPS-SAT, machine learning, data leakage,
reproducibility, imbalanced classification, AUC-PR, synthetic data generation, data augmentation,
SHAP, interpretability.

---

## İçindekiler

1. [Giriş](#1-giriş)
2. [Literatür ve Kuramsal Arka Plan](#2-literatür-ve-kuramsal-arka-plan)
3. [Veri Seti ve Ön İşleme](#3-veri-seti-ve-ön-i̇şleme)
4. [Yöntem](#4-yöntem)
5. [Deneysel Düzenek](#5-deneysel-düzenek)
6. [Bulgular](#6-bulgular)
7. [Özgün Katkılar](#7-özgün-katkılar)
8. [Tartışma](#8-tartışma)
9. [Sonuç ve Gelecek Çalışmalar](#9-sonuç-ve-gelecek-çalışmalar)
- [Kaynaklar](#kaynaklar)
- [Ek A: Notebook ve Modül Haritası](#ek-a-notebook-ve-modül-haritası)

---

## 1. Giriş

### 1.1 Problem Tanımı ve Motivasyon

Yörüngedeki bir uzay aracı, alt sistemlerinin (güç, yönelim kontrolü, termal, haberleşme) durumunu
sürekli olarak telemetri kanalları aracılığıyla yere bildirir. Bu kanallardaki **anomaliler** —
sensör arızaları, beklenmeyen rejim değişiklikleri, yazılım hataları veya çevresel etkiler —
zamanında tespit edilmediğinde görev kaybına kadar uzanan sonuçlar doğurabilir. Ancak anomali
tespiti uzay alanında üç temel güçlük taşır:

1. **Sınıf dengesizliği:** Nominal davranış baskındır; anomaliler azınlıktadır (bu çalışmada
   ~%20). Doğruluk (accuracy) gibi ölçütler bu rejimde yanıltıcıdır.
2. **Etiket kıtlığı ve çeşitlilik:** Anomaliler nadir, heterojen ve önceden tam tanımlanamazdır.
3. **Kaynak kısıtı:** Hem yer istasyonunda hem de uydu üzerinde (onboard) hesaplama, enerji ve
   bant genişliği sınırlıdır.

Bu güçlüklerin yanı sıra, makine öğrenmesi temelli yayınların önemli bir kısmı **tekrarüretilebilir
değildir**: rastgele bölmeler, veri sızıntısı (data leakage) ve tutarsız metrik raporlaması,
"ilerleme yanılsaması" yaratır (Kapoor ve Narayanan, 2023; Wu ve Keogh, 2022). Bu tez, ESA'nın
OPS-SAT kıyaslaması üzerinde, bu metodolojik tuzaklardan kaçınan **sağlam bir referans çizgisi**
inşa etmeyi ve bunun üzerine özgün katkılar koymayı amaçlar.

### 1.2 Araştırma Soruları

- **AS1.** OPSSAT-AD kıyaslamasında, resmi bölmeye sadık ve veri sızıntısından arınmış bir
  protokol altında klasik/topluluk/gözetimsiz modeller hangi performansa ulaşır ve bu sonuçlar
  özgün makaleyle ne ölçüde örtüşür (tekrarüretilebilirlik)?
- **AS2.** Anomali kararını hangi özellikler yönlendirir; özellik uzayı bilgi kaybı olmadan ne
  kadar küçültülebilir (ablasyon ve yorumlanabilirlik)?
- **AS3.** Gerçek telemetri istatistiklerine dayalı **profil-temelli sentetik veri** üretmek
  mümkün müdür ve bu veri, gerçek test setinde tespiti iyileştirir mi (sentetik augmentasyon)?
- **AS4.** Farklı dengeleme/augmentasyon stratejileri (SMOTE, sinyal-temelli ICCS-ω, sentetik)
  performansı ve operasyonel dengeleri nasıl etkiler?
- **AS5.** Algoritmaların hesaplama maliyeti ile doğruluğu arasındaki ödünleşim, onboard
  dağıtım için nasıl bir öneri haritası sunar?

### 1.3 Katkılar

Bu tezin katkıları üç fazda toplanmıştır:

- **Faz 1 — Metodolojik temel.** Resmi train/test bölmesi, yedi zorunlu metrik (AUC-PR'a göre
  sıralı), iki ayrı veri sızıntısı kaynağının giderilmesi ve tüm modelleri tek noktadan üreten
  **kanonik eğitim motoru**. 42 modellik bir havuz Ψ üzerinde değerlendirilmiştir.
- **Faz 2 — Kıyaslama hizalaması.** Sonuçların Ruszczak ve ark. (2024) Tablo 3 referansıyla
  **aynı test seti** üzerinde karşılaştırılması; gözetimli modellerde ortalama |ΔAUC-PR| = 0.004
  ile reprodüksiyon.
- **Faz 3 — Özgün katkılar.** (i) Üç aşamalı, profil-temelli sentetik telemetri üreteci; (ii)
  sentetik augmentasyon ablasyonu ve KS-temelli dağılım analizi; (iii) SMOTE/ICCS-ω/sentetik
  augmentasyon karşılaştırması; (iv) 64 algoritmanın güç/verimlilik profili; (v) SHAP ile
  yorumlanabilirlik.

### 1.4 Tez Organizasyonu

Bölüm 2 literatürü ve kuramsal arka planı; Bölüm 3 veri setini ve sızıntısız ön işlemeyi;
Bölüm 4 değerlendirme metriklerini ve model havuzunu; Bölüm 5 deneysel düzeneği; Bölüm 6 referans
çizgisi bulgularını; Bölüm 7 özgün katkıları; Bölüm 8 tartışmayı; Bölüm 9 sonuçları sunar.

---

## 2. Literatür ve Kuramsal Arka Plan

### 2.1 Uydu Telemetrisinde Anomali Tespiti

Uzay telemetrisinde anomali tespiti yaklaşımları geniş bir yelpazeye yayılır: eşik-temelli ve
istatistiksel yöntemler (out-of-limit kontrolleri), klasik makine öğrenmesi (ağaç toplulukları,
SVM), gözetimsiz aykırı değer (outlier) dedektörleri ve derin öğrenme temelli yöntemler
(otokodlayıcılar, LSTM/transformer mimarileri). Telemetri doğası gereği **çok kanallı zaman
serisi** olduğundan, yaklaşımlar ya ham sinyali doğrudan modeller ya da bu çalışmada olduğu gibi
**segment-temelli özellik çıkarımı** ile sabit boyutlu öznitelik vektörlerine indirger.

### 2.2 ESA OPS-SAT Kıyaslaması

OPS-SAT, ESA'nın "uçan yazılım laboratuvarı" olarak tasarlanmış bir CubeSat görevidir. Ruszczak
ve ark. (2023, 2024), bu uydudan toplanan telemetriyi etiketleyerek kamuya açık bir anomali tespit
kıyaslaması (**OPSSAT-AD**) oluşturmuştur. Kıyaslamanın iki temel katkısı vardır: (i) telemetri
segmentlerinden çıkarılmış **18 handcrafted özellik**, ve (ii) tekrarüretilebilirlik için sabit bir
**train/test bölmesi**. Bu tez, kıyaslamanın hem özellik tanımına hem de bölmesine birebir sadık
kalarak literatürle doğrudan kıyaslanabilir sonuçlar üretir.

### 2.3 Tekrarüretilebilirlik Krizi ve Veri Sızıntısı

Makine öğrenmesi temelli bilimde **veri sızıntısı**, eğitim sürecine test bilgisinin sızması
sonucu performansın yapay biçimde şişmesidir (Kapoor ve Narayanan, 2023). Zaman serisi anomali
tespitinde bu sorun özellikle yaygındır (Wu ve Keogh, 2022). İki klasik sızıntı kaynağı:

1. **Dengeleme (ör. SMOTE) bölme öncesi uygulanması:** Test noktalarından türeyen sentetik
   örneklerin eğitime sızması.
2. **Ölçekleyicinin (scaler) tüm veride fit edilmesi:** Test dağılım istatistiklerinin eğitime
   sızması.

Bu tez, her iki kaynağı da açıkça giderir (bkz. Bölüm 3.5).

### 2.4 Dengesiz Sınıflandırmada Metrikler

Sınıf dengesizliğinde doğruluk yanıltıcıdır; literatür **Matthews Korelasyon Katsayısı (MCC)**'nı
(Chicco ve Jurman, 2020) ve azınlık sınıfına duyarlı **AUC-PR**'ı önerir. ROC eğrisi altı alan
(AUC-ROC) eşikten bağımsız ayrışmayı ölçerken, dengesiz veride **AUC-PR** birincil ölçüt olarak
tercih edilir. Bu çalışma yedi metriği birlikte raporlar ve tabloları AUC-PR'a göre sıralar.

---

## 3. Veri Seti ve Ön İşleme

### 3.1 OPSSAT-AD Veri Seti

Veri seti iki dosyadan oluşur:

| Dosya | Boyut | Rol |
|-------|:-----:|-----|
| `segments.csv` | 303.493 satır | **Ham telemetri** — her satır tek bir zaman noktasında tek bir sensör ölçümü |
| `dataset.csv` | 2.123 satır | **Özellik matrisi** — her satır bir segmentin 18 özellikli özeti |

303.493 ham ölçüm, **2.123 segmente** bölünmüş (ortalama ~143 ölçüm/segment) ve her segment için
18 ESA özelliği çıkarılmıştır. Her segment tek bir kanala aittir.

### 3.2 Kanallar ve Sinyal Karakteristiği

Veri **9 kanal** içerir; bunlar iki fiziksel sensör tipine ayrılır:

| Tip | Kanallar | Karakter |
|-----|----------|----------|
| Manyetometre | CADC0872/0873/0874 | ~10⁻⁵ genlikli, yavaş değişen, neredeyse monoton sinyal |
| Fotodiyot | CADC0884–0894 (6 kanal) | 0–π/2 açı; yörünge gölge-aydınlık döngüsüyle çoğunlukla sıfıra yakın |

Bu iki rejimin istatistiksel farklılığı (manyetometrenin durağanlığa yakınlığı vs fotodiyotun
periyodik, düşük-değer eğilimli yapısı) hem özellik mühendisliğini hem de sentetik üretimi
(Bölüm 7.1) doğrudan etkiler.

### 3.3 On Sekiz ESA Handcrafted Özellik

Özellikler üç mantıksal gruba ayrılır:

- **Grup 1 — Ham segmentten (12):** `mean`, `var`, `std`, `kurtosis`, `skew`, `n_peaks`
  (%10 prominence ile tepe sayısı), `duration`, `len`, `gaps_squared` (Σ Δt²), `len_weighted`
  (len × sampling), `var_div_duration`, `var_div_len`.
- **Grup 2 — Yumuşatılmış segmentten (2):** `smooth10_n_peaks`, `smooth20_n_peaks` (10 ve 20
  noktalı hareketli ortalama sonrası tepe sayısı).
- **Grup 3 — Türevlerden (4):** `diff_peaks`, `diff2_peaks` (1. ve 2. türev tepe sayısı),
  `diff_var`, `diff2_var` (1. ve 2. türev varyansı).

Karşılaştırılabilirlik adına **yalnızca bu 18 özellik** kullanılır; meta sütunlar (`channel`,
`sampling`) ve kanal kimliği öznitelik olarak dahil edilmez. Otomatik özellik çıkarım hattı,
ESA'nın orijinal `dataset.csv`'sini **r ≈ 0.999** korelasyonla yeniden üretir (yalnız
`smooth20_n_peaks` r = 0.982; konvolüsyon kenar etkisi).

### 3.4 Resmi Train/Test Bölmesi (Ψ)

Tekrarüretilebilirlik için kıyaslama, `dataset.csv` içindeki `train` kolonunda kodlanmış sabit bir
bölme tanımlar:

| Küme | Segment | Nominal | Anomali | Anomali Oranı |
|------|:------:|:------:|:-------:|:-------------:|
| **T** (eğitim, `train=1`) | 1.594 | 1.273 | 321 | %20,1 |
| **Ψ** (test, `train=0`) | 529 | 416 | 113 | %21,4 |
| **Toplam** | 2.123 | 1.689 | 434 | %20,4 |

Tüm modeller **aynı Ψ** üzerinde değerlendirilir. Kendi rastgele bölmesini kullanmak makaleyle
kıyaslanabilirliği ortadan kaldıracağından, pipeline boyunca yalnızca resmi `train` kolonu
kullanılmıştır.

### 3.5 Veri Sızıntısından Arınmış Ön İşleme

İki sızıntı kaynağı açıkça giderilmiştir:

- **(a) SMOTE yalnızca eğitim katmanına (T) uygulanır;** Ψ test seti asla dengelenmez/dokunulmaz,
  gerçek %21,4 anomali oranı korunur.
- **(b) Ölçekleyici (StandardScaler/RobustScaler) yalnızca T üzerinde fit edilir;** Ψ'ye yalnızca
  `transform` uygulanır.

Sızıntısız akış şu şekildedir:

```
dataset.csv → resmi bölme (T / Ψ)
            → preprocessor.fit(T) → transform(T), transform(Ψ)
            → SMOTE(yalnız T)     → T'den doğrulama (validation) ayır
            → X_train (SMOTE'lu T), X_val, X_test (= dokunulmamış Ψ)
```

---

## 4. Yöntem

### 4.1 Değerlendirme Metrikleri

Tüm modeller **yedi zorunlu metrikle** raporlanır ve tablolar birincil ölçüt **AUC-PR**'a göre
sıralanır:

| Metrik | Tanım | Not |
|--------|-------|-----|
| Accuracy | (TP+TN)/Toplam | Dengesiz veride yanıltıcı |
| Precision | TP/(TP+FP) | Yanlış alarm kontrolü |
| Recall | TP/(TP+FN) | Kaçırma kontrolü |
| F1 | 2·P·R/(P+R) | P–R harmonik ortalaması |
| **MCC** | Matthews Korelasyon Katsayısı | Dengesiz sınıflandırmada tercih edilen |
| **AUC-ROC** | ROC eğrisi altı alan | Eşikten bağımsız ayrışma |
| **AUC-PR** | Precision–Recall eğrisi altı alan | **Birincil sıralama ölçütü** |

Operasyonel ek ölçütler olarak yanlış alarm oranı (FAR), kaçırma oranı (FNR) ve çıkarım süresi
(Inf.Time) da hesaplanır.

### 4.2 Model Havuzu

42 modellik kanonik karşılaştırma üç paradigmayı kapsar:

- **Gözetimli (klasik/topluluk):** RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost,
  HistGradientBoosting, GradientBoosting, AdaBoost, Bagging, Voting/Stacking topluluk, SVM, LSVC,
  KNN, LogisticRegression, Ridge, SGD, DecisionTree, NaiveBayes, LDA, QDA, MLP, XGBOD.
- **Gözetimsiz dedektörler:** IsolationForest, OneClassSVM, KMeans, LOF, GMM, EllipticEnvelope,
  PCA, DBSCAN ve PyOD ailesi (ECOD, COPOD, HBOS, CBLOF, ABOD, COF, SOD, SOS, LODA, INNE, LMDD).
  Bunlar **yalnız nominal örneklerde** eğitilir; karar eşiği doğrulama setinde seçilir.
- **Derin sıralı ağlar (opsiyonel, 15 mimari):** LSTM, BiLSTM, GRU, BiGRU, CNN1D, CNN-LSTM,
  CNN-BiLSTM, CNN-GRU, Transformer, TCN, Attention-BiLSTM, FCN, ResNet1D, InceptionTime, LSTM-FCN.

### 4.3 Kanonik Eğitim Motoru

Tüm modeller, tek bir kanonik kaynak (`train_all_models.py`) tarafından **aynı 18 özellik, aynı
resmi bölme ve yalnız-T'de-fit edilen ölçekleyici** ile eğitilir. Bu motor, otoriter sonuç
tablosunu (`final_comparison.json`, 7 metrik, AUC-PR sıralı), eğitilmiş modelleri, kanonik
ölçekleyiciyi ve dokunulmamış Ψ test verisini üretir. Notebook'lar bu çıktıyı **tüketir**; kendi
bölmesini/modelini üretip kanonik artefaktları ezmez. Bu tasarım, "iki ayrı boru hattının birbirini
ezmesi" sorununu yapısal olarak ortadan kaldırır.

---

## 5. Deneysel Düzenek

- **Yazılım:** Python 3.11; scikit-learn, imbalanced-learn, XGBoost, LightGBM, CatBoost, PyOD,
  TensorFlow/Keras, SHAP, pandas/NumPy/SciPy.
- **Donanım:** CPU-temelli (TensorFlow CPU; CUDA devre dışı), tek-işlem.
- **Protokol:** Resmi bölme → sızıntısız ön işleme → eğitim → Ψ'de 7 metrik. Gözetimsiz modellerde
  eşik seçimi yalnız doğrulama setinde yapılır.
- **Tekrarüretilebilirlik:** Sabit rastgelelik tohumu (seed = 42); kanonik motor deterministik
  çıktı üretir. Çalışma 14 Jupyter notebook + kanonik eğitim betiği ile baştan üretilebilir
  (bkz. Ek A).

---

## 6. Bulgular

### 6.1 Kanonik Model Sıralaması (Faz 1)

Resmi test seti Ψ (529 segment) üzerinde, 42 model, AUC-PR'a göre sıralı. İlk 15:

| # | Model | AUC-PR | AUC-ROC | F1 | MCC | Precision | Recall |
|--:|-------|:------:|:-------:|:--:|:---:|:---------:|:------:|
| 1 | ExtraTrees | **0,983** | 0,994 | 0,931 | 0,914 | 0,971 | 0,894 |
| 2 | Voting Ensemble | 0,980 | 0,994 | 0,920 | 0,903 | 0,980 | 0,867 |
| 3 | MLP | 0,979 | 0,990 | **0,946** | **0,932** | 0,963 | **0,929** |
| 4 | HistGradientBoosting | 0,974 | 0,990 | 0,923 | 0,903 | 0,944 | 0,903 |
| 5 | XGBOD | 0,973 | 0,991 | 0,922 | 0,903 | 0,953 | 0,894 |
| 6 | CatBoost | 0,972 | 0,990 | 0,922 | 0,903 | 0,962 | 0,885 |
| 7 | Stacking Ensemble | 0,971 | 0,990 | 0,871 | 0,851 | 0,989 | 0,779 |
| 8 | RandomForest | 0,967 | 0,988 | 0,911 | 0,891 | 0,970 | 0,858 |
| 9 | LightGBM | 0,967 | 0,988 | 0,909 | 0,886 | 0,935 | 0,885 |
| 10 | XGBoost | 0,962 | 0,979 | 0,918 | 0,897 | 0,944 | 0,894 |
| 11 | SVM | 0,959 | 0,977 | 0,896 | 0,873 | 0,960 | 0,841 |
| 12 | Bagging | 0,958 | 0,981 | 0,921 | 0,903 | 0,971 | 0,876 |
| 13 | GradientBoosting | 0,953 | 0,978 | 0,890 | 0,862 | 0,924 | 0,858 |
| 14 | QDA | 0,934 | 0,973 | 0,826 | 0,795 | 0,943 | 0,734 |
| 15 | AdaBoost | 0,933 | 0,970 | 0,832 | 0,791 | 0,881 | 0,788 |

**Çıkarımlar.** (i) Ağaç-toplulukları ve MLP, AUC-PR ≈ 0,97–0,98 ile lider grubu oluşturur.
(ii) En yüksek **F1 ve MCC, MLP**'dedir (recall avantajı, F1 0,946 / MCC 0,932). (iii) Stacking ve
Voting toplulukları yüksek kesinliğe (≥ 0,98) fakat görece düşük duyarlılığa eğilimlidir;
yanlış-alarma duyarlı operasyonel senaryolar için uygundur. Sıralamanın alt ucunda gözetimsiz
dedektörler (HBOS AUC-PR 0,213, EllipticEnvelope 0,240) beklendiği gibi azınlık sınıfında zayıftır.

### 6.2 Kıyaslama Reprodüksiyonu (Faz 2)

Kanonik sonuçlar, Ruszczak ve ark. (2024) Tablo 3 ile **aynı Ψ** üzerinde karşılaştırılmıştır
(22 eşleşen algoritma):

| Grup | Eşleşen | Ortalama \|ΔAUC-PR\| |
|------|:-------:|:-------------------:|
| **Gözetimli** | 7 | **0,004** |
| Gözetimsiz | 15 | 0,093 |

Gözetimli tarafta reprodüksiyon neredeyse birebirdir (FCNN≈MLP ΔAUC-PR = 0,000; XGBOD −0,002;
RF+ICCS +0,004; Linear+L2 +0,004). Bu, metodolojik temelin (resmi bölme + sızıntısızlık + 7 metrik)
doğruluğunun güçlü bir kanıtıdır. Gözetimsiz tarafta sapma daha büyük ve iki yönlüdür (ör. KNN
+0,252 iyileşme, LMDD −0,301 gerileme); bu kategori eşik/kontaminasyon seçimine ve ön işleme
detaylarına yüksek duyarlılık gösterir — makalede de bu kategori daha değişkendir.

### 6.3 Özellik Ablasyonu (AS2)

Tek-tek özellik çıkarma deneyleri (modeller arası ortalama ΔAUC) en kritik ve en gereksiz
özellikleri ortaya koyar:

| En kritik 5 (çıkarınca AUC düşüşü) | ΔAUC | En az etkili (çıkarınca artış) | ΔAUC |
|---|:---:|---|:---:|
| `n_peaks` | −0,018 | `smooth10_n_peaks` | +0,006 |
| `diff2_peaks` | −0,010 | `diff2_var` | +0,003 |
| `skew` | −0,004 | `smooth20_n_peaks` | +0,001 |
| `len` | −0,003 | `diff_var` | +0,001 |
| `duration` | −0,002 | `var_div_duration` | +0,001 |

Kümülatif ablasyon, **yalnızca 11 özellik** ile maksimum performansa ulaşılabildiğini gösterir;
geri kalan 7 özellik (std, kurtosis, mean, var_div_duration, diff_var, smooth20_n_peaks, diff2_var,
smooth10_n_peaks) bilgi kaybı olmadan çıkarılabilir. Tepe-temelli özellikler (`n_peaks`,
`diff2_peaks`) en yüksek ayırt edici güce sahiptir — fiziksel olarak anlamlıdır, çünkü anomaliler
tipik olarak sinyal morfolojisini (tepe yapısını) değiştirir.

### 6.4 Yorumlanabilirlik (SHAP)

SHAP analizi, kararların büyük ölçüde **varyans/türev ve tepe-temelli özellikler** tarafından
yönlendirildiğini gösterir; bu, ablasyon bulgusuyla (n_peaks/diff2_peaks'in kritikliği) tutarlıdır
ve fiziksel olarak yorumlanabilir: anomaliler segment değişkenliğini ve tepe sayısını artırır.

---

## 7. Özgün Katkılar

### 7.1 Profil-Temelli Sentetik Telemetri Üretimi (Üç Aşama)

Gerçek yaşam döngüsünü taklit eden üç ayrı aşamalı bir üreteç önerilmiştir:

| Aşama | İşlev | Çıktı | Gerçek-veri dayanağı |
|-------|-------|-------|----------------------|
| 1. Ham üretim | `generate_raw_stream()` | sürekli akış (segment/etiket YOK) | sinyal istatistikleri + kampanya/boşluk yapısı |
| 2. Segmentasyon | `segment_raw_telemetry()` | `segments.csv` formatı | uzunluk dağılımı + boşluk eşikleri (hibrit) |
| 3. Özellik çıkarımı | `extract_esa_features()` | `dataset.csv` formatı | 18 ESA özelliği |

**Aşama 1 — Sürekli ham akış.** Kanal profilleri gerçek OPSSAT-AD verisinden çıkarılır.
Manyetometre kanalları **Ornstein–Uhlenbeck** süreciyle (ortalamaya dönen, durağan std'li yürüyüş),
fotodiyot kanalları yörünge-periyodik, düşük-değer eğilimli (gamma-üssü) sinüzoidal döngüyle
üretilir. Altı anomali türü (spike, shift, noise, gap, flat, deformation) ve beş onboard artefakt
(mikro-boşluk, sıfır-zaman farkı, sıfır-değer, sensör donması, büyük boşluk) gerçek oranlarla
enjekte edilir.

**Aşama 2 — Hibrit segmentasyon.** Önce boşluk-bölme (Δt eşiği aşıldığında yeni koşu), ardından
kanalın gerçek uzunluk dağılımına göre uzunluk-penceresi bölme; segment etiketi anomali örtüşmesinden
(≥ %10 veya ≥ 3 örnek) türetilir.

**Sadakat doğrulaması.** Segment uzunluk dağılımı tüm kanallarda gerçeğe yakın; fotodiyot
ortalama/maksimum oranı düzeltme sonrası 0,28–0,38 (gerçek 0,165–0,353).

### 7.2 Sentetik Augmentasyon Ablasyonu ve Dağılım Açığı (AS3)

**Soru:** Profil-temelli sentetik telemetri, gerçek Ψ'de tespiti iyileştirir mi?

- **Tam-veri rejimi:** 0/250/500/1000/2000 sentetik segment eklemek AUC-PR'ı **anlamlı
  değiştirmemiştir** (ExtraTrees ≈ 0,983 sabit; RandomForest 0,967 → 0,967–0,971 bandı).
- **Az-veri rejimi:** Gerçek verinin %15/30/50/100'ü + 1000 sentetik; etki küçük ve karışıktır
  (ΔAUC-PR: −0,013, −0,004, +0,008, +0,002), MCC çoğunlukla hafif düşmüştür.

**Neden nötr? — Dağılım açığı.** Sentetik vs gerçek 18 özellik üzerinde Kolmogorov–Smirnov
mesafesi ortalama **0,29** (aralık [0,12, 0,52]) çıkmıştır. Kritik bulgu: en iyi ayırt edici özellik
olan **`n_peaks`, sentetik tarafta en kötü taklit edilen özelliktir (KS = 0,52)**; en iyi uyum
ise `diff2_peaks`'tedir (KS = 0,12). Yani üreteç, modelin en çok güvendiği özelliğin dağılımını en
zayıf yeniden üretmektedir — bu, augmentasyonun neden kazanç sağlamadığını doğrudan açıklar ve
**dürüst bir negatif bulgu** olarak raporlanır.

### 7.3 Augmentasyon Stratejisi Karşılaştırması (AS4)

SMOTE (öznitelik-uzayı enterpolasyonu), ICCS-ω (nominal-yalnız **sinyal-temelli** augmentasyon:
ω1 dikey ayna, ω2 zaman tersleme, ω3 dairesel kaydırma) ve sentetik augmentasyon, aynı Ψ'de
karşılaştırılmıştır:

| Model | Strateji | AUC-PR | Precision | Recall | F1 |
|-------|----------|:------:|:---------:|:------:|:--:|
| ExtraTrees | Baseline | 0,983 | 0,971 | 0,894 | 0,931 |
| ExtraTrees | +SMOTE | 0,984 | 0,928 | **0,912** | 0,920 |
| ExtraTrees | +ICCS-ω | 0,961 | **1,000** | 0,770 | 0,870 |
| RandomForest | Baseline | 0,967 | 0,970 | 0,858 | 0,911 |
| RandomForest | +SMOTE | 0,964 | 0,886 | 0,894 | 0,890 |
| RandomForest | +ICCS-ω | 0,951 | 0,989 | 0,770 | 0,866 |

**Örüntü:** Zengin-etiket rejiminde augmentasyon güçlü modellerin AUC-PR'ını anlamlı
iyileştirmez; asıl etki **kesinlik–duyarlılık dengesini kaydırmaktır** — ICCS-ω kesinliği artırır
(ExtraTrees'te precision = 1,000), SMOTE duyarlılığı artırır. Strateji seçimi bu nedenle
operasyonel önceliğe (yanlış alarmı mı yoksa kaçırmayı mı en aza indirmek) göre yapılmalıdır.

### 7.4 Güç Tüketimi ve Verimlilik (AS5)

64 algoritma için CPU gücü, eğitim süresi, çıkarım gecikmesi, bellek, enerji ve CO₂ profili
çıkarılmıştır. En verimli uçta hafif istatistiksel/lineer modeller (NaiveBayes, Ridge, KNN ≈
0,0003–0,0005 Wh), en pahalı uçta derin/GAN modelleri (AnoGAN 7,78 Wh, ALAD 7,29 Wh, Transformer
7,08 Wh) yer alır. Ağaç-toplulukları, yüksek doğruluğu **çok daha düşük maliyetle** sağlayarak
verimlilik–doğruluk dengesinde öne çıkar. **Onboard dağıtım** için öneri: doğruluk önceliğinde
ExtraTrees/HistGradientBoosting/CatBoost (yer istasyonu), kısıtlı güçte hafif dedektörler
(ECOD/HBOS/LODA).

---

## 8. Tartışma

Çalışmanın en güçlü yanı, gözetimli modellerde **|ΔAUC-PR| = 0,004**'lük reprodüksiyonun, kurulan
metodolojik temelin (resmi bölme + sızıntısızlık + 7 metrik) güvenilirliğini doğrulamasıdır. Bu,
referans çizgisinin üzerine konan özgün katkıların sağlam bir zemine oturduğunu gösterir.

İkinci olarak, sentetik augmentasyonun nötr sonucu bir başarısızlık değil, **bilgilendirici bir
negatif bulgudur**: (i) zengin-etiket rejiminde topluluk modelleri zaten doygunluğa yakındır, ve
(ii) üreteç, en ayırt edici özelliğin (`n_peaks`) dağılımını en zayıf taklit etmektedir. Bu iki
gözlem, "sentetik veri ne zaman fayda sağlar?" sorusuna somut bir yanıt verir: **az-etiket
rejiminde ve dağılım sadakatinin yüksek olduğu özelliklerde**. Ablasyon ve KS analizinin birlikte
okunması, gelecekteki üreteç iyileştirmelerinin doğrudan `n_peaks`/tepe-morfolojisi sadakatine
odaklanması gerektiğini söyler.

Üçüncü olarak, augmentasyonun asıl etkisinin **dengeyi kaydırmak** olması, metrik seçiminin
operasyonel hedefe bağlanması gerektiğini vurgular: tek bir "en iyi model" yerine, yanlış-alarm
maliyetine göre kesinlik-odaklı (ICCS-ω) veya kaçırma maliyetine göre duyarlılık-odaklı (SMOTE)
yapılandırmalar önerilir.

**Sınırlılıklar.** (i) Çalışma segment-temelli handcrafted özelliklere dayanır; ham-sinyal uçtan
uca derin modelleme kanonik karşılaştırmaya tam dahil edilmemiştir. (ii) Sentetik üreteç tek-kanallı
istatistiklere dayanır; kanallar-arası korelasyon modellenmemiştir. (iii) Güç profili, ölçülmüş
donanım sayaçları yerine literatür-temelli tahminlere dayanır. (iv) catch22 zaman-serisi özellik
kütüphanesi ortam kısıtları nedeniyle kullanılamamış, yerine ICCS-ω sinyal augmentasyonu
benimsenmiştir.

---

## 9. Sonuç ve Gelecek Çalışmalar

Bu tez, ESA OPS-SAT kıyaslamasında anomali tespitini **tekrarüretilebilir, sızıntısız ve
literatürle birebir kıyaslanabilir** bir çerçevede ele almış; 42 modellik bir havuzda en iyi
sonucu **ExtraTrees (AUC-PR 0,983)** ile elde etmiş ve gözetimli tarafta makaleyi
**|ΔAUC-PR| = 0,004** ile yeniden üretmiştir. Özgün katkı olarak, üç aşamalı profil-temelli sentetik
telemetri üreteci önerilmiş; sentetik augmentasyonun zengin-etiket rejiminde nötr kaldığı, etkisinin
kesinlik–duyarlılık dengesini kaydırmak olduğu dürüstçe gösterilmiş ve bu sonuç KS-temelli dağılım
analiziyle açıklanmıştır. Ablasyon, 18 özelliğin 11'inin yeterli olduğunu; SHAP, kararların
fiziksel olarak anlamlı (tepe/varyans-temelli) özelliklerce yönlendirildiğini ortaya koymuştur.

**Gelecek çalışmalar.** (i) Üretecin `n_peaks`/tepe-morfolojisi sadakatini hedefleyen iyileştirmesi
ve az-etiket rejiminde sentetik-temelli yarı-gözetimli öğrenme; (ii) kanallar-arası korelasyonu
modelleyen çok-değişkenli sentetik üretim; (iii) ham-sinyal uçtan uca derin modellerin kanonik
karşılaştırmaya tam entegrasyonu; (iv) onboard dağıtım için nicemlenmiş (quantized) hafif modellerin
ölçülmüş enerji profili; (v) çevrim-içi (online) ve kavram kayması (concept drift) altında uyarlanır
eşik mekanizmaları.

---

## Kaynaklar

1. Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. (2024). *The OPS-SAT benchmark for detecting
   anomalies in satellite telemetry.* arXiv:2407.04730.
2. Ruszczak, B., Kotowski, K., Andrzejewski, J., et al. (2023). *Machine Learning Detects Anomalies
   in OPS-SAT Telemetry.* International Conference on Computational Science (ICCS) 2023, LNCS,
   doi:10.1007/978-3-031-35995-8_21.
3. Chicco, D., Jurman, G. (2020). *The advantages of the Matthews correlation coefficient (MCC)
   over F1 score and accuracy in binary classification evaluation.* BMC Genomics, 21:6.
4. Kapoor, S., Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine
   learning-based science.* Patterns, 4(9):100804.
5. Wu, R., Keogh, E. (2022). *Current time series anomaly detection benchmarks are flawed and are
   creating the illusion of progress.* IEEE Transactions on Knowledge and Data Engineering.
6. Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority
   Over-sampling Technique.* Journal of Artificial Intelligence Research, 16:321–357.
7. Lundberg, S. M., Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.*
   Advances in Neural Information Processing Systems (NeurIPS) 30.
8. Zhao, Y., Nasrullah, Z., Li, Z. (2019). *PyOD: A Python Toolbox for Scalable Outlier Detection.*
   Journal of Machine Learning Research, 20(96):1–7.

---

## Ek A: Notebook ve Modül Haritası

Çalışma 14 Jupyter notebook + kanonik eğitim betiği ile baştan üretilebilir. Ayrıntılı teknik
dokümantasyon `docs/` klasöründedir.

**Notebook'lar:** 01 Veri İnceleme (EDA) · 02 Sızıntısız Ön İşleme · 03 Özellik Mühendisliği
(keşif) · 04 Gözetimli (demo) · 05 Gözetimsiz (demo) · 06 Model Karşılaştırma · 07 SHAP
Yorumlanabilirlik · 08 Özellik Ablasyonu · 09 Sentetik Üretim (3 aşama) · 10 ESA Feature Pipeline
(r ≈ 0.999 doğrulama) · 11 Güç Tüketimi · 12 Benchmark Reprodüksiyonu · 13 Sentetik Augmentasyon
Ablasyonu · 14 Augmentasyon Stratejisi Karşılaştırması.

**Çekirdek modüller (`src/`):** `metrics.py` (7 metrik standardı) · `benchmark_reference.py`
(makale Tablo 3) · `feature_engineer.py` (18 ESA çıkarımı, hibrit segmentasyon, ICCS-ω) ·
`synthetic_generator.py` (profil-temelli üreteç) · `preprocessor.py` (sızıntısız ön işleme) ·
`models/` (gözetimli/gözetimsiz/değerlendirici). Kanonik motor: `train_all_models.py`.

**Arayüz (`app/`):** Sonuçların interaktif keşfi için Dash tabanlı bir kontrol paneli
(ikincil öncelik; akademik sonuçların kaynağı notebook'lar + kanonik motordur).
