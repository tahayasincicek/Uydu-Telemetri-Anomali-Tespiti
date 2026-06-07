# Notebook Metinleri — Yazım İskeleti

Bu belge, 14 notebook'un metin (markdown) hücrelerini **kendi cümlelerinle yeniden yazman**
için bir iskelettir. Her bölüm için "ne anlatılmalı" özeti ve kullanacağın anahtar sayı/terimler
verilmiştir. Hazır paragraf **bilinçli olarak verilmedi** — amaç metnin senin sesin, senin emeğin
olması; böylece içeriği anlar ve savunabilirsin.

## Nasıl kullanılır

1. Her bölüm için aşağıdaki maddeleri oku, **ana fikri kendi cümlelerinle** yaz.
2. Sayıları yazmadan önce ilgili notebook çıktısından **doğrula** (rakamları ezberden alma).
3. Bölümler arası geçişleri kendi mantığınla kur ("önce X'i inceledik, çünkü Y'ye ihtiyaç vardı").

## Kendi sesinle yazmak için (meşru ipuçları)

- **Önce anla, sonra yaz.** Bir bölümü açıklayamıyorsan, kodu/çıktıyı tekrar incele; anladığını
  yazınca metin doğal olarak senin olur.
- **"Neden" ekle.** Her teknik adımın gerekçesini kendi bakışınla yaz (ör. "rastgele bölme yerine
  resmi bölmeyi seçtik, çünkü…"). Gerekçe senin yorumundur, kopyalanamaz.
- **Birinci çoğul şahıs** kullan (yaptık, inceledik, gözlemledik) — tez/rapor dilinde doğaldır.
- **Kendi gözlemini kat.** Beklediğinle çıkanı karşılaştır; sürprizleri/kısıtları belirt.
- **Tutarlı terminoloji** kullan ama cümle yapını çeşitlendir; her bölümü aynı kalıpla başlatma.

---

## NB01 — Veri İnceleme (EDA)

**Amaç (kendi cümlenle):** ESA OPS-SAT telemetri verisini modellemeden önce tanımak; yapısını,
dağılımını ve anomali örüntüsünü keşfetmek.

- **Giriş / kanallar:** Veri iki sensör tipinden gelir — 3 manyetometre (CADC0872–0874) ve
  6 fotodiyot (CADC0884–0894), toplam **9 kanal**. Manyetometre yavaş/durağana yakın (~10⁻⁵
  genlik); fotodiyot yörünge gölge-aydınlık döngüsüyle çoğunlukla sıfıra yakın. Bu farkın neden
  önemli olduğunu kendi cümlenle bağla (özellik ve model davranışını etkiler).
- **1. Veri yükleme:** İki dosya var — `segments.csv` (303.493 ham ölçüm) ve `dataset.csv`
  (2.123 segment × 18 özellik). Birinin ham sinyal, diğerinin segment-özeti olduğunu vurgula.
- **2. İstatistiksel analiz:** Kanal bazında temel istatistikler; iki sensör tipinin ölçek farkı.
- **3. Eksik veri:** Eksik/boşluk yapısını ve kanal bazında veri yoğunluğunu özetle.
- **4. Dağılım analizi:** Histogram/KDE, box ve violin (normal vs anomali) ile özelliklerin
  ayırt ediciliğine ilk bakış.
- **5. Korelasyon:** Pearson/Spearman; yüksek korelasyonlu özellik kümeleri (ileride ablasyonla
  bağ kuracaksın).
- **6. Zaman serisi:** Rolling mean/std ve segment trendleriyle sinyal davranışı.
- **7. Anomali etiketi:** Sınıf dengesizliği — genelde **~%20 anomali**; kanal/zaman bazında
  anomali dağılımı. Dengesizliğin metrik seçimini neden etkilediğine değin.
- **8. Bulgular:** 3-4 maddede kendi çıkarımların (dengesizlik, iki sensör rejimi, ayırt edici
  özellik adayları).

---

## NB02 — Veri Ön İşleme (Sızıntısız)

**Amaç:** Veriyi modele hazırlarken **veri sızıntısını (data leakage) önlemek** ve resmi bölmeyi
uygulamak.

- **Amaç/adımlar:** EDA bulgularından ön işlemeye köprü kur.
- **B1 Ham doğrulama / B2 Eksik veri / B3 Gürültü (filtre) / B4 Outlier / B5 Normalizasyon:**
  Her adımın **neyi neden** yaptığını kısa yaz (ör. outlier'ı silmek yerine kırpmak — anomali
  bilgisini kaybetmemek için).
- **B6 Resmi split + Preprocessor (kritik bölüm):** Burası tezin metodolojik kalbi. Anlatılacaklar:
  - Bölme **`dataset.csv`'deki `train` kolonundan** gelir (rastgele değil): T = **1.594**,
    Ψ = **529**. Makaleyle kıyaslanabilirlik için bu şart.
  - **Ölçekleyici yalnızca T üzerinde fit edilir**, Ψ'ye yalnız transform — test istatistiğinin
    eğitime sızmaması için.
- **B7 SMOTE (yalnız T):** Dengeleme **sadece eğitim katmanına** uygulanır; **Ψ asla
  dengelenmez/dokunulmaz** (gerçek %21,4 oran korunur). Önceki sürümde SMOTE'un bölme öncesi
  uygulanmasının neden sızıntı olduğunu açıkla.
- **B8 Validation:** Doğrulama seti T'den ayrılır; Ψ korunur.
- **B9 Kaydetme:** `X_train` (SMOTE'lu T), `X_val`, `X_test` (= dokunulmamış Ψ).
- **Referans:** Kapoor & Narayanan (2023), Wu & Keogh (2022) — sızıntı/tekrarüretilebilirlik.

---

## NB03 — Özellik Mühendisliği (keşif)

**Amaç:** ESA'nın 18 özelliğine ek olarak kendi sinyal-işleme özelliklerini (RMS, P2P, crest, ZCR)
keşfetmek.

- Bunun bir **keşif/genişletme** denemesi olduğunu, **kanonik baseline'ın yalnız 18 ESA özelliğini**
  kullandığını net belirt (karşılaştırılabilirliği kirletmemek için).
- Özel özellik çıkarımı → birleştirme → katalog kaydı akışını özetle.
- Kendi yorumun: ek özelliklerin potansiyel katkısı vs baseline saflığı ödünleşimi.

---

## NB04 — Gözetimli Öğrenme (demo)

**Amaç:** Resmi bölme + 18 özellik üzerinde temsili gözetimli modelleri eğitip Ψ'de 7 metrikle
değerlendirmek.

- **Bu bir gösterim notebook'u**; tam 42-model havuzu `train_all_models.py` (kanonik) ile üretilir,
  bu notebook kanonik artefaktları **ezmez**. Bunu açıkça yaz.
- **B1 Veri:** Resmi split + 18 ESA özelliği.
- **B2–B4 Modeller:** RF, XGBoost, SVM, MLP — neden bu temsilciler.
- **B5 Değerlendirme:** 7 metrik (Accuracy, Precision, Recall, F1, **MCC**, **AUC-ROC**, **AUC-PR**),
  AUC-PR birincil.
- **B6 ROC / B7 Kaydetme.**
- **Gelişmiş + genişletilmiş bölümler:** LightGBM/CatBoost/Stacking ve klasik(14)+derin sıralı(15)
  ağların eklendiğini, asıl üretimin kanonik motorda olduğunu belirt.

---

## NB05 — Gözetimsiz Öğrenme (demo)

**Amaç:** Etiket kullanmadan anomali tespiti; modeller **yalnız nominal** veriyle eğitilir, eşik
doğrulamada seçilir.

- Protokolü vurgula: nominal-üzerinde-eğit, **eşiği validation'da seç**, Ψ'de değerlendir.
- B0 Veri → B1 IsolationForest → B2 OneClassSVM → B3 K-Means → B4 LOF → B5 Autoencoder → B6 ROC →
  B7 Kaydetme; her modelin anomali skorunu nasıl tanımladığını kısaca.
- Ek modeller: GMM, EllipticEnvelope, PCA (yeniden-yapım hatası), DBSCAN (çekirdek-uzaklığı) +
  PyOD (ECOD/COPOD/HBOS/CBLOF).
- Kendi yorumun: gözetimsizlerin gözetimlilerin neden gerisinde kaldığı (etiketsizlik bedeli).

---

## NB06 — Tüm Modellerin Karşılaştırılması

**Amaç:** Tüm modelleri **aynı Ψ** üzerinde 7 metrikle kıyaslamak; kanonik tabloyu görselleştirmek.

- **Önemli:** Bu notebook kanonik `final_comparison.json`'ı **yükler** (yeniden hesaplamaz/ezmez).
- B1 Ψ + modelleri yükle → B2 metrikler → B3 ROC/PR → B4 confusion → B5 hesaplama verimliliği →
  B6 sonuç/öneriler → B7 export.
- **Sonuç bölümü (kendi sentezin):**
  - En iyi (AUC-PR): **ExtraTrees 0,983**, Voting 0,980, MLP 0,979, HistGB 0,974, XGBOD 0,973,
    CatBoost 0,972.
  - Gözetimliler benchmark'ı domine eder; gözetimsizler geride.
  - Augmentasyon (NB13–14): zengin-etiket rejiminde anlamlı kazanç yok, denge kayar.
  - Operasyonel öneri: yer istasyonu için ağaç-toplulukları; onboard için hafif dedektörler.

---

## NB07 — SHAP Analizi (Yorumlanabilirlik)

**Amaç:** Modellerin kararını hangi özelliklerin yönlendirdiğini açıklamak.

- B1 Hazırlık → B2 SHAP (RF/XGBoost/MLP) → B3 model karşılaştırma → B4 kaydet → B5 bulgular.
- **Bulgular (kendi cümlenle):** Kararlar büyük ölçüde **varyans/türev ve tepe-temelli** özelliklere
  dayanır; bu fiziksel olarak anlamlıdır (anomali, sinyal değişkenliğini/tepe yapısını bozar).
  Bu sonucu NB08 ablasyonuyla (n_peaks/diff2_peaks kritikliği) bağla.

---

## NB08 — Ablasyon (Özellik Önemi)

**Amaç:** 18 özelliğin katkısını sistematik çıkarma deneyleriyle ölçmek; özellik uzayını küçültmek.

- B2 Tekil çıkarma → B3 kümülatif azaltma → B4 grup → B5 tip → B6 görseller → B7-8 en iyi set →
  B9 bulgular.
- **Anahtar sayılar:**
  - En kritik özellikler: **`n_peaks` (−0,018 AUC)**, `diff2_peaks` (−0,010), skew, len, duration.
  - **Yalnızca ~11 özellik** maksimum performansa yeter; 7 özellik bilgi kaybı olmadan çıkarılabilir.
- Kendi yorumun: tepe-temelli özelliklerin neden en ayırt edici olduğu (anomali morfolojiyi bozar).

---

## NB09 — Sentetik Telemetri (3 Aşama)

**Amaç:** Gerçeğe sadık sentetik telemetriyi **üç aşamada** üretmek: ham akış → segmentasyon →
özellik.

- **Neden bu mimari?** Tek-adımlı üretim yerine gerçek yaşam döngüsünü taklit; segmentasyonun
  **gerçek bir algoritma adımı** olması. Bunu kendi gerekçenle yaz.
- B1 kanal profilleri → B2 sürekli sinyal modelleri (manyetometre = **Ornstein-Uhlenbeck**,
  fotodiyot = yörünge-periyodik gamma-eğilimli) → B3 anomali enjeksiyonu (**6 tür**) →
  B4 onboard artefaktlar (**5 tür**, anomali değil) → B5 Aşama 1 ham akış → B6 Aşama 2 hibrit
  segmentasyon (boşluk-bölme + uzunluk-penceresi) → B7 Aşama 3 ESA özellikleri → B8 doğrulama →
  B9 kaydet.
- **Sadakat:** Segment uzunluk dağılımı gerçeğe yakın; fotodiyot ort/max düzeltme sonrası 0,28–0,38
  (gerçek 0,165–0,353).

---

## NB10 — ESA Feature Extraction Pipeline

**Amaç:** Ham segmentlerden 18 ESA özelliğini otomatik çıkaran hattı belgelemek ve **doğrulamak**.

- **Neden otomatikleştirdik:** Tekrarüretilebilirlik + sentetik veriye uygulanabilirlik.
- B2 18 özelliğin anatomisi (3 grup: 12 ham + 2 yumuşatılmış + 4 türev) → B3 tek segment adım adım →
  B4 tüm veriye uygula → B5 **ESA orijinali ile doğrula** → B6 ayırt edicilik → B7 kaydet.
- **Anahtar bulgu:** Otomatik çıkarım, ESA'nın `dataset.csv`'sini **r ≈ 0,999** korelasyonla yeniden
  üretir (yalnız `smooth20_n_peaks` r=0,982; konvolüsyon kenar etkisi). Referans: Ruszczak et al.
  (2024), arXiv:2407.04730.

---

## NB11 — Güç Tüketimi / Hesaplama Maliyeti

**Amaç:** 64 algoritmanın enerji/bellek/çıkarım maliyetini modelleyip verimlilik-doğruluk dengesini
çıkarmak.

- **Neden önemli:** Onboard kaynak kısıtı; sürdürülebilirlik (karbon).
- B2 enerji/karbon modeli → B3 model bazında enerji → B4 **enerji vs F1 verimlilik haritası** →
  B5 kategori/karmaşıklık → B6 bellek/çıkarım → B7 veri-boyutu ölçekleme → B8 karbon + öneriler.
- **Anahtar sayılar:** En verimli ≈ NaiveBayes/Ridge/KNN (~0,0003–0,0005 Wh); en pahalı AnoGAN
  7,78 Wh, Transformer 7,08 Wh. Ağaç-toplulukları yüksek doğruluğu düşük maliyetle sağlar.
- **Not:** Profil **literatür-temelli tahmindir**, ölçülmüş donanım sayacı değil — bunu dürüstçe
  belirt (sınırlılık).

---

## NB12 — Benchmark Karşılaştırması

**Amaç:** Kanonik sonuçları Ruszczak et al. (2024) Tablo 3 ile **aynı Ψ** üzerinde karşılaştırmak.

- B1 referans baseline + kanonik sonuçları yükle → B2 gözetimli reprodüksiyon → B3 gözetimsiz
  karşılaştırma → B4 görsel (paper vs bizim) → B5 bulgular → B6 kaydet.
- **Anahtar bulgu:** Gözetimli reprodüksiyon ortalama **|ΔAUC-PR| = 0,004** (neredeyse birebir;
  FCNN Δ=0,000). Gözetimsizde sapma büyük ve iki yönlü (eşik/kontaminasyon duyarlılığı).
- Yaklaşık/paradigma eşleşmelerini (`~`) açıkça işaretle: FCNN≈MLP, RF+ICCS≈RandomForest,
  Linear+L2≈Ridge.
- Kendi yorumun: 0,004'lük sapmanın metodolojik temelin doğruluğunu kanıtlaması.

---

## NB13 — Sentetik Augmentasyon Ablasyonu

**Amaç:** "Profil-temelli sentetik veri gerçek Ψ'de tespiti iyileştirir mi?" sorusunu test etmek.

- **Metodoloji:** Sentetik yalnız **eğitime** eklenir; Ψ gerçek kalır (sızıntısız).
- İki deney: (A) tam-veri, (B) az-veri rejimi.
- B2 gerçek vs sentetik dağılım (**neden analizi**) → B3 Deney A → B4 Deney B → B5 bulgular.
- **Anahtar bulgu (dürüst negatif sonuç):**
  - Tam-veride AUC-PR **anlamlı değişmez**; az-veride etki küçük/karışık.
  - **Neden:** sentetik-gerçek ortalama **KS = 0,29**; kritik nokta — en ayırt edici özellik
    **`n_peaks` sentetik tarafta en kötü taklit edilen** özelliktir (**KS = 0,52**), en iyi uyum
    `diff2_peaks` (0,12).
- **Bilimsel katkı:** "Makul görünen" sentetik verinin tespiti garanti etmediğini, dağılım
  sadakatinin (özellikle ayırt edici özelliklerde) belirleyici olduğunu göster.

---

## NB14 — Augmentasyon Stratejisi Karşılaştırması

**Amaç:** **SMOTE** (özellik-uzayı), **ICCS-ω** (sinyal-temelli), **Sentetik** stratejilerini aynı
Ψ'de kıyaslamak.

- **Metodoloji:** Üçü için ortak, sızıntısız protokol; ICCS-ω yalnız-nominal sinyal augmentasyonu
  (ω1 dikey ayna, ω2 zaman tersleme, ω3 dairesel kaydırma).
- B2 üç augmentasyon setini hazırla → B3 çoklu-model 7 metrik karşılaştırma → B4 **precision-recall
  dengesi** → B5 bulgular.
- **Anahtar örüntü:** Zengin-etiket rejiminde AUC-PR'da anlamlı kazanç yok; asıl etki **dengeyi
  kaydırmak** — **ICCS-ω → precision↑** (ExtraTrees'te 1,000), **SMOTE → recall↑**.
- **Bilimsel katkı:** OPS-SAT'ta augmentasyon stratejilerinin ilk birleşik karşılaştırması; strateji
  seçiminin operasyonel önceliğe (yanlış alarm vs kaçırma) bağlanması.

---

## Son hatırlatma

- Yazdıktan sonra her notebook için **kendine sor:** "Bunu bir jüri üyesine sözlü anlatabilir
  miyim?" Cevap evetse, metin gerçekten senin.
- Sayıları yazmadan **`reports/metrics/` çıktılarından doğrula**; bu hem doğruluğu hem de
  metnin senin ürünün olduğunu güçlendirir.
- Şeffaflık için tez/makaleye, araçların hangi adımlarda (kod, analiz, taslak düzenleme) yardımcı
  olduğunu belirten kısa bir **"araç kullanımı" beyanı** eklemen önerilir.
