# Endüstriyel Kusur Segmentasyonu (U-Net / DeepLabV3+)

## Problem Tanımı

MVTec AD veri setinde piksel-seviyesinde kusur segmentasyonu yaparak "hangi pikseller kusurlu" sorusuna cevap veren bir model geliştirmek. Sınıflandırma/tespit (bounding box) yerine segmentasyon, endüstriyel kalite kontrol ve tıbbi görüntüleme gibi alanlarda asıl aranan yetenektir.

## Veri Seti

- **MVTec AD** ([mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)) — 15 kategori, normal + anormal görüntüler + piksel-seviyesi ground truth maskeler.
- **Kategori:** `hazelnut`. İndirme talimatları için [data/raw/NOTES.md](data/raw/NOTES.md).
- **Split stratejisi:** MVTec AD'nin resmi `train/` split'i sadece normal (good) görüntü içerir; piksel-seviyesi maskeler yalnızca `test/` split'inde bulunur. Bu yüzden resmi `test/` (good + 4 kusur tipi, maskeli, 110 görsel) etiketli havuz olarak alındı, buna resmi `train/good` havuzundan rastgele 60 ek negatif örnek eklendi (toplam 170 görsel) ve kategoriye göre stratifiye 70/15/15 train/val/test split'i üretildi (bkz. `src/prepare_data.py`).
- **EDA bulguları** (`data/processed/eda_stats.json`):
  - Tüm görseller 1024×1024 çözünürlükte.
  - Kusurlu piksel oranı ortalama **%3.35** (min %0.23, max %27.7) — belirgin class imbalance, bu yüzden salt BCE yerine BCE+Dice (Combo) loss kullanıldı.
  - Split dağılımı: train 120 (crack 13, cut 12, hole 13, print 12, good 28, good_train 42), val 27, test 23.

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Bu depoda geliştirme ve eğitim, CUDA'lı PyTorch içeren bir conda ortamında (`defect_seg_env`, Python 3.11, torch 2.10+cu126) yürütüldü; GPU: NVIDIA RTX 4060 8GB.

## Çalıştırma Sırası

```bash
python src/prepare_data.py                                   # Faz 1: split + EDA
python src/train.py --config configs/unet_baseline.yaml      # Faz 2
python src/train.py --config configs/deeplabv3plus.yaml      # Faz 3
python src/evaluate.py --configs configs/unet_baseline.yaml configs/deeplabv3plus.yaml   # Faz 3
python src/visualize.py --config configs/unet_baseline.yaml  # Faz 4
python src/visualize.py --config configs/deeplabv3plus.yaml  # Faz 4
```

## Klasör Yapısı

```
defect-segmentation/
├── data/
│   ├── raw/                    # MVTec AD ham veri (bkz. NOTES.md)
│   └── processed/              # splits.csv + eda_stats.json (prepare_data.py çıktısı)
├── src/
│   ├── prepare_data.py         # Faz 1: split manifesti + EDA istatistikleri
│   ├── dataset.py              # Dataset ve DataLoader sınıfları
│   ├── augmentations.py        # albumentations pipeline'ları
│   ├── models/factory.py       # U-Net / DeepLabV3+ model seçici
│   ├── train.py                # Combo (BCE+Dice) loss ile eğitim döngüsü
│   ├── evaluate.py             # test split değerlendirme + karşılaştırma tablosu
│   └── visualize.py            # tahmin galerisi + Grad-CAM + failure gallery
├── configs/
│   ├── unet_baseline.yaml
│   └── deeplabv3plus.yaml
├── notebooks/
│   ├── eda.ipynb              # Faz 1: split dagilimi, goruntu boyutu, maske piksel orani
│   └── explainability.ipynb   # Faz 4: CAM yontem karsilastirma, model karsilastirma, basarisiz ornek analizi
├── outputs/
│   ├── checkpoints/             # {model}_best.pt (en iyi val Dice)
│   ├── logs/                    # {model}_history.json (epoch bazlı metrikler)
│   └── predictions/
│       ├── comparison.md        # model karşılaştırma tablosu
│       └── {model}/gallery, failure_gallery
└── requirements.txt
```

## Roadmap

- [x] **Faz 1 — Veri Hazırlığı ve EDA**: MVTec AD (hazelnut) indirme, görüntü boyutu/class imbalance analizi, train/val/test split stratejisi.
- [x] **Faz 2 — Baseline Model (U-Net)**: `segmentation-models-pytorch` ile ResNet34 encoder, Combo (BCE+Dice) loss, Dice/IoU skorları.
- [x] **Faz 3 — Karşılaştırma Modeli (DeepLabV3+)**: Aynı pipeline ile eğitim, model karşılaştırma tablosu (Dice, IoU, inference süresi, parametre sayısı).
- [x] **Faz 4 — Yorumlanabilirlik ve Görselleştirme**: Grad-CAM ısı haritaları, false positive/negative galerisi.
- [x] **Faz 5 — Rapor**: Karşılaştırma tablosu, örnek görseller, sonuç ve gelecek çalışmalar.
- [ ] (Opsiyonel) Streamlit ile "görsel yükle, kusur maskesini gör" demo'su.

## Değerlendirme Metrikleri

Dice katsayısı, IoU (Intersection over Union), pixel accuracy — imbalance nedeniyle sadece accuracy yeterli değil (bkz. EDA), üçü birlikte raporlandı.

## Sonuçlar

Her iki model 256×256 girdi, ResNet34 encoder (ImageNet pretrained), Adam + cosine annealing LR, 50 epoch, Combo (BCE+Dice) loss ile eğitildi (bkz. `configs/*.yaml`, `outputs/logs/*_history.json`).

**En iyi validation Dice:** U-Net 0.8495, DeepLabV3+ 0.8740.

**Test split karşılaştırması** (`outputs/predictions/comparison.md`, 23 görsel, RTX 4060):

| Model | Encoder | Dice | IoU | Pixel Acc | Inference (ms) | Parametre |
|---|---|---|---|---|---|---|
| U-Net | resnet34 | 0.7516 | 0.6021 | 0.9961 | 24.55 | 24,436,369 |
| DeepLabV3+ | resnet34 | 0.7480 | 0.5974 | 0.9961 | 22.36 | 22,437,457 |

Validation'da DeepLabV3+ hafif önde (0.874 vs 0.850 Dice), test split'inde (23 görsel — küçük örneklem, gürültülü) iki model neredeyse ayırt edilemez düzeyde yakın çıktı; DeepLabV3+ biraz daha az parametre ve biraz daha hızlı inference sunuyor. Bu ölçekte (170 toplam görsel) aradaki fark mimari üstünlükten çok örneklem gürültüsünü yansıtıyor olabilir — daha güvenilir bir karşılaştırma için kategori sayısı ve örnek sayısı artırılmalı.

**Yorumlanabilirlik:** `outputs/predictions/{model}/gallery/` altında her test görseli için girdi / gerçek maske / tahmin / Grad-CAM ısı haritası yan yana kaydedildi (örnek: `outputs/predictions/unet/gallery/000_crack.png`). Grad-CAM, ResNet34 encoder'ın son bloğundan (`layer4`) alındığı için düşük çözünürlüklü ve geniş bir aktivasyon haritası üretiyor — modelin kusurlu bölgeye kabaca odaklandığını gösteriyor ama piksel-hassas değil (beklenen bir sınırlama, tahmin maskesinin kendisi piksel-hassas sonucu veriyor).

Daha derin bir XAI analizi için [notebooks/explainability.ipynb](notebooks/explainability.ipynb) hazırlandı: GradCAM/GradCAM++/EigenCAM yöntemlerini aynı görsel üzerinde karşılaştırır, U-Net ile DeepLabV3+'in dikkat haritalarını yan yana gösterir ve en düşük Dice'lı örnekleri Grad-CAM ile birlikte inceler. Bulgu: model, düşük skorlu `crack`/`cut` örneklerinde kusurun civarına doğru odaklanıyor ama ince/düşük kontrastlı sınırı tam yakalayamıyor — hata "yanlış yere bakma"dan değil "doğru yere bakıp sınırı kaçırma"dan kaynaklanıyor.

**Başarısız örnekler:** Her iki modelde de en düşük Dice skorları `crack` ve `cut` kategorilerinde (ince, düşük kontrastlı kusurlar) görüldü — bkz. `outputs/predictions/{model}/failure_gallery/`. `hole` ve `print` gibi daha belirgin/yüksek kontrastlı kusurlarda skorlar gözle görülür şekilde daha yüksek.

## Sınırlamalar

- Tek kategori (`hazelnut`) ve küçük veri seti (170 görsel) — sonuçlar diğer MVTec AD kategorilerine doğrudan genellenemez.
- Resmi `train/good` havuzundan sadece 60 ek negatif örnek kullanıldı; tüm 391 görsel kullanılsaydı negatif/pozitif dengesi ve genelleme farklı çıkabilirdi.
- Test split'i 23 görsel ile küçük — metrikler örneklem gürültüsüne duyarlı, güven aralığı raporlanmadı.
- Grad-CAM tek bir encoder katmanından (`layer4`) alındı; çok ölçekli bir yorumlanabilirlik analizi yapılmadı.

## CV Cümlesi

> "MVTec AD (hazelnut) veri setinde U-Net ve DeepLabV3+ mimarilerini (ResNet34 encoder) karşılaştırarak piksel-seviyesi endüstriyel yüzey kusuru segmentasyonu gerçekleştirdim; %87 validation Dice skoru elde ettim, Grad-CAM ile model yorumlanabilirliğini ve başarısız örnek profilini analiz ettim."

## Kaynak

Bu proje, [cv-projeleri-plani.md](../cv-projeleri-plani.md) dosyasındaki Proje 1 planına göre yapılandırılmıştır.
