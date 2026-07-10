# MVTec AD Verisini İndirme

1. https://www.mvtec.com/company/research/datasets/mvtec-ad adresinden (akademik kullanım için ücretsiz) kayıt ol ve indirme linkini al.
2. Başlangıç kategorisi olarak **hazelnut** seçildi — sadece `hazelnut.tar.xz` indirmek yeterli (tüm veri setini indirmek zorunlu değil).
3. Arşivi bu klasöre (`data/raw/`) aç. Sonuçta şu yapı oluşmalı:

```
data/raw/hazelnut/
├── train/
│   └── good/                # sadece normal görüntüler
├── test/
│   ├── good/
│   ├── crack/
│   ├── cut/
│   ├── hole/
│   └── print/
└── ground_truth/
    ├── crack/
    ├── cut/
    ├── hole/
    └── print/
```

4. Faz 1'de bu klasör `dataset.py` ve `notebooks/eda.ipynb` tarafından okunacak; `data/processed/` altına train/val/test split'i buradan üretilecek.

Not: MVTec AD'nin resmi `train/` split'i sadece normal (good) görüntü içerir — anomali/kusur örnekleri yalnızca `test/` altında bulunur. Val split'i oluştururken bu kısıtı dikkate al (plan dosyasındaki Faz 1 notuna bak).
