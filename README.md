# JobShield AI

Makine öğrenmesi ile sahte iş ilanlarını sınıflandırmaya yönelik bitirme projesi.

## Proje amacı
İlan metinleri ve veri setinde bulunan yapısal özellikleri kullanarak, `fraudulent` etiketiyle işaretlenmiş ilanları ayırt eden modeller geliştirmek ve karşılaştırmak.

## Araştırma sorusu
İlan metinlerine ek olarak yapısal özellikleri kullanmak, sahte ilanları tespit etme başarısını artırıyor mu?

## Veri seti
Kaggle: [Real or Fake? Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

CSV dosyasını indirdikten sonra `data/raw/fake_job_postings.csv` konumuna yerleştir. Ham veri dosyası `.gitignore` nedeniyle Git'e eklenmez. Veri setinin lisans ve kullanım koşullarını Kaggle sayfasından kontrol et.

## Klasör yapısı
- `data/raw/`: indirilen orijinal veri (Git'e eklenmez)
- `data/processed/`: işlenmiş veri çıktıları (Git'e eklenmez)
- `notebooks/`: keşifsel analiz ve deneyler
- `src/`: tekrar kullanılabilir kod
- `models/`: eğitilmiş model dosyaları (varsayılan olarak Git'e eklenmez)
- `reports/figures/`: grafikler
- `app/`: Streamlit arayüzü
- `tests/`: temel testler

## Başlangıç
Python 3.11 önerilir.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Çalışma ilkeleri
1. Önce veri ve etiketleri anla; sonra model eğit.
2. Test verisini model seçimi sırasında kullanma.
3. Dengesiz sınıflarda accuracy'yi tek başına raporlama.
4. Tekrarlanan açıklamaların eğitim/test kümeleri arasında sızıntı oluşturup oluşturmadığını kontrol et.
5. Model tahminini kesin dolandırıcılık kanıtı gibi sunma.
6. Her günün sonunda yapılanları `reports/progress_log.md` dosyasına kaydet.
