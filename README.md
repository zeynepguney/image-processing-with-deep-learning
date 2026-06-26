# image-processing-with-deep-learning
# Etkinlik Afişlerinden Derin Öğrenme Tabanlı Otomatik Veri Yapılandırma ve Kalite Kontrol Boru Hattı 🚀

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![VLM](https://img.shields.io/badge/VLM-Qwen2--VL%20%2F%20Florence--2-red.svg)](https://huggingface.co/)
[![Vaka Çalışması](https://img.shields.io/badge/Vaka%20%C3%87al%C4%B1%C5%9Fmas%C4%B1-T%C3%BCrkiye%20Yapay%20Zeka%20Toplulu%C4%9Fu-green.svg)](https://turkiye.ai/)

Bu depo, otomatik olarak üretilen YouTube kapak görsellerinin ve etkinlik afişlerinin **otomatik Kalite Kontrolü (QA) ve yapısal çapraz doğrulaması** için tasarlanmış derin öğrenme tabanlı bir görsel-dil çerçevesinin resmi uygulamasını içermektedir.

Balıkesir Üniversitesi **"Derin Öğrenme ile Görüntü İşleme"** yüksek lisans dersi Final Projesi kapsamında geliştirilen bu proje, **Türkiye Yapay Zeka Topluluğu**'nun medya üretim süreçlerindeki gerçek bir operasyonel darboğazı çözmeyi hedeflemektedir.

---

## 📌 Proje Genel Bakışı ve Motivasyon

Modern topluluk yönetiminde görsel içerik üretimini ölçeklendirmek; grafik tasarım şablonlarının (örn. Figma) veri tablolarıyla (örn. Google Sheets) senkronize edilmesini gerektirir. Bu yaklaşım yüzlerce afişin toplu olarak otomatik üretilmesini sağlasa da beraberinde ciddi bir **Kalite Kontrol (QA) darboğazı** getirmektedir. Metin kutularının taşması, font işleme hataları, Türkçe karakterlerin bozulması veya satır eşleşme hataları görsellerin tek tek insan gözüyle kontrol edilmesini zorunlu kılmakta ve bu da otomasyonun getirdiği zaman kazancını sekteye uygular.

Bu proje, üretilen görselleri orijinal kaynak veri tabanıyla çapraz kontrol eden **iki aşamalı akıllı bir doğrulama boru hattı** sunmaktadır:
1. **Sınır Kutusu Korumalı Hibrit Mimari:** Hafif sıklet görsel modellerden gelen konumsal bölge önerilerini (`Florence-2-Base`), yerelleştirilmiş Türkçe optimizasyonlu OCR (`EasyOCR`) ve karakter tabanlı karakter dizisi mesafe metrikleriyle (`SequenceMatcher`) birleştiren yaklaşım.
2. **Uçtan Uca Üretken VLM Mimarisi:** Herhangi bir ön işleme veya harici sözlük katmanına ihtiyaç duymadan, sıfır-atış (zero-shot) yöntemiyle doğrudan görsellerden şemaya uygun JSON çıktısı üretebilen talimat ayarlı büyük Görsel-Dil Modelleri (`Qwen2-VL-7B-Instruct`).

---

## 📂 Depo Yapısı ve Betik Dizini

```text
├── colab/
│   └── qwen_colab_run.ipynb       # # Yerel donanım sınırları nedeniyle projenin Google             
                                   Drive üzerinden Google Colab (T4 GPU) ile bulutta çalıştırılmasını sağlayan otomasyon defteri için
├── dataset/
│   ├── afisler/                   # Test süreçlerinde kullanılan 200 adet sentetik/orijinal afiş
│   └── sentetik_bootcamp_verisi2.csv # Zemin gerçekliği (Ground Truth) ilişkisel veri tabanı
├── src/
│   ├── 1_baseline_easyocr.py      # Standart doğrusal OCR metin havuzu çıkaran temel çizgi betiği
│   ├── 2_vlm_test.py              # Keşifsel Florence-2-Large çıkarım betiği
│   ├── 3_vlm_test.py              # Parametrik Florence-2-Base çıkarım betiği
│   ├── 4_hybrid_ocr.py            # Geliştirilen hibrit doğrulama hattı için tekli görsel test betiği
│   ├── 5_evaluate_hybrid.py       # Hibrit model için toplu değerlendirme boru hattı (200 afiş)
│   ├── 6_qwen_vlm_test.py         # Qwen2-VL kullanarak tekli görselden yapılandırılmış veri çıkarımı
│   ├── 7_evaluate_qwen.py         # Qwen2-VL için toplu değerlendirme ve JSON şema doğrulama betiği
│   ├── hybrid_ocr_utils.py        # Geometrik kırpma ve sözlük eşleştirme için yardımcı fonksiyonlar
│   └── qwen_vlm_utils.py          # Qwen2-VL veri hazırlığı için görsel işleme yardımcı fonksiyonları
├── .gitignore                     # Git takip sınırları (venv, önbellekler ve sonuçları hariç tutar)
├── requirements.txt               # Yerel ortam gereksinimleri (EasyOCR, Florence-2)
└── requirements-qwen.txt          # GPU/Bulut ortamı gereksinimleri (Qwen2-VL konfigürasyonları)

```

---
### ☁️ Neden Google Colab ve Google Drive Entegrasyonu Kullanıldı?

Projede kullanılan **Qwen2-VL-7B-Instruct** modeli yaklaşık **16.6 GB** ağırlığında devasa bir modeldir. Bu boyuttaki modeller, yerel bilgisayarların (özellikle MacBook Air gibi paylaşımlı bellek kullanan CPU/MPS donanımların) RAM ve VRAM sınırlarını fazlasıyla aşmakta, kilitlenme (crash) ve bellek yetersizliği hatalarına yol açmaktadır.

Bu donanım darboğazını aşmak ve projenin otonom QA (Kalite Kontrol) sürecini kesintisiz yürütebilmek adına şu hibrit mühendislik stratejisi uygulanmıştır:
* **Veri Bütünlüğü ve Drive Köprüsü:** Yerel bilgisayarda geliştirilen tüm kod tabanı (`src/`) ve veri seti (`dataset/`), Google Drive üzerine taşınmıştır.
* **Bulut Tabanlı Donanım Hızlandırma:** `colab/qwen_colab_run.ipynb` otomasyon defteri, Google Colab ortamında çalıştırılarak buluttaki **Nvidia T4 GPU (CUDA)** donanımını tetikler.
* **Otonom Çalışma:** Bu notebook, Drive'daki proje klasörüne bağlanır (`drive.mount`), bulut sunucusuna gerekli kütüphaneleri yükler ve `6_qwen_vlm_test.py` ile `7_evaluate_qwen.py` betiklerini hiçbir kod değişikliğine gerek kalmadan CUDA altyapısıyla çalıştırır.

Bu sayede, yerel donanımın yetersiz kaldığı durumlarda projeyi bulut mimarisine taşıyarak "tekrarlanabilir" ve "ölçeklenebilir" bir yapay zeka boru hattı kurulmuştur.

---


### 🛠️ Kurulum ve Çalıştırma
Yerel Ortam Kurulumu (Hibrit Boru Hattı ve Kıyaslamalar)

Depoyu yerel dizininize kopyalayın:

Bash
git clone [https://github.com/zeynepguney/image-processing-with-deep-learning.git](https://github.com/zeynepguney/image-processing-with-deep-learning.git)
cd image-processing-with-deep-learning
Bir Python sanal ortamı oluşturun ve aktifleştirin:

```bash
python3 -m venv venv
source venv/bin/activate
Gerekli bağımlılıkları yükleyin:

Bash
pip install -r requirements.txt
Temel çizgi veya hibrit doğrulama betiklerini çalıştırın:

Bash
python src/1_baseline_easyocr.py
python src/4_hybrid_ocr.pyd image-processing-with-deep-learning
```
---
### Bulut Ortamı Kurulumu (Qwen2-VL)
colab/qwen_colab_run.ipynb defterini Google Drive'ınıza yükleyin veya doğrudan Google Colab'da açın.

GPU Donanım Hızlandırıcısını seçin (Runtime -> Change runtime type -> T4 GPU).

Sürücünüzü bağlamak, bulut paketlerini yüklemek (requirements-qwen.txt) ve model doğruluğunu takip etmek için hücreleri sırayla çalıştırın:

---

```bash
!python src/6_qwen_vlm_test.py --model Qwen/Qwen2-VL-7B-Instruct --image afis_200.png --device cuda