import easyocr
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

csv_path = os.path.join(PROJECT_ROOT, "dataset", "sentetik_bootcamp_verisi2.csv")
df = pd.read_csv(csv_path)


print("EasyOCR modeli yükleniyor...")
reader = easyocr.Reader(['tr', 'en'])

# Test İçin Sadece İlk Afişi (afis_1.jpg) 
image_path = os.path.join(PROJECT_ROOT, "dataset", "afisler", "afis_01.png")


if not os.path.exists(image_path):
    print(f"HATA: Görsel bulunamadı! Yol: {image_path}")
else:

    okunan_metinler = reader.readtext(image_path, detail=0)
    

    print("\n" + "="*50)
    print("TEST EDİLEN GÖRSEL: afis_01.png")
    print("="*50)
    
    print("\n❌ EASYOCR ÇIKTISI (Düzensiz ve Karmaşık):")
    print(" ".join(okunan_metinler))
    
    print("\n✅ OLMASI GEREKEN (Bizim Excel Verimiz):")
    orijinal_veri = df.iloc[0].to_dict() # İlk satırı al
    for anahtar, deger in orijinal_veri.items():
        if anahtar not in ["konusmaci-görsel", "tyzt-logo", "partner-logo"]:
            print(f" - {anahtar.upper()}: {deger}")