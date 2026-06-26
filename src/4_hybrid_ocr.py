import os

import pandas as pd
from PIL import Image

from hybrid_ocr_utils import (
    CSV_PATH,
    SKIP_COLUMNS,
    afis_row_index,
    build_vocabulary,
    correct_with_vocabulary,
    load_models,
    ocr_crop,
    preprocess_image,
    quad_to_bbox,
    run_florence_ocr_regions,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_NAME = "afis_200.png"

print("Modeller yükleniyor...")
model, processor, ocr_reader = load_models()

image_path = os.path.join(PROJECT_ROOT, "dataset", "afisler", IMAGE_NAME)
df = pd.read_csv(CSV_PATH)
vocabulary = build_vocabulary(df)
row_idx = afis_row_index(IMAGE_NAME)

if not os.path.exists(image_path):
    print(f"HATA: Görsel bulunamadı! Yol: {image_path}")
else:
    image = preprocess_image(Image.open(image_path).convert("RGB"))

    print("\n" + "=" * 50)
    print(f"🔍 TEST EDİLEN GÖRSEL: {IMAGE_NAME} (CSV satır {row_idx + 1})")
    print("=" * 50)

    print("\n1) Florence-2 ile metin bölgeleri tespit ediliyor...")
    regions = run_florence_ocr_regions(model, processor, image)

    print("2) Her bölge EasyOCR ile (Türkçe) okunuyor...")
    hybrid_results = []
    for florence_text, quad in zip(regions["labels"], regions["quad_boxes"]):
        bbox = quad_to_bbox(quad)
        easyocr_text = ocr_crop(ocr_reader, image, bbox)
        corrected = correct_with_vocabulary(easyocr_text or florence_text, vocabulary)
        hybrid_results.append(
            {
                "florence": florence_text.lstrip("</s>").strip(),
                "easyocr": easyocr_text,
                "duzeltilmis": corrected,
            }
        )

    print("\n📊 KARŞILAŞTIRMA (bölge bazlı):")
    print(f"{'Florence-2':<35} | {'EasyOCR (kırpılmış)':<35} | {'Sözlük düzeltmesi'}")
    print("-" * 110)
    for item in hybrid_results:
        print(f"{item['florence']:<35} | {item['easyocr']:<35} | {item['duzeltilmis']}")

    print("\n✅ OLMASI GEREKEN (doğru CSV satırı):")
    ground_truth = df.iloc[row_idx].to_dict()
    for anahtar, deger in ground_truth.items():
        if anahtar not in SKIP_COLUMNS:
            print(f" - {anahtar.upper()}: {deger}")

    print("\n💡 NOT:")
    print(
        "Hibrit yaklaşım: Florence-2 'nerede' sorusunu, EasyOCR 'ne yazıyor' sorusunu yanıtlar. "
        "Türkçe karakterler (ç, ğ, ö, ş, ü) Florence-2'ye göre belirgin iyileşir; "
        "ancak %100 garanti yoktur. CSV sözlüğü ile kalan küçük hatalar (Otomasya→Otomasyon) düzeltilebilir."
    )
    print("=" * 50)
