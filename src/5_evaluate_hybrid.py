
import argparse
import os
import sys
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from PIL import Image
from tqdm import tqdm

from hybrid_ocr_utils import (
    AFIS_DIR,
    CSV_PATH,
    TEXT_COLUMNS,
    afis_row_index,
    build_vocabulary,
    extract_hybrid_texts,
    list_afis_images,
    load_models,
    preprocess_image,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MATCH_THRESHOLD = 0.85


def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def match_field(expected: str, candidates: list[str]) -> tuple[str, float]:
    best_text, best_score = "", 0.0
    for candidate in candidates:
        score = similarity(expected, candidate)
        if score > best_score:
            best_score = score
            best_text = candidate
    return best_text, best_score


def evaluate_image(
    model,
    processor,
    ocr_reader,
    image_name: str,
    ground_truth: dict,
    vocabulary: list[str],
) -> dict:
    image_path = os.path.join(AFIS_DIR, image_name)
    image = preprocess_image(Image.open(image_path).convert("RGB"))
    extracted = extract_hybrid_texts(model, processor, ocr_reader, image, vocabulary)

    row = {"gorsel": image_name}
    for field in TEXT_COLUMNS:
        expected = str(ground_truth[field]).strip()
        predicted, score = match_field(expected, extracted)
        row[f"{field}_beklenen"] = expected
        row[f"{field}_tahmin"] = predicted
        row[f"{field}_benzerlik"] = round(score, 4)
        row[f"{field}_dogru"] = score >= MATCH_THRESHOLD
    row["tum_alanlar_dogru"] = all(row[f"{f}_dogru"] for f in TEXT_COLUMNS)
    row["dogru_alan_sayisi"] = sum(row[f"{f}_dogru"] for f in TEXT_COLUMNS)
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Hibrit OCR değerlendirmesi")
    parser.add_argument("--limit", type=int, default=None, help="İşlenecek maksimum afiş sayısı")
    parser.add_argument("--start", type=int, default=1, help="Başlangıç afiş numarası (dahil)")
    parser.add_argument("--end", type=int, default=200, help="Bitiş afiş numarası (dahil)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    vocabulary = build_vocabulary(df)

    all_images = list_afis_images()
    selected = [
        img
        for img in all_images
        if args.start <= afis_row_index(img) + 1 <= args.end
    ]
    if args.limit:
        selected = selected[: args.limit]

    print(f"Değerlendirilecek afiş sayısı: {len(selected)}")
    print(f"{MATCH_THRESHOLD:.0%} benzerlik eşiği ile alan doğruluğu hesaplanıyor...\n")

    print("Modeller yükleniyor...")
    model, processor, ocr_reader = load_models()

    results = []
    for image_name in tqdm(selected, desc="Afişler"):
        row_idx = afis_row_index(image_name)
        ground_truth = df.iloc[row_idx].to_dict()
        results.append(
            evaluate_image(model, processor, ocr_reader, image_name, ground_truth, vocabulary)
        )

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "evaluation_hybrid.csv")
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 55)
    print("📊 DEĞERLENDİRME ÖZETİ")
    print("=" * 55)
    print(f"Toplam afiş: {len(results_df)}")
    print(
        f"Tüm alanları doğru afiş: {results_df['tum_alanlar_dogru'].sum()} "
        f"({100 * results_df['tum_alanlar_dogru'].mean():.1f}%)"
    )
    print(f"Ortalama doğru alan / afiş: {results_df['dogru_alan_sayisi'].mean():.2f} / {len(TEXT_COLUMNS)}")
    print()
    print("Alan bazında doğruluk:")
    for field in TEXT_COLUMNS:
        acc = results_df[f"{field}_dogru"].mean()
        avg_sim = results_df[f"{field}_benzerlik"].mean()
        print(f"  {field:12} → %{100 * acc:5.1f} doğru  (ort. benzerlik: {avg_sim:.3f})")
    print()
    print(f"Detaylı sonuçlar: {out_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
