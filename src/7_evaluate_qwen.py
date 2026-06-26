import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from qwen_vlm_utils import (
    AFIS_DIR,
    CSV_FIELD_MAP,
    CSV_PATH,
    DEFAULT_MODEL,
    JSON_FIELDS,
    afis_row_index,
    extract_json,
    list_afis_images,
    load_model,
    pick_device,
    run_qwen,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "evaluation_qwen.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen2-VL toplu değerlendirme")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="evaluation_qwen.csv varsa işlenmiş afişleri atla",
    )
    return parser.parse_args()


def evaluate_one(model, processor, device, image_name, ground_truth) -> dict:
    image_path = os.path.join(AFIS_DIR, image_name)
    raw = run_qwen(model, processor, image_path, device)
    parsed = extract_json(raw)

    row = {"gorsel": image_name, "ham_cikti": raw, "json_ok": parsed is not None}
    for json_key, csv_key in CSV_FIELD_MAP.items():
        expected = str(ground_truth[csv_key]).strip()
        predicted = str(parsed.get(json_key, "")).strip() if parsed else ""
        row[f"{json_key}_beklenen"] = expected
        row[f"{json_key}_tahmin"] = predicted
        row[f"{json_key}_dogru"] = expected.lower() == predicted.lower()

    row["tum_alanlar_dogru"] = all(row[f"{k}_dogru"] for k in JSON_FIELDS)
    row["dogru_alan_sayisi"] = sum(row[f"{k}_dogru"] for k in JSON_FIELDS)
    return row


def print_summary(results_df: pd.DataFrame):
    print("\n" + "=" * 55)
    print("📊 QWEN-VL DEĞERLENDİRME ÖZETİ")
    print("=" * 55)
    print(f"Toplam afiş: {len(results_df)}")
    print(f"JSON ayrıştırma başarısı: {results_df['json_ok'].mean():.1%}")
    print(
        f"Tüm alanları doğru afiş: {results_df['tum_alanlar_dogru'].sum()} "
        f"({100 * results_df['tum_alanlar_dogru'].mean():.1f}%)"
    )
    print(f"Ortalama doğru alan / afiş: {results_df['dogru_alan_sayisi'].mean():.2f} / {len(JSON_FIELDS)}")
    print()
    print("Alan bazında doğruluk:")
    for field in JSON_FIELDS:
        acc = results_df[f"{field}_dogru"].mean()
        print(f"  {field:12} → %{100 * acc:5.1f}")
    print(f"\nDetaylı sonuçlar: {OUTPUT_CSV}")
    print("=" * 55)


def main():
    args = parse_args()
    device = args.device or pick_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    selected = [
        img
        for img in list_afis_images()
        if args.start <= afis_row_index(img) + 1 <= args.end
    ]
    if args.limit:
        selected = selected[: args.limit]

    done = set()
    existing_rows = []
    if args.resume and os.path.exists(OUTPUT_CSV):
        prev = pd.read_csv(OUTPUT_CSV)
        done = set(prev["gorsel"].tolist())
        existing_rows = prev.to_dict("records")
        print(f"Devam modu: {len(done)} afiş zaten işlenmiş, atlanacak.")

    todo = [img for img in selected if img not in done]
    print(f"Değerlendirilecek afiş: {len(todo)} (toplam hedef: {len(selected)})")
    print(f"Model: {args.model} | Cihaz: {device}")
    if device == "cuda" and len(todo) > 0:
        est_min = len(todo) * 1.5  # ~90 sn/afiş ortalama (7B T4)
        print(f"Tahmini süre: ~{est_min:.0f}–{est_min * 1.5:.0f} dakika\n")

    model, processor = load_model(args.model, device)
    results = list(existing_rows)

    for image_name in tqdm(todo, desc="Qwen-VL"):
        row_idx = afis_row_index(image_name)
        ground_truth = df.iloc[row_idx].to_dict()
        results.append(evaluate_one(model, processor, device, image_name, ground_truth))
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    if results:
        print_summary(pd.DataFrame(results))


if __name__ == "__main__":
    main()
