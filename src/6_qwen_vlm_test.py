import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from qwen_vlm_utils import (
    AFIS_DIR,
    CSV_FIELD_MAP,
    CSV_PATH,
    DEFAULT_MODEL,
    afis_row_index,
    extract_json,
    load_model,
    pick_device,
    run_qwen,
)

DEFAULT_IMAGE = "afis_200.png"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or pick_device()
    image_path = os.path.join(AFIS_DIR, args.image)

    if not os.path.exists(image_path):
        print(f"HATA: Görsel bulunamadı: {image_path}")
        return

    df = pd.read_csv(CSV_PATH)
    row_idx = afis_row_index(args.image)
    ground_truth = df.iloc[row_idx]

    model, processor = load_model(args.model, device)
    print(f"\nQwen-VL çalışıyor ({args.image})...")
    raw_output = run_qwen(model, processor, image_path, device)
    parsed = extract_json(raw_output)

    print("\n" + "=" * 50)
    print(f"🔍 TEST EDİLEN GÖRSEL: {args.image} (CSV satır {row_idx + 1})")
    print("=" * 50)
    print("\n🤖 QWEN-VL HAM ÇIKTI:")
    print(raw_output)
    print("\n📦 Ayrıştırılmış JSON:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else "(JSON ayrıştırılamadı)")
    print("\n✅ OLMASI GEREKEN:")
    for json_key, csv_key in CSV_FIELD_MAP.items():
        print(f" - {json_key.upper()}: {ground_truth[csv_key]}")

    if parsed:
        print("\n📊 ALAN KARŞILAŞTIRMA:")
        for json_key, csv_key in CSV_FIELD_MAP.items():
            expected = str(ground_truth[csv_key]).strip()
            predicted = str(parsed.get(json_key, "")).strip()
            mark = "✓" if expected.lower() == predicted.lower() else "✗"
            print(f"  {mark} {json_key}: {predicted!r}  (beklenen: {expected!r})")
    print("=" * 50)


if __name__ == "__main__":
    main()
