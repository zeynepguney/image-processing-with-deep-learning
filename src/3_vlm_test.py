import os
import re
import unittest.mock as mock

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGE_NAME = "afis_200.png"


def afis_row_index(image_name: str) -> int:
    """afis_01.png -> 0, afis_200.png -> 199"""
    match = re.search(r"afis_(\d+)", image_name)
    if not match:
        raise ValueError(f"Geçersiz afiş adı: {image_name}")
    return int(match.group(1)) - 1


model_id = "microsoft/Florence-2-base"
print(f"{model_id} yükleniyor...")

device = "cpu"
if torch.backends.mps.is_available():
    print("Not: Florence-2-large MPS belleğini aşabilir; CPU kullanılıyor.")


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Florence-2 remote code flash_attn bağımlılığını Mac'te atla."""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


with mock.patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

image_path = os.path.join(PROJECT_ROOT, "dataset", "afisler", IMAGE_NAME)
csv_path = os.path.join(PROJECT_ROOT, "dataset", "sentetik_bootcamp_verisi2.csv")
df = pd.read_csv(csv_path)
row_idx = afis_row_index(IMAGE_NAME)

if not os.path.exists(image_path):
    print(f"HATA: Görsel bulunamadı! Yol: {image_path}")
else:
    image = Image.open(image_path).convert("RGB")

    def run_florence_task(task_prompt: str) -> dict:
        """Florence-2 yalnızca görev etiketlerini (<OCR>, <CAPTION> vb.) kabul eder."""
        inputs = processor(text=task_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            do_sample=False,
            num_beams=1,
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        return processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

    tasks = ["<OCR_WITH_REGION>", "<MORE_DETAILED_CAPTION>"]

    print("\n" + "=" * 50)
    print(f"🔍 TEST EDİLEN GÖRSEL: {IMAGE_NAME} (CSV satır {row_idx + 1})")
    print("=" * 50)

    for task_prompt in tasks:
        print(f"\nVLM çalışıyor ({task_prompt}), lütfen bekleyin...")
        parsed_answer = run_florence_task(task_prompt)
        print(f"\n🤖 VLM (Florence-2) ÇIKTISI — {task_prompt}:")
        print(parsed_answer)

    print("\n✅ OLMASI GEREKEN (Bizim Excel Verimiz):")
    orijinal_veri = df.iloc[row_idx].to_dict()
    for anahtar, deger in orijinal_veri.items():
        if anahtar not in ["konusmaci-görsel", "tyzt-logo", "partner-logo"]:
            print(f" - {anahtar.upper()}: {deger}")
