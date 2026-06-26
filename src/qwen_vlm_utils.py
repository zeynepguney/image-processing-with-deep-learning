import json
import os
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AFIS_DIR = os.path.join(PROJECT_ROOT, "dataset", "afisler")
CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "sentetik_bootcamp_verisi2.csv")

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

JSON_FIELDS = ["isim", "unvan", "yayin_gunu", "yayin_adi", "tag1", "tag2", "tag3"]
CSV_FIELD_MAP = {
    "isim": "isim",
    "unvan": "unvan",
    "yayin_gunu": "yayin-gun",
    "yayin_adi": "yayin-adi",
    "tag1": "tag1",
    "tag2": "tag2",
    "tag3": "tag3",
}

PROMPT = """Bu görsel bir etkinlik afişidir. Görselden şu bilgileri çıkar ve YALNIZCA geçerli JSON formatında yanıt ver.
Türkçe karakterleri (ç, ğ, ı, ö, ş, ü) doğru yaz.

Alanlar:
- isim: sağ alttaki konuşmacının adı soyadı
- unvan: konuşmacının mesleği/unvanı (isim altındaki satır)
- yayin_gunu: üst orta bölgedeki seri ve gün bilgisi (örn. "Deep Learning Serisi Gün 3")
- yayin_adi: orta bölgedeki büyük ana başlık (örn. "Görüntü İşleme ile Otomasyon")
- tag1, tag2, tag3: alttaki üç etiket kutusu, soldan sağa

Örnek format:
{"isim": "...", "unvan": "...", "yayin_gunu": "...", "yayin_adi": "...", "tag1": "...", "tag2": "...", "tag3": "..."}
"""


def afis_row_index(image_name: str) -> int:
    match = re.search(r"afis_(\d+)", image_name)
    if not match:
        raise ValueError(f"Geçersiz afiş adı: {image_name}")
    return int(match.group(1)) - 1


def list_afis_images() -> list[str]:
    return sorted(
        f for f in os.listdir(AFIS_DIR) if f.startswith("afis_") and f.endswith(".png")
    )


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str):
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    print(f"Model yükleniyor: {model_id} ({device}, {dtype})...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    min_pixels = 256 * 28 * 28
    max_pixels = 768 * 28 * 28 if device == "cpu" else 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model.eval()
    return model, processor


def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def run_qwen(model, processor, image_path: str, device: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    if device == "cuda" and hasattr(model, "device"):
        target = model.device
    else:
        target = device
    inputs = {k: v.to(target) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
