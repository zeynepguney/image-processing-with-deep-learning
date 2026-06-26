import os
import re
import unittest.mock as mock

import easyocr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_ID = "microsoft/Florence-2-base"
DEVICE = "cpu"

TEXT_COLUMNS = ["isim", "unvan", "yayin-gun", "yayin-adi", "tag1", "tag2", "tag3"]
SKIP_COLUMNS = {"konusmaci-görsel", "tyzt-logo", "partner-logo"}
AFIS_DIR = os.path.join(PROJECT_ROOT, "dataset", "afisler")
CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "sentetik_bootcamp_verisi2.csv")


def afis_row_index(image_name: str) -> int:
    match = re.search(r"afis_(\d+)", image_name)
    if not match:
        raise ValueError(f"Geçersiz afiş adı: {image_name}")
    return int(match.group(1)) - 1


def list_afis_images() -> list[str]:
    files = sorted(
        f for f in os.listdir(AFIS_DIR) if f.startswith("afis_") and f.endswith(".png")
    )
    return files


def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    image = ImageEnhance.Sharpness(image).enhance(1.3)
    return image


def quad_to_bbox(quad: list[float], pad: int = 8) -> tuple[int, int, int, int]:
    xs = quad[0::2]
    ys = quad[1::2]
    return (
        max(0, int(min(xs)) - pad),
        max(0, int(min(ys)) - pad),
        int(max(xs)) + pad,
        int(max(ys)) + pad,
    )


def build_vocabulary(df: pd.DataFrame) -> list[str]:
    values: set[str] = set()
    for col in TEXT_COLUMNS:
        for value in df[col].dropna().astype(str):
            values.add(value.strip())
    return sorted(values)


def correct_with_vocabulary(text: str, vocabulary: list[str], cutoff: float = 0.72) -> str:
    cleaned = text.strip().lstrip("</s>").strip()
    if not cleaned:
        return cleaned
    from difflib import get_close_matches

    match = get_close_matches(cleaned, vocabulary, n=1, cutoff=cutoff)
    return match[0] if match else cleaned


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_models():
    with mock.patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    ocr_reader = easyocr.Reader(["tr", "en"], verbose=False)
    return model, processor, ocr_reader


def run_florence_ocr_regions(model, processor, image: Image.Image) -> dict:
    task_prompt = "<OCR_WITH_REGION>"
    inputs = processor(text=task_prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=512,
        do_sample=False,
        num_beams=1,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height),
    )
    return parsed[task_prompt]


def ocr_crop(reader: easyocr.Reader, image: Image.Image, bbox: tuple[int, int, int, int]) -> str:
    crop = np.array(image.crop(bbox))
    results = reader.readtext(crop, detail=0, paragraph=False)
    return " ".join(results).strip()


def extract_hybrid_texts(
    model,
    processor,
    ocr_reader: easyocr.Reader,
    image: Image.Image,
    vocabulary: list[str],
) -> list[str]:
    regions = run_florence_ocr_regions(model, processor, image)
    texts: list[str] = []
    for florence_text, quad in zip(regions["labels"], regions["quad_boxes"]):
        bbox = quad_to_bbox(quad)
        easyocr_text = ocr_crop(ocr_reader, image, bbox)
        corrected = correct_with_vocabulary(easyocr_text or florence_text, vocabulary)
        if corrected:
            texts.append(corrected)
    return texts
