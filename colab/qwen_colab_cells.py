# Colab — Qwen toplu değerlendirme (200 afiş)

# Hücre 1: GPU + kurulum
# !pip install -q transformers accelerate qwen-vl-utils pillow pandas tqdm

# Hücre 2: Drive
# from google.colab import drive
# drive.mount("/content/drive")
# %cd /content/drive/MyDrive/image-processing-with-deep-learning

# Hücre 3: Tek afiş test (~1-2 dk)
# !python src/6_qwen_vlm_test.py --model Qwen/Qwen2-VL-7B-Instruct --device cuda --image afis_200.png

# Hücre 4: 5 afiş hızlı test (~5-10 dk)
# !python src/7_evaluate_qwen.py --model Qwen/Qwen2-VL-7B-Instruct --device cuda --limit 5

# Hücre 5: Tüm 200 afiş (~1.5-3 saat) — oturum koparsa --resume ile devam
# !python src/7_evaluate_qwen.py --model Qwen/Qwen2-VL-7B-Instruct --device cuda --resume

# Colab'e yüklenecekler:
#   src/qwen_vlm_utils.py
#   src/6_qwen_vlm_test.py      (isteğe bağlı, tek test)
#   src/7_evaluate_qwen.py      (toplu değerlendirme)
#   dataset/afisler/*.png
#   dataset/sentetik_bootcamp_verisi2.csv
