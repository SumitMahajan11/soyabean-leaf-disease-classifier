import os
import requests

# Find one image from healthy class as test input
BASE_DIR = r"e:\soyabean12\soyabean11\final_dataset_enhanced_hierarchical"
search_dirs = [
    os.path.join(BASE_DIR, "healthy"),
    os.path.join(BASE_DIR, "diseases", "fungal", "rust"),
]

image_path = None
for d in search_dirs:
    if not os.path.isdir(d):
        continue
    for name in os.listdir(d):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            image_path = os.path.join(d, name)
            break
    if image_path:
        break

if not image_path:
    raise SystemExit("No test image found in configured folders")

print("Using test image:", image_path)

url = "http://127.0.0.1:5000/api/classify"
with open(image_path, "rb") as f:
    files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
    resp = requests.post(url, files=files)

print("Status:", resp.status_code)
print("Body:")
print(resp.text)
